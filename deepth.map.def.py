# detect_yolo_with_depth_autorange.py
# Detección de objetos + mapa de profundidad con CALIBRACIÓN AUTOMÁTICA (sin teclas extra).
# Muestra SOLO objetos cuya distancia estimada ≤ 3 m (por defecto).
# ÚNICA TECLA: [V] alterna vista (ORIGINAL ↔ PROCESADA con profundidad y cajas). [ESC] salir.
#
# Cómo funciona la distancia sin intervenir:
#   - MiDaS da profundidad RELATIVA (valores mayores ≈ más cerca).
#   - En cada frame estimamos dos puntos de referencia (percentiles globales del mapa):
#       * d_near = P95 (zonas más cercanas de la escena)
#       * d_far  = P20 (fondo típico)
#     y resolvemos m ≈ a*(1/d) + b forzando: d_near→m_near (0.5 m) y d_far→m_far (5.0 m).
#   - Suavizamos a,b con un EMA para estabilidad. Es heurístico pero completamente automático.
#
# Requisitos:
#   pip install ultralytics opencv-python torch torchvision timm numpy

import argparse, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# —— Configuración general ——
CANDIDATE_WEIGHTS = ["yolo11n.pt", "yolov11n.pt", "yolov10n.pt", "yolov8n.pt"]
SHRINK_BOX = 0.18            # recorta bbox para evitar bordes ruidosos al medir profundidad
DEPTH_BLUR_SIGMA = 1.0       # suavizado del mapa de profundidad
RANGE_MAX_M = 3.0            # rango máximo a mostrar (en metros)
AUTO_NEAR_M = 0.5            # ancla "cerca" (metros) para la calibración automática
AUTO_FAR_M  = 5.0            # ancla "lejos" (metros) para la calibración automática
EMA_ALPHA   = 0.12           # suavizado temporal de a,b (0..1), mayor = responde más rápido
VIEW_BLEND  = 0.55           # mezcla RGB/depth en vista procesada

def load_best_yolo(custom=None):
    if custom:
        return YOLO(custom)
    last_err = None
    for w in CANDIDATE_WEIGHTS:
        try:
            return YOLO(w)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"No pude cargar YOLO ({CANDIDATE_WEIGHTS}). Último error: {last_err}")

def load_midas(device):
    repo = "intel-isl/MiDaS"
    model = torch.hub.load(repo, "MiDaS_small", trust_repo=True)  # rápido; cambiar a DPT_Hybrid en PC potente
    model.to(device).eval()
    transforms = torch.hub.load(repo, "transforms", trust_repo=True)
    transform = transforms.small_transform
    return model, transform

def colorize_depth(depth):
    d = depth.astype(np.float32)
    d -= d.min()
    d /= (d.max() + 1e-6)
    return cv2.applyColorMap((d*255).astype(np.uint8), cv2.COLORMAP_TURBO)

def robust_depth_in_box(depth, x1, y1, x2, y2, shrink=SHRINK_BOX, pct=90):
    H, W = depth.shape
    dx = int((x2 - x1) * shrink)
    dy = int((y2 - y1) * shrink)
    xa = max(0, x1 + dx); ya = max(0, y1 + dy)
    xb = min(W, x2 - dx);  yb = min(H, y2 - dy)
    if xb <= xa or yb <= ya:
        return None
    patch = depth[ya:yb, xa:xb].astype(np.float32)
    if patch.size < 32:
        return None
    return float(np.percentile(patch, pct))  # percentil alto ≈ zona más cercana del objeto

class AutoDepthScaler:
    """
    Ajusta m ≈ a*(1/d) + b automáticamente usando dos percentiles del mapa:
      d_near=P95 → AUTO_NEAR_M
      d_far =P20 → AUTO_FAR_M
    Suaviza a,b con EMA para estabilidad.
    """
    def __init__(self, m_near=AUTO_NEAR_M, m_far=AUTO_FAR_M, ema_alpha=EMA_ALPHA):
        self.m_near = m_near
        self.m_far  = m_far
        self.a = None
        self.b = None
        self.ema_alpha = ema_alpha

    def update(self, depth):
        if depth is None or depth.size == 0:
            return
        d_near = np.percentile(depth, 95)
        d_far  = np.percentile(depth, 20)
        if d_near <= 0 or d_far <= 0 or abs(d_near - d_far) < 1e-6:
            return
        u1, u2 = 1.0/max(d_near,1e-6), 1.0/max(d_far,1e-6)
        # Resolver a,b con:
        #   m_near = a*u1 + b
        #   m_far  = a*u2 + b
        denom = (u1 - u2)
        if abs(denom) < 1e-9:
            return
        a_new = (self.m_near - self.m_far) / denom
        b_new = self.m_near - a_new * u1
        # EMA
        if self.a is None or self.b is None:
            self.a, self.b = a_new, b_new
        else:
            k = self.ema_alpha
            self.a = (1-k)*self.a + k*a_new
            self.b = (1-k)*self.b + k*b_new

    def to_meters(self, d_val):
        if d_val is None or d_val <= 0 or self.a is None or self.b is None:
            return None
        return float(self.a * (1.0/max(d_val,1e-6)) + self.b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Índice de cámara")
    ap.add_argument("--imgsz", type=int, default=512, help="Resolución detector (416–640 recomendado)")
    ap.add_argument("--conf", type=float, default=0.5, help="Umbral de confianza YOLO")
    ap.add_argument("--model", type=str, default=None, help="Ruta/peso YOLO (opcional)")
    ap.add_argument("--range_m", type=float, default=RANGE_MAX_M, help="Rango máximo a mostrar (m)")
    ap.add_argument("--no_preview", action="store_true", help="Sin ventana (solo logging)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = load_best_yolo(args.model)
    midas, midas_tf = load_midas(device)
    scaler = AutoDepthScaler(m_near=AUTO_NEAR_M, m_far=AUTO_FAR_M, ema_alpha=EMA_ALPHA)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {args.cam}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 25)

    show_processed = True
    range_m = float(args.range_m)

    t0, frames = time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]

            # —— Profundidad (MiDaS) ——
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = midas_tf(rgb).to(device)
            with torch.no_grad():
                d_pred = midas(inp)
                d_pred = F.interpolate(d_pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False)
                depth = d_pred.squeeze().cpu().numpy()
            if DEPTH_BLUR_SIGMA > 0:
                depth = cv2.GaussianBlur(depth, (0,0), sigmaX=DEPTH_BLUR_SIGMA)

            # Actualiza calibración automática (a,b)
            scaler.update(depth)

            # —— Detección (YOLO) ——
            t_det = time.time()
            results = yolo.predict(source=frame, imgsz=args.imgsz, conf=args.conf, iou=0.5,
                                   device=device, verbose=False)
            det_ms = (time.time() - t_det) * 1000

            # Filtrar por rango en metros
            kept = []  # (label, dist_m, (x1,y1,x2,y2,conf))
            if results and len(results):
                r = results[0]
                names = r.names
                if r.boxes is not None and len(r.boxes) > 0:
                    for b in r.boxes:
                        cls_id = int(b.cls.item())
                        label = names[cls_id]
                        conf  = float(b.conf.item())
                        x1, y1, x2, y2 = map(int, b.xyxy.squeeze().tolist())
                        d_box = robust_depth_in_box(depth, x1, y1, x2, y2, pct=90)
                        dist_m = scaler.to_meters(d_box)
                        if dist_m is not None and 0.0 < dist_m <= range_m:
                            kept.append((label, dist_m, (x1,y1,x2,y2,conf)))

            # —— Visualización ——
            depth_color = colorize_depth(depth)
            vis = cv2.addWeighted(frame, 1.0 - VIEW_BLEND, depth_color, VIEW_BLEND, 0) if show_processed else frame.copy()

            for label, dist_m, (x1, y1, x2, y2, conf) in kept:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(vis, f"{label} {conf:.2f} | {dist_m:.2f} m",
                            (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            # HUD
            frames += 1
            fps = frames / max(1e-6, (time.time() - t0))
            mode_txt = "PROCESADA" if show_processed else "ORIGINAL"
            cal_status = "OK" if (scaler.a is not None and scaler.b is not None) else "AUTO..."
            hud = f"{mode_txt} | FPS~{fps:.1f} | det {det_ms:.0f} ms | Range={range_m:.1f} m | Calib={cal_status}"
            cv2.putText(vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2, cv2.LINE_AA)

            if not args.no_preview:
                cv2.imshow("Detección + Profundidad (V alterna, ESC salir)", vis)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
                elif k in (ord('v'), ord('V')):
                    show_processed = not show_processed
            else:
                time.sleep(0.01)

    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
