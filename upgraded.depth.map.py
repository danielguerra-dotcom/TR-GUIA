# assist_nav_yolo_depth_range.py
# Detección + profundidad + voz, anunciando SOLO objetos dentro de un rango (p. ej., 2 m).
# Teclas:
#   [V] alterna vista (original ↔ procesada)
#   [C] calibrar  (apunta al objeto a ~RANGO metros en el centro y pulsa C)
#   [-] baja rango en 0.25 m   |   [=] sube rango en 0.25 m
#   [ESC] salir
#
# Requisitos:
#   pip install ultralytics opencv-python torch torchvision timm pyttsx3 numpy

import argparse, time
import cv2, numpy as np, torch, torch.nn.functional as F
from ultralytics import YOLO
import pyttsx3
from collections import defaultdict

# Modelos YOLO candidatos (carga el más eficiente disponible)
CANDIDATE_WEIGHTS = ["yolo11n.pt", "yolov11n.pt", "yolov10n.pt", "yolov8n.pt"]

ANNOUNCE_COOLDOWN_S = 2.0     # enfriar por etiqueta-lado para no hablar en bucle
SIDE_BOUNDS = (1/3, 2/3)      # corte en tercios: izquierda/centro/derecha
CENTER_CALIB_BOX = 0.2        # tamaño de la ventana central (20% ancho/alto) para calibrar

def load_best_yolo(custom=None):
    if custom:
        return YOLO(custom)
    last_err = None
    for w in CANDIDATE_WEIGHTS:
        try:
            return YOLO(w)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"No pude cargar ningún YOLO ({CANDIDATE_WEIGHTS}). Último error: {last_err}")

def load_midas(device):
    repo = "intel-isl/MiDaS"
    model = torch.hub.load(repo, "MiDaS_small", trust_repo=True)  # rápido
    model.to(device).eval()
    transforms = torch.hub.load(repo, "transforms", trust_repo=True)
    transform = transforms.small_transform
    return model, transform

def colorize_depth(depth):
    d = depth.astype(np.float32)
    d -= d.min()
    d /= (d.max() + 1e-6)
    return cv2.applyColorMap((d*255).astype(np.uint8), cv2.COLORMAP_TURBO)

def median_depth_in_box(depth, x1, y1, x2, y2, shrink=0.15):
    h, w = depth.shape
    dx = int((x2 - x1) * shrink)
    dy = int((y2 - y1) * shrink)
    xa = max(0, x1 + dx); ya = max(0, y1 + dy)
    xb = min(w-1, x2 - dx); yb = min(h-1, y2 - dy)
    if xb <= xa or yb <= ya:
        return None
    patch = depth[ya:yb, xa:xb]
    if patch.size < 16:
        return None
    return float(np.median(patch))

def side_from_center(xc, w):
    nx = xc / max(1, w)
    if nx < SIDE_BOUNDS[0]:   return "left"
    if nx > SIDE_BOUNDS[1]:   return "right"
    return "center"

def depth_to_meters(depth_val, calib_k):
    """
    Conversión simple a metros asumiendo proporcionalidad inversa:
      meters ≈ calib_k / depth_val
    donde calib_k se fija con una calibración: calib_k = RANGO(m) * depth_ref (en el centro).
    """
    if calib_k is None or depth_val is None or depth_val <= 0:
        return None
    return float(calib_k / max(depth_val, 1e-6))

def calibrate_k(depth_map, range_m):
    """ Calcula calib_k usando la mediana en una caja central y el rango elegido. """
    H, W = depth_map.shape
    sw = int(W * CENTER_CALIB_BOX / 2.0)
    sh = int(H * CENTER_CALIB_BOX / 2.0)
    cx, cy = W // 2, H // 2
    xa, ya = max(0, cx - sw), max(0, cy - sh)
    xb, yb = min(W, cx + sw), min(H, cy + sh)
    ref = np.median(depth_map[ya:yb, xa:xb])
    if ref <= 0 or np.isnan(ref):
        return None
    return float(range_m * ref)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Índice de cámara")
    ap.add_argument("--imgsz", type=int, default=512, help="Resolución de entrada del detector")
    ap.add_argument("--conf", type=float, default=0.5, help="Umbral de confianza YOLO")
    ap.add_argument("--model", type=str, default=None, help="Ruta/peso YOLO (opcional)")
    ap.add_argument("--range_m", type=float, default=2.0, help="Rango de aviso en metros")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Modelos
    yolo = load_best_yolo(args.model)
    midas, midas_tf = load_midas(device)

    # Voz
    tts = pyttsx3.init()
    tts.setProperty("rate", 175)

    # Cámara
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {args.cam}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 25)

    last_said = defaultdict(lambda: 0.0)
    show_processed = True
    calib_k = None  # factor de calibración para pasar de profundidad relativa a metros
    range_m = float(args.range_m)

    t0, frames = time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]

            # Profundidad
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = midas_tf(rgb).to(device)
            with torch.no_grad():
                d_pred = midas(inp)
                d_pred = F.interpolate(d_pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False)
                depth = d_pred.squeeze().cpu().numpy()  # mayor valor ≈ más cerca (relativo)

            # Detección
            t_det = time.time()
            results = yolo.predict(source=frame, imgsz=args.imgsz, conf=args.conf, iou=0.5,
                                   device=device, verbose=False)
            det_ms = (time.time() - t_det) * 1000

            to_announce = []  # (label, side, dist_m, bbox)
            if results and len(results):
                r = results[0]
                names = r.names
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls.item())
                        label = names[cls_id]
                        conf = float(box.conf.item())
                        x1, y1, x2, y2 = map(int, box.xyxy.squeeze().tolist())
                        md = median_depth_in_box(depth, x1, y1, x2, y2)
                        if md is None:
                            continue
                        dist_m = depth_to_meters(md, calib_k) if calib_k is not None else None
                        # Si no calibrado aún, intentamos una aproximación basada en percentil (fallback visual)
                        if dist_m is None:
                            dist_m = None  # no anunciar todavía hasta calibrar
                        if dist_m is not None and dist_m <= range_m:
                            side = side_from_center((x1 + x2) / 2.0, w)
                            to_announce.append((label, side, dist_m, (x1, y1, x2, y2, conf)))

            # Hablar (con cooldown por etiqueta-lado)
            now = time.time()
            for label, side, dist_m, _ in to_announce:
                key = f"{label}_{side}"
                if now - last_said[key] >= ANNOUNCE_COOLDOWN_S:
                    phrase = f"Caution, {label} on the {side}, distance {dist_m:.1f} meters"
                    try:
                        tts.say(phrase)
                        tts.runAndWait()
                    except Exception:
                        pass
                    last_said[key] = now

            # Visualización
            depth_color = colorize_depth(depth)
            vis = frame.copy()
            if show_processed:
                vis = cv2.addWeighted(frame, 0.45, depth_color, 0.55, 0)

            # Dibujo de detecciones dentro de rango
            for label, side, dist_m, (x1, y1, x2, y2, conf) in to_announce:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(vis, f"{label} {conf:.2f} | {dist_m:.1f} m | {side}",
                            (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            # Info HUD
            frames += 1
            fps = frames / max(1e-6, (time.time() - t0))
            calib_txt = "OK" if calib_k is not None else "NOT SET"
            mode_txt = "PROCESSED" if show_processed else "ORIGINAL"
            cv2.putText(vis, f"{mode_txt} | FPS~{fps:.1f} | det {det_ms:.0f} ms | Range={range_m:.2f}m | Calib={calib_txt}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2, cv2.LINE_AA)

            # Caja central para ayudar a calibrar
            if calib_k is None:
                cx, cy = w//2, h//2
                sw = int(w * CENTER_CALIB_BOX / 2.0)
                sh = int(h * CENTER_CALIB_BOX / 2.0)
                cv2.rectangle(vis, (cx - sw, cy - sh), (cx + sw, cy + sh), (0, 200, 255), 2)
                cv2.putText(vis, "Point an object at ~Range and press C to calibrate",
                            (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow("Assistive nav (V: vista, C: calibrar, -/=: rango, ESC: salir)", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            elif k == ord('v') or k == ord('V'):
                show_processed = not show_processed
            elif k == ord('c') or k == ord('C'):
                ck = calibrate_k(depth, range_m)
                if ck is not None and ck > 0:
                    calib_k = ck
            elif k in (ord('-'), ord('_')):
                range_m = max(0.5, range_m - 0.25)
            elif k in (ord('='), ord('+')):
                range_m = min(10.0, range_m + 0.25)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
