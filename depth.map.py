# depth_webcam.py
# Depth map en color (MiDaS) con dos perfiles: fast (MiDaS_small) / quality (DPT_Hybrid).
# Teclas: [v] alterna RGB/Depth | [s] guarda RGB y Depth | [ESC] salir

import argparse, time
import cv2, numpy as np, torch
import torch.nn.functional as F

def load_midas(profile: str, device: str):
    repo = "intel-isl/MiDaS"
    if profile == "fast":
        model = torch.hub.load(repo, "MiDaS_small", trust_repo=True)
    else:
        model = torch.hub.load(repo, "DPT_Hybrid", trust_repo=True)
    model.to(device).eval()

    transforms = torch.hub.load(repo, "transforms", trust_repo=True)
    transform = transforms.small_transform if profile == "fast" else transforms.dpt_transform
    return model, transform

def colorize(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    d -= d.min()
    d /= (d.max() + 1e-6)
    d8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_MAGMA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--profile", choices=["fast","quality"], default="fast")
    ap.add_argument("--imgsz", type=int, default=512)   # 320–512 recomendado
    ap.add_argument("--cap_w", type=int, default=640)
    ap.add_argument("--cap_h", type=int, default=480)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Cargando MiDaS ({'MiDaS_small' if args.profile=='fast' else 'DPT_Hybrid'}) en {device}...")
    model, transform = load_midas(args.profile, device)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {args.cam}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_h)
    cap.set(cv2.CAP_PROP_FPS, 25)

    show_depth = True
    saved = 0
    t0, frames = time.time(), 0

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            # Preproc (MiDaS transform incluye resize/normalización)
            inp = transform(cv2.resize(rgb, (args.imgsz, args.imgsz))).to(device)

            t = time.time()
            with torch.no_grad():
                pred = model(inp)  # [1,H',W']
                pred = F.interpolate(pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False)
                depth = pred.squeeze().cpu().numpy()  # [H,W]
            depth_color = colorize(depth)

            vis = depth_color if show_depth else bgr

            # FPS/latencia
            infer_ms = (time.time() - t) * 1000
            frames += 1
            fps = frames / max(1e-6, (time.time() - t0))
            cv2.putText(vis, f"{args.profile.upper()} | {fps:.1f} FPS | {infer_ms:.1f} ms",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow("Depth (v: alternar, s: guardar, ESC: salir)", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            elif k == ord('v'):
                show_depth = not show_depth
            elif k == ord('s'):
                cv2.imwrite(f"rgb_{saved:03d}.png", bgr)
                cv2.imwrite(f"depth_{saved:03d}.png", depth_color)
                print(f"[INFO] Guardado rgb_{saved:03d}.png y depth_{saved:03d}.png")
                saved += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

