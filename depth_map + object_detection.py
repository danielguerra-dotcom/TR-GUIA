import argparse, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

YOLO_CANDIDATES = ["yolo11n.pt", "yolov11n.pt", "yolov10n.pt", "yolov8n.pt"]

def load_best_yolo(custom=None):
    if custom:
        return YOLO(custom)
    last_err = None
    for w in YOLO_CANDIDATES:
        try:
            return YOLO(w)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load any model: {last_err}")

def load_midas(profile: str, device: str):
    repo = "intel-isl/MiDaS"
    if profile == "quality":
        model = torch.hub.load(repo, "DPT_Hybrid", trust_repo=True)
        transforms = torch.hub.load(repo, "transforms", trust_repo=True).dpt_transform
    else:
        model = torch.hub.load(repo, "MiDaS_small", trust_repo=True)
        transforms = torch.hub.load(repo, "transforms", trust_repo=True).small_transform
    model.to(device).eval()
    return model, transforms

def colorize_depth(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    d -= d.min()
    d /= (d.max() + 1e-6)
    return cv2.applyColorMap((d*255).astype(np.uint8), cv2.COLORMAP_TURBO)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--profile", choices=["fast","quality"], default="fast")
    ap.add_argument("--cap_w", type=int, default=640)
    ap.add_argument("--cap_h", type=int, default=480)
    ap.add_argument("--blend", type=float, default=0.55)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = load_best_yolo(args.model)
    midas, midas_tf = load_midas(args.profile, device)

    try:
        if device == "cuda":
            yolo.model.half()
    except Exception:
        pass

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.cam}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_h)
    cap.set(cv2.CAP_PROP_FPS, 25)

    show_processed = True
    t0, frames = time.time(), 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = midas_tf(rgb).to(device)
            with torch.no_grad():
                pred = midas(inp)
                pred = F.interpolate(pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False)
                depth = pred.squeeze().cpu().numpy()
            depth_vis = colorize_depth(depth)

            t_det = time.time()
            results = yolo.predict(source=frame, imgsz=args.imgsz, conf=args.conf, iou=0.5, device=device, verbose=False)
            det_ms = (time.time() - t_det) * 1000

            vis = frame.copy()
            if show_processed:
                vis = cv2.addWeighted(frame, 1.0 - args.blend, depth_vis, args.blend, 0)

            if results and len(results):
                r = results[0]
                names = r.names
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy.squeeze().tolist())
                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        label = f"{names[cls_id]} {conf:.2f}"
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0,255,0), -1)
                        cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

            frames += 1
            fps = frames / max(1e-6, (time.time() - t0))
            mode = "PROCESSED" if show_processed else "ORIGINAL"
            cv2.putText(vis, f"{mode} | FPS~{fps:.1f} | det {det_ms:.0f} ms | Depth: {args.profile}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow("Depth Map + Object Detection", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k in (ord('v'), ord('V')):
                show_processed = not show_processed

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()