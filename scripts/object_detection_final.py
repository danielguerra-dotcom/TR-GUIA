import argparse
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from typing import Optional

CANDIDATE_WEIGHTS = ["yolo11n.pt", "yolov11n.pt", "yolov10n.pt", "yolov8n.pt"]

def load_best_model(user_model: Optional[str]):
    if user_model:
        return YOLO(user_model)
    last_err = None
    for w in CANDIDATE_WEIGHTS:
        try:
            return YOLO(w)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load any model: {last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--no_preview", action="store_true")
    args = ap.parse_args()

    model = load_best_model(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = (device == "cuda")
    try:
        model.model.half() if half else model.model.float()
    except Exception:
        pass

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.cam}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 25)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = model.predict(source=dummy_frame, imgsz=args.imgsz, conf=args.conf, iou=0.5, device=device, verbose=False)

    t0, frames = time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t_start = time.time()
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=0.5,
                device=device,
                half=half,
                verbose=False
            )

            if not args.no_preview:
                if results and len(results):
                    r = results[0]
                    names = r.names
                    if r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            cls_id = int(box.cls.item())
                            cls_name = names[cls_id]
                            score = float(box.conf.item())
                            x1, y1, x2, y2 = map(int, box.xyxy.squeeze().tolist())
                            label = f"{cls_name} {score:.2f}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
                            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

                infer_ms = (time.time() - t_start) * 1000
                frames += 1
                elapsed = max(1e-6, time.time() - t0)
                fps = frames / elapsed
                cv2.putText(frame, f"FPS: {fps:.1f} | latency: {infer_ms:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("Object Detection Final (ESC to exit)", frame)
                if cv2.waitKey(1) == 27:
                    break

    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()