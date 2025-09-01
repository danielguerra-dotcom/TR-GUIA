# deteccion_webcam_todo.py
# Detecta TODAS las clases que YOLOv8n conoce (COCO: ~80 clases) sin filtros.
import argparse
import time
import cv2
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Índice de cámara (0, 1, 2...)")
    ap.add_argument("--imgsz", type=int, default=512, help="Tamaño de entrada del modelo")
    ap.add_argument("--conf", type=float, default=0.6, help="Umbral de confianza")
    ap.add_argument("--device", type=str, default=None, help="'cuda' o 'cpu' (auto por defecto)")
    args = ap.parse_args()

    model = YOLO("yolov8n.pt")  # se descarga la 1ª vez

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)  # CAP_DSHOW acelera en Windows
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {args.cam}")

    # Forzar resolución de cámara para ganar FPS (ajusta si quieres)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        t0, frames = time.time(), 0
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
                device=args.device,
                verbose=False
            )

            # Dibujar TODAS las detecciones (sin filtro de clases)
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
                        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # FPS/latencia
            infer_ms = (time.time() - t_start) * 1000
            frames += 1
            elapsed = max(1e-6, time.time() - t0)
            fps = frames / elapsed
            cv2.putText(frame, f"FPS: {fps:.1f} | latencia: {infer_ms:.1f} ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("YOLOv8n - Todas las clases (ESC para salir)", frame)
            if cv2.waitKey(1) == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()