from multiprocessing import freeze_support
import cv2
from ultralytics import YOLO

# Toggle this:
TRAIN_MODE = False  # True = train custom tennis-ball model, False = run inference

VIDEO_PATH = "6 UTR VS 10 UTR+ 🔥 #TENNIS.mp4"


def main() -> None:
    if TRAIN_MODE:
        model = YOLO("yolov8n.pt")
        model.train(
            data="Tennis Ball Detection FINAL.v4i.yolov8/data.yaml",
            epochs=50,
            imgsz=640,
            project="runs/detect",
            name="tennis_yolov8n",
            # workers=0,  # uncomment if multiprocessing still causes issues
        )
    else:
        # COCO model for person (class 0)
        person_model = YOLO("yolov8n.pt")

        # Your fine-tuned model for tennis ball
        ball_model = YOLO("runs/detect/runs/detect/tennis_yolov8n/weights/best.pt")
        ball_model.model.names[0] = "Tennis Ball"

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            person_res = person_model.predict(frame, classes=[0], conf=0.30, verbose=False)[0]
            ball_res = ball_model.predict(frame, conf=0.30, verbose=False)[0]

            # Draw persons (green)
            for b in person_res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

            # Draw tennis balls (orange)
            for b in ball_res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, f"Tennis Ball {conf:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cv2.imshow("Person + Tennis Ball", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    freeze_support()
    main()

