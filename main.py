from multiprocessing import freeze_support
from ultralytics import YOLO

# Toggle this:
TRAIN_MODE = False  # True = train, False = skip training and run inference

def main() -> None:
    if TRAIN_MODE:
        model = YOLO("yolov8s.pt")
        model.train(
            data="Tennis Ball Detection FINAL.v4i.yolov8/data.yaml",
            epochs=50,
            imgsz=640,
            project="runs/detect",
            name = "tennis_yolov8s",
            # workers=0,  # uncomment if multiprocessing still causes issues
        )
    else:
        model = YOLO("runs/detect/runs/detect/tennis_yolov8s/weights/best.pt")  #with the small model 
        model.model.names[0] = "Tennis Ball"  # rename class id 0 label
        model.predict(source="BallBlazer Pro Professional-Grade Tennis Ball Collection #smalleyes #sportsequipment #funny.mp4", show=True, save=True, conf=0.25)

if __name__ == "__main__":
    freeze_support()  # Windows-safe multiprocessing startup
    main()

