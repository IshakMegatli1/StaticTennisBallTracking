from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='Tennis Ball Detection FINAL.v4i.yolov8/data.yaml', epochs=50, imgsz=640)