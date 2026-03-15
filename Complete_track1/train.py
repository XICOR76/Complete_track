from ultralytics import YOLO

model = YOLO("yolo11s-seg.pt")

model.train(
    data="complete_track.v1i.yolov11/data.yaml",  # <-- FIXED PATH
    epochs=150,
    imgsz=640,
    batch=8,
    patience=30
)