from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11-seg.yaml")

model.train(
    data="dataset/data.yaml",
    epochs=1,
    imgsz=640,
    batch=32,
    patience=50,
    device="cpu",
    workers=2,
    seed=42,
    project="runs/BUSBRA",
    name="yolo11n_seg_baseline",
    plots=True
)