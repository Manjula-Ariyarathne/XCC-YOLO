from ultralytics import YOLO

model = YOLO("cfg/models/11/yolo11l.yaml")

results = model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    name="yolo11l_scratch_canada_7_3",
    pretrained=False
)
