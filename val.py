from ultralytics import YOLO

model = YOLO("runs/detect/yolo11l_scratch_canada_7_3/weights/best.pt")  # Or your final .pt path

metrics = model.val(
    data="dataset.yaml",
    imgsz=640,
    batch=16
)
