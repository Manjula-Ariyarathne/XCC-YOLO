from ultralytics import YOLO

model = YOLO("runs/detect/yolo11l_scratch_canada_7_3/weights/best.pt")  # Or your final .pt path

metrics = model.val(
    data="dataset.yaml",
    imgsz=640,
    batch=16
)

# Print key metrics
print(f"Precision:  {metrics.box.precision:.4f}")
print(f"Recall:     {metrics.box.recall:.4f}")
print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")
