from ultralytics import YOLO

# Load your trained YOLOv11 model
model = YOLO("runs/detect/yolo11_4head_4ca_carafe/weights/best.pt")

# Inference on validation images
results = model.predict(
    source="dataset_canada_7_3/images/val",  # path to images
    imgsz=640,                    # image size
    conf=0.25,                    # confidence threshold
    save=False,                   # do not save images
    save_txt=True,                # save bounding boxes as .txt
    save_conf=True                # include confidence in .txt
)
