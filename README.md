# XCC-YOLO: An Enhanced YOLOv11 Model for Mining Detection

This repository presents **XCC-YOLO**, an enhanced version of [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics), customized for detecting surface mining-related features such as **open-pit mines** and **tailings ponds** from high-resolution remote sensing images.  
The improvements aim to boost detection performance by integrating attention mechanisms and advanced upsampling techniques.

---

## About This Repository

This repository is based on the official [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) codebase, licensed under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).  
All original credits and license terms are preserved. Modifications have been made solely for **academic research purposes**, focusing on remote sensing-based mining activity detection.

---

## Improvements Made

We propose the **XCC-YOLO** model with the following key architectural enhancements:

- **Fourth Detection Head**: Added to improve multi-scale detection performance.
- **Channel Attention (CA) Modules**: Four CA modules integrated to enhance spatial feature learning.
- **CARAFE Upsampling**: Replaces standard upsampling layers with **Content-Aware ReAssembly of FEatures (CARAFE)** for finer spatial detail recovery.

---

## Datasets

XCC-YOLO was trained and evaluated on the following datasets:

- **Canada Mining Dataset** *(custom-built)*: A comprehensive dataset covering mining regions across Canada, curated for object detection tasks in high-resolution satellite imagery.  
  *→ Will be made public soon.*
  
- **NWPU VHR-10 Dataset**: A widely used benchmark dataset for object detection in remote sensing, consisting of 800 images annotated with 10 object classes. Spatial resolutions range from 0.08 m to 2 m.  
  [[Dataset Link]](https://github.com/chaozhong2010/VHR-10_dataset_coco)

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Manjula-Ariyarathne/XCC-YOLO.git
cd XCC-YOLO
pip install -r requirements.txt
```

## Training
```bash
python train.py
```

## Validation
```bash
python val.py
```