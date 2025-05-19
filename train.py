from ultralytics import YOLO

model = YOLO("cfg/models/11/yolo11_4head_4ca_carafe.yaml") # yolo11_ca, yolo11l, yolo11_mbh, yolo11_3ca, yolo11_convca, yolo11_aifi, yolo11_3ca_aifi, yolo11_3ca_aifi_mbh, yolo11_c3tr_repeat_2, yolo11_c3tr_repeat_6
# yolo11_3ca_aifi_repeat_4, yolo11_3ca_1c3tr_repeat_6, yolo11_3ca_1c3tr_repeat_4, yolo11_3ca_1aifi_1c3tr_repeat_4, yolo11_cbam, yolo11_3cbam, yolo11_cbam_c3tr_repeat_2, yolo11_cbam_1c3tr_repeat_4, yolo11_cbam_c3tr_repeat_4
# yolo11_cbam_2ca, yolo11_mftayolo, yolo11_mftayolo_full, yolo11_mftayolo_cbam, yolo11l_max_c_1024, yolo11_cbam_mc_1024, yolo11_mbh_cbam, yolo11_cbam_4mf_3ta, yolo11_cbam_4mf_3ca, yolo11_cbam_2ca_1c3tr_repeat_4, 
# yolo11_cbam_2ca_1c3tr_repeat_3, yolo11_carafe, yolo11_3ca_1c3tr_repeat_4_carafe, yolo11_cbam_carafe, yolo11_3ca_carafe, yolo11_3ca_aifi_carafe, yolo11_cbam_c3tr_repeat_2_carafe, yolo11_3cbam_head, yolo11_rtdetrdecoder
# yolo11_carafe_new, yolo11_3ca_carafe_new, yolo11_3ca_1c3tr_repeat_4_carafe_new, yolo11_3ca_carafe_512, yolo11, yolo11_ghostconv, yolo11_4head, yolo11_4head_4ca_carafe, yolo11_4head_4ca_carafe_ghostconv

results = model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    name="yolo11_4head_4ca_carafe",
    pretrained=False
    # lr0=0.001,
    # optimizer="Adam",
    # patience=15,
    # box=0.05,   # box loss weight
    # cls=0.25,   # class loss weight
    # dfl=0.5 
)




########## resume #######################

# from ultralytics import YOLO

# # Resume from last checkpoint
# model = YOLO("runs/detect/yolo11_4head_4ca_carafe/weights/last.pt")

# results = model.train(
#     resume=True               # ensure resume flag is enabled
# )