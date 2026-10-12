# cnn_summer_yolov9_2cls_v14.py
#
# Purpose: Train YOLOv9m on summer tiles, 2 classes
# Version 14: YOLOv9m architecture comparison to YOLOv8m and RT-DETR
# YOLOv9 introduces GELAN (Generalized Efficient Layer Aggregation Network)
# improves gradient flow through the network — potentially better feature
# extraction for small objects compared to YOLOv8m
# Uses same best settings as YOLOv8 v6: 200 epochs, v4 augmentation
# Pretrained weights: COCO (yolov9m.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_yolov9_2cls_v14/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_yolov9_2cls_v14"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO("yolov9m.pt")
    model.train(
        data          = f"{DATA_BASE}/fold{fold}/dataset.yaml",
        epochs        = 200,
        imgsz         = 640,
        batch         = 16,
        workers       = 8,
        device        = 0,
        project       = OUTPUT_BASE,
        name          = f"fold{fold}",
        exist_ok      = True,
        patience      = 50,
        save          = True,
        plots         = True,
        seed          = 42,
        flipud        = 0.5,
        mosaic        = 0.0,
        copy_paste    = 0.3,
        warmup_epochs = 5,
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")