# cnn_fall_coco_2cls_v3.py
#
# Purpose: Train YOLOv8m on fall tiles, 2 classes
# Version 3: higher resolution (1280px) for better small object detection
# Changes from v1: imgsz=1280, batch=8, flipud=0.5, patience=30, warmup_epochs=5
# Pretrained weights: COCO (yolov8m.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/fall_coco_2cls_v3/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_fall_2cls"
OUTPUT_BASE = f"{DSS}/results/fall_coco_2cls_v3"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO("yolov8m.pt")
    model.train(
        data          = f"{DATA_BASE}/fold{fold}/dataset.yaml",
        epochs        = 100,
        imgsz         = 1280,
        batch         = 8,
        workers       = 8,
        device        = 0,
        project       = OUTPUT_BASE,
        name          = f"fold{fold}",
        exist_ok      = True,
        patience      = 30,
        save          = True,
        plots         = True,
        seed          = 42,
        flipud        = 0.5,
        warmup_epochs = 5,
        close_mosaic  = 15,
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")