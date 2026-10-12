# cnn_summer_coco_2cls_v5.py
#
# Purpose: Train YOLOv8m on summer tiles, 2 classes
# Version 5: augmentation tuning for small object detection
# Changes from v1:
#   copy_paste=0.5  — copies plant instances onto background areas of other tiles
#                     increases instance diversity without new labeled data
#   degrees=45      — random rotation up to ±45 degrees
#                     appropriate for top-down drone imagery, plant looks same
#                     from any compass direction
#   flipud=0.5      — vertical flip, appropriate for top-down imagery
#   mosaic=1.0      — back to default (was disabled in v4)
#   patience=30     — more epochs before early stopping
#   warmup_epochs=5 — longer warmup for small dataset stability
# Pretrained weights: COCO (yolov8m.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_coco_2cls_v5/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_coco_2cls_v5"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO("yolov8m.pt")
    model.train(
        data          = f"{DATA_BASE}/fold{fold}/dataset.yaml",
        epochs        = 100,
        imgsz         = 640,
        batch         = 16,
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
        copy_paste    = 0.5,
        degrees       = 45,
        mosaic        = 1.0,
        warmup_epochs = 5,
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")