# cnn_summer_coco_2cls_v4.py
#
# Purpose: Train YOLOv8m on summer tiles, 2 classes
# Version 4: augmentation tuning for small object detection
# Changes from v1:
#   mosaic=0.0     — disables mosaic augmentation which scales objects down
#                    and can harm detection of already-small seedlings
#   copy_paste=0.3 — randomly copies plant instances between tiles
#                    helps with class imbalance and rare instances
#   flipud=0.5     — vertical flip, appropriate for top-down drone imagery
#   patience=30    — more epochs before early stopping
#   warmup_epochs=5 — longer warmup for small dataset stability
# Pretrained weights: COCO (yolov8m.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_coco_2cls_v4/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_coco_2cls_v4"

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
        flipud        = 0.5,       # vertical flip — top-down imagery
        mosaic        = 0.0,       # disable mosaic — harmful for small objects
        copy_paste    = 0.3,       # copy plant instances between tiles
        warmup_epochs = 5,         # longer warmup for small dataset
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")