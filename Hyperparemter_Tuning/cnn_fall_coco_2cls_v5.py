# cnn_fall_coco_2cls_v5.py
#
# Purpose: Train YOLOv8m on fall tiles, 2 classes
# Version 5: same augmentation tuning as summer v5
# Output: results/fall_coco_2cls_v5/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_fall_2cls"
OUTPUT_BASE = f"{DSS}/results/fall_coco_2cls_v5"

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