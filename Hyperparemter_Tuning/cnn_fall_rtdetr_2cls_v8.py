# cnn_fall_rtdetr_2cls_v8.py
#
# Purpose: Train RT-DETR on fall tiles, 2 classes
# Version 8: same hyperparameter tuning as summer v8
# Output: results/fall_rtdetr_2cls_v8/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_fall_2cls"
OUTPUT_BASE = f"{DSS}/results/fall_rtdetr_2cls_v8"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO("rtdetr-l.pt")
    model.train(
        data          = f"{DATA_BASE}/fold{fold}/dataset.yaml",
        epochs        = 150,
        imgsz         = 640,
        batch         = 16,
        workers       = 8,
        device        = 0,
        project       = OUTPUT_BASE,
        name          = f"fold{fold}",
        exist_ok      = True,
        patience      = 40,
        save          = True,
        plots         = True,
        seed          = 42,
        lr0           = 0.0001,
        lrf           = 0.0001,
        warmup_epochs = 10,
        flipud        = 0.5,
        copy_paste    = 0.3,
        mosaic        = 0.0,
        degrees       = 45,
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")