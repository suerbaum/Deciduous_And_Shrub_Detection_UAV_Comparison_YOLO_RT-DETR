# cnn_summer_rtdetr_3cls_v11.py
#
# Purpose: Train RT-DETR on summer tiles, 3 classes
# Version 11: 3-class model including coniferous as distractor class
# Hypothesis: explicit coniferous class reduces false positives for
# deciduous_tree and shrub — model learns to distinguish all three
# rather than misclassifying conifers as target classes
# Note: during inference filter out class 0 (coniferous) detections
# Inherits best settings from v8: lr0=0.0001, batch=16, 150ep,
# warmup_epochs=10, mosaic=0, copy_paste=0.3, degrees=45
# Pretrained weights: COCO (rtdetr-l.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_rtdetr_3cls_v11/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_3cls"
OUTPUT_BASE = f"{DSS}/results/summer_rtdetr_3cls_v11"

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