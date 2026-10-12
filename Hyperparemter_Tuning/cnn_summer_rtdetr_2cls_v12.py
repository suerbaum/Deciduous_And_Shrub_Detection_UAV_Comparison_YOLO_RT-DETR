# cnn_summer_rtdetr_2cls_v12.py
#
# Purpose: Train RT-DETR on summer tiles, 2 classes
# Version 12: lower IoU threshold for more lenient box matching
# Rationale: tiny seedlings and shrubs at 1cm GSD are difficult to
# precisely localize — even correct detections may be slightly offset
# from ground truth boxes. Lowering iou from 0.7 to 0.6 reduces
# penalties for correct but imprecise localizations, potentially
# improving recall for small objects
# Changes from v7 (best baseline):
#   iou=0.6  — lower training IoU threshold (default 0.7)
# All other settings identical to v7
# Pretrained weights: COCO (rtdetr-l.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_rtdetr_2cls_v12/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_rtdetr_2cls_v12"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO("rtdetr-l.pt")
    model.train(
        data          = f"{DATA_BASE}/fold{fold}/dataset.yaml",
        epochs        = 100,
        imgsz         = 640,
        batch         = 8,
        workers       = 8,
        device        = 0,
        project       = OUTPUT_BASE,
        name          = f"fold{fold}",
        exist_ok      = True,
        patience      = 20,
        save          = True,
        plots         = True,
        seed          = 42,
        flipud        = 0.5,
        iou           = 0.6,
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")