# cnn_summer_rtdetrx_2cls_v15.py
#
# Purpose: Train RT-DETR-X on summer tiles, 2 classes
# Version 15: larger RT-DETR variant (67M parameters vs 32M for RT-DETR-L)
# RT-DETR-X has more capacity — potentially better feature extraction
# for visually similar small objects but higher overfitting risk
# on limited training data (331 tiles)
# Pretrained weights: COCO (rtdetr-x.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_rtdetrx_2cls_v15/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_rtdetrx_2cls_v15"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO("rtdetr-x.pt")
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
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")