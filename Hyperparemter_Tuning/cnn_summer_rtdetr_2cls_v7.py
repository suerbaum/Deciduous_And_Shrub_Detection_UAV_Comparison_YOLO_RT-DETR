# cnn_summer_rtdetr_2cls_v7.py
#
# Purpose: Train RT-DETR (Real-Time Detection Transformer) on summer tiles, 2 classes
# Version 7: transformer-based architecture comparison to YOLOv8m
# RT-DETR uses self-attention to capture global context — potentially better
# for visually similar small objects (seedlings vs shrubs)
# Pretrained weights: COCO (rtdetr-l.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_rtdetr_2cls_v7/
# Note: RT-DETR-L has 32M parameters, comparable to YOLOv8m (26M)

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_rtdetr_2cls_v7"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO("rtdetr-l.pt")  # RT-DETR large, COCO pretrained
    model.train(
        data          = f"{DATA_BASE}/fold{fold}/dataset.yaml",
        epochs        = 100,
        imgsz         = 640,
        batch         = 8,        # reduced — transformer needs more memory
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