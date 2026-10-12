# cnn_summer_p2_2cls_v2.py
#
# Purpose: Train YOLOv8m with P2 head on summer tiles, 2 classes
# Version 2: fixes missing COCO weight transfer from v1
# P2 head adds stride-4 detection layer for small object detection
# Pretrained weights: COCO transferred to P2 architecture via model.load()
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_p2_2cls_v2/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_p2_2cls_v2"
MODEL_CFG   = f"{DSS}/scripts/yolov8m-p2.yaml"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO(MODEL_CFG)
    model = model.load("yolov8m.pt")  # transfer COCO pretrained weights
    model.train(
        data        = f"{DATA_BASE}/fold{fold}/dataset.yaml",
        epochs      = 100,
        imgsz       = 640,
        batch       = 16,
        workers     = 8,
        device      = 0,
        project     = OUTPUT_BASE,
        name        = f"fold{fold}",
        exist_ok    = True,
        patience    = 20,
        save        = True,
        plots       = True,
        seed        = 42,
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")