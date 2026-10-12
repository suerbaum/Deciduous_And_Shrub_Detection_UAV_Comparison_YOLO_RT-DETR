# cnn_fall_p2_2cls.py
#
# Purpose: Train YOLOv8m with P2 head on fall tiles, 2 classes
# P2 head adds stride-4 detection layer for small object detection
# Pretrained weights: COCO (yolov8m.pt) transferred to P2 architecture
# CV: 5-fold spatially-aware (knndm)
# Output: results/fall_p2_2cls/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_fall_2cls"
OUTPUT_BASE = f"{DSS}/results/fall_p2_2cls"
MODEL_CFG   = f"{DSS}/scripts/yolov8m-p2.yaml"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO(MODEL_CFG)
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