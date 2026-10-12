# cnn_summer_yolov10_2cls_v16.py
#
# Purpose: Train YOLOv10m on summer tiles, 2 classes
# Version 16: YOLOv10m architecture comparison
# YOLOv10 eliminates NMS at inference using dual-label assignment
# potentially more precise detections for dense small object scenarios
# Pretrained weights: COCO (yolov10m.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_yolov10_2cls_v16/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_yolov10_2cls_v16"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO("yolov10m.pt")
    model.train(
        data          = f"{DATA_BASE}/fold{fold}/dataset.yaml",
        epochs        = 200,
        imgsz         = 640,
        batch         = 16,
        workers       = 8,
        device        = 0,
        project       = OUTPUT_BASE,
        name          = f"fold{fold}",
        exist_ok      = True,
        patience      = 50,
        save          = True,
        plots         = True,
        seed          = 42,
        flipud        = 0.5,
        mosaic        = 0.0,
        copy_paste    = 0.3,
        warmup_epochs = 5,
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")