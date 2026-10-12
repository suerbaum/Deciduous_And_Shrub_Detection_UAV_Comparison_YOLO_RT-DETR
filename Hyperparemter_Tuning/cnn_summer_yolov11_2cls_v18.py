# cnn_summer_yolov11_2cls_v18.py
#
# Purpose: Train YOLOv11m on summer tiles, 2 classes
# Version 18: YOLOv11 architecture — newest YOLO generation (late 2024)
# YOLOv11 introduces improved backbone with C3k2 blocks and SPPF
# enhanced feature extraction compared to YOLOv8 and YOLOv9
# Uses same settings as best YOLOv8 model (v6):
#   epochs=200, patience=50, mosaic=0, copy_paste=0.3,
#   flipud=0.5, warmup_epochs=5
# Pretrained weights: COCO (yolo11m.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_yolov11_2cls_v18/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_yolov11_2cls_v18"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO("yolo11m.pt")
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