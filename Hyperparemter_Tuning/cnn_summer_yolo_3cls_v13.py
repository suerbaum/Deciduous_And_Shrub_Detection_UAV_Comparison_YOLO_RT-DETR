# cnn_summer_yolo_3cls_v13.py
#
# Purpose: Train YOLOv8m on summer tiles, 3 classes
# Version 13: 3-class YOLOv8m with coniferous as distractor class
# Hypothesis: explicit coniferous class reduces false positives for
# deciduous_tree and shrub — same approach as RT-DETR v11 but with
# best YOLOv8 settings (v6: 200 epochs, v4 augmentation)
# During inference filter out class 0 (coniferous) detections
# Class mapping: 0=coniferous, 1=deciduous_tree, 2=shrub
# Changes from YOLOv8 v6:
#   nc=3 (via dataset yaml) — adds coniferous class
#   uses cv_folds_3cls data (original labels, not filtered)
# Inherits from v6: epochs=200, patience=50, mosaic=0, copy_paste=0.3,
#                   flipud=0.5, warmup_epochs=5
# Pretrained weights: COCO (yolov8m.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_yolo_3cls_v13/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_3cls"
OUTPUT_BASE = f"{DSS}/results/summer_yolo_3cls_v13"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO("yolov8m.pt")
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