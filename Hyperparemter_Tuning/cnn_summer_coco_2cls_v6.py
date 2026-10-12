# cnn_summer_coco_2cls_v6.py
#
# Purpose: Train YOLOv8m on summer tiles, 2 classes
# Version 6: extended training of v4 augmentation strategy
# Notes: identical augmentation to v4 (mosaic=0.0, copy_paste=0.3, flipud=0.5)
#        v4 showed best generalization (smallest train/val loss gap) but
#        never converged within 100 epochs — extended to 200 epochs here
#        patience increased to 50 to match longer training
# Changes from v4:
#   epochs=200    — up from 100, model was still improving at epoch 100
#   patience=50   — up from 30, proportional to longer training
# Pretrained weights: COCO (yolov8m.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_coco_2cls_v6/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_coco_2cls_v6"

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
        flipud        = 0.5,       # same as v4
        mosaic        = 0.0,       # same as v4 — disabled for small objects
        copy_paste    = 0.3,       # same as v4
        warmup_epochs = 5,         # same as v4
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")