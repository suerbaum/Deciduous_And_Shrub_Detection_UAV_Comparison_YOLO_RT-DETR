# cnn_summer_rtdetr_2cls_v8.py
#
# Purpose: Train RT-DETR on summer tiles, 2 classes
# Version 8: hyperparameter tuning for RT-DETR transformer architecture
# Changes from v7:
#   lr0=0.0001      — lower learning rate, transformers need careful updates
#   lrf=0.0001      — low final lr, maintain careful updates throughout
#   warmup_epochs=10 — longer warmup, transformers benefit more than CNNs
#   batch=16        — larger batch for more stable attention learning
#   epochs=150      — more epochs, fold 5 in v7 showed improvement until ep81
#   patience=40     — proportional to longer training
#   mosaic=0.0      — disable mosaic, harmful for small objects
#   copy_paste=0.3  — instance diversity augmentation
#   degrees=45      — rotation augmentation for top-down imagery
#   flipud=0.5      — vertical flip for top-down imagery
# Pretrained weights: COCO (rtdetr-l.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_rtdetr_2cls_v8/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_rtdetr_2cls_v8"

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