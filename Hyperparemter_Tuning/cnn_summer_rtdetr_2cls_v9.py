# cnn_summer_rtdetr_2cls_v9.py
#
# Purpose: Train RT-DETR on summer tiles, 2 classes
# Version 9: higher resolution (1280px) for RT-DETR
# Hypothesis: transformer attention benefits more from higher resolution
# than CNN-based YOLOv8 — larger receptive field improves fine-grained
# texture distinction between visually similar small objects
# Changes from v7:
#   imgsz=1280    — double resolution
#   batch=4       — reduced for VRAM (RT-DETR uses more memory than YOLOv8)
#   lr0=0.0001    — lower lr from v8 finding
#   warmup_epochs=10 — longer warmup from v8
#   flipud=0.5    — top-down appropriate
# Pretrained weights: COCO (rtdetr-l.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_rtdetr_2cls_v9/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_rtdetr_2cls_v9"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO("rtdetr-l.pt")
    model.train(
        data          = f"{DATA_BASE}/fold{fold}/dataset.yaml",
        epochs        = 100,
        imgsz         = 1280,
        batch         = 4,
        workers       = 8,
        device        = 0,
        project       = OUTPUT_BASE,
        name          = f"fold{fold}",
        exist_ok      = True,
        patience      = 30,
        save          = True,
        plots         = True,
        seed          = 42,
        lr0           = 0.0001,
        lrf           = 0.0001,
        warmup_epochs = 10,
        flipud        = 0.5,
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")