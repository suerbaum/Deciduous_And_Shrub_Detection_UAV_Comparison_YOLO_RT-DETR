# cnn_summer_rtdetr_2cls_v17c.py
#
# Purpose: Train RT-DETR on summer tiles, 2 classes
# Version 17c: dropout + scale augmentation without mixup
# Hypothesis: dropout=0.1 regularizes transformer layers to reduce overfitting
#             scale=0.7 simulates plants at varying apparent sizes (0.3-1.7x)
#             both are safe with RT-DETR — no label ambiguity issues
# Why not mixup: mixup creates fractional label assignments incompatible
#                with RT-DETR Hungarian matching → NaN gradients (confirmed v17)
# Why not explicit AdamW: auto-optimizer already selects AdamW lr=0.001667
#                         explicit override breaks this → divergence (confirmed v17b)
# Changes from v7 (best baseline):
#   dropout=0.1  — regularization inside transformer layers
#   scale=0.7    — random scale augmentation 0.3-1.7x
# All other settings identical to v7
# Pretrained weights: COCO (rtdetr-l.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_rtdetr_2cls_v17c/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_rtdetr_2cls_v17c"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO("rtdetr-l.pt")
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
        dropout       = 0.1,
        scale         = 0.7,
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")