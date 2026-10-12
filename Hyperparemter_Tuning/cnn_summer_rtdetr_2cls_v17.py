# cnn_summer_rtdetr_2cls_v17.py
#
# Purpose: Train RT-DETR on summer tiles, 2 classes
# Version 17: untested hyperparameter combinations for RT-DETR
# Based on v7 (best baseline) with three new additions:
#   optimizer="AdamW" — standard optimizer for transformer architectures
#                       designed specifically for transformers in vision/NLP
#                       better weight decay handling than Adam/SGD
#   dropout=0.1       — adds regularization inside transformer layers
#                       addresses overfitting confirmed in training curves
#                       small value (0.1) to regularize without hurting learning
#   mixup=0.1         — blends two training images with random weight
#                       creates new synthetic training examples
#                       specifically helps with limited training data (331 tiles)
#   scale=0.7         — random scale augmentation 0.3-1.7x
#                       simulates plants at different apparent sizes
#                       appropriate for varying plant sizes in your data
# All other settings identical to v7 (best model)
# Pretrained weights: COCO (rtdetr-l.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_rtdetr_2cls_v17/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_rtdetr_2cls_v17"

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
        optimizer     = "AdamW",
        dropout       = 0.1,
        mixup         = 0.1,
        scale         = 0.7,
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")