# cnn_summer_rtdetr_2cls_v10.py
#
# Purpose: Train RT-DETR on summer tiles, 2 classes
# Version 10: regularization and cosine learning rate schedule
# Hypothesis: weight decay reduces overfitting by penalizing large weights,
# cosine lr schedule decays learning rate more smoothly than linear —
# both particularly beneficial for transformer architectures
# Changes from v8:
#   weight_decay=0.001  — stronger regularization (default 0.0005)
#                         penalizes large weights, encourages generalization
#   cos_lr=True         — cosine annealing instead of linear lr decay
#                         slower decay early, faster near end — better for
#                         transformer attention convergence
# Inherits from v8: lr0=0.0001, batch=16, epochs=150, patience=40,
#                   warmup_epochs=10, mosaic=0, copy_paste=0.3,
#                   degrees=45, flipud=0.5
# Pretrained weights: COCO (rtdetr-l.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_rtdetr_2cls_v10/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_rtdetr_2cls_v10"

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
        weight_decay  = 0.001,
        cos_lr        = True,
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")