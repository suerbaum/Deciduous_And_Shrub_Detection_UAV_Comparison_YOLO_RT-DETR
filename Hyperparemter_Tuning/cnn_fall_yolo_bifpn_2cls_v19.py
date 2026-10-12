# cnn_fall_yolo_bifpn_2cls_v19.py
#
# Purpose: Train YOLOv8m with BiFPN on fall tiles, 2 classes
# Version 19: same BiFPN approach as summer v19
# Output: results/fall_yolo_bifpn_2cls_v19/

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_fall_2cls"
OUTPUT_BASE = f"{DSS}/results/fall_yolo_bifpn_2cls_v19"
MODEL_CFG   = f"{DSS}/scripts/yolov8m-bifpn.yaml"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO(MODEL_CFG)
    model = model.load("yolov8m.pt")
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