# cnn_fall_rtdetr_2cls_v10_resume.py
#
# Purpose: Resume fall RT-DETR v10 fold 5 which was interrupted
# Resumes from last saved checkpoint in fold5/weights/last.pt

from ultralytics import YOLO
import os

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
OUTPUT_BASE = f"{DSS}/results/fall_rtdetr_2cls_v10"

# resume fold 5 only
fold = 5
print(f"========= Resuming Fold {fold} =========")
model = YOLO(f"{OUTPUT_BASE}/fold{fold}/weights/last.pt")
model.train(resume=True)
print(f"Fold {fold} complete.") 