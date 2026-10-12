from ultralytics import YOLO
import os
import torch

# force single GPU non-distributed training
os.environ["RANK"] = "-1"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "-1"

# disable any distributed init
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()

DSS = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds"
OUTPUT_BASE = f"{DSS}/results/summer_coco"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO("yolov8m.pt")
    model.train(
        data        = f"{DATA_BASE}/fold{fold}/dataset.yaml",
        epochs      = 100,
        imgsz       = 640,
        batch       = 16,
        workers     = 8,
        device      = 0,
        project     = OUTPUT_BASE,
        name        = f"fold{fold}",
        exist_ok    = True,
        patience    = 20,
        save        = True,
        plots       = True,
        seed        = 42,
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")