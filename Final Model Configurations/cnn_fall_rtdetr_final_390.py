# cnn_fall_rtdetr_final_390.py
#
# Purpose: Train FINAL RT-DETR-L model on ALL 390 fall tiles
# Same settings as summer 390 model
# Output: results/final_fall_rtdetr_390/

from ultralytics import YOLO

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_YAML   = f"{DSS}/data/final_model_fall_390_2cls/dataset_final.yaml"
OUTPUT_BASE = f"{DSS}/results/final_fall_rtdetr_390"

model = YOLO("rtdetr-l.pt")
model.train(
    data          = DATA_YAML,
    epochs        = 100,
    imgsz         = 640,
    batch         = 8,
    workers       = 8,
    device        = 0,
    project       = OUTPUT_BASE,
    name          = "final",
    exist_ok      = True,
    patience      = 0,
    save          = True,
    plots         = True,
    seed          = 42,
    flipud        = 0.5,
)

print("Final fall 390-tile model training complete.")
print(f"Weights: {OUTPUT_BASE}/final/weights/best.pt")