# cnn_fall_rtdetr_final.py
#
# Purpose: Train FINAL RT-DETR-L model on ALL 331 fall tiles
# Same settings as summer final model
# Output: results/final_fall_rtdetr/

from ultralytics import YOLO

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_YAML   = f"{DSS}/data/final_model_fall_2cls/dataset_final.yaml"
OUTPUT_BASE = f"{DSS}/results/final_fall_rtdetr"

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

print("Final fall model training complete.")
print(f"Weights saved to: {OUTPUT_BASE}/final/weights/best.pt")