# cnn_summer_rtdetr_final_390.py
#
# Purpose: Train FINAL RT-DETR-L model on ALL 390 summer tiles
# Includes 331 train+val tiles AND 59 test tiles
# This model is used ONLY for field plot evaluation
# Test set evaluation (Phase 3) is already complete — those results are final
# Using all 390 tiles gives the strongest possible model for field plot inference
# Hyperparameters identical to v7 (best CV model)
# Fixed epochs=100, patience=0 (no early stopping)
# Output: results/final_summer_rtdetr_390/

from ultralytics import YOLO

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_YAML   = f"{DSS}/data/final_model_390_2cls/dataset_final.yaml"
OUTPUT_BASE = f"{DSS}/results/final_summer_rtdetr_390"

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

print("Final summer 390-tile model training complete.")
print(f"Weights: {OUTPUT_BASE}/final/weights/best.pt")