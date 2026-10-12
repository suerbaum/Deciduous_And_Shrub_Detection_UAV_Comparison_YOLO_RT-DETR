# cnn_summer_rtdetr_final.py
#
# Purpose: Train FINAL RT-DETR-L model on ALL 331 summer tiles
# This is the definitive model for test set evaluation and SAHI inference
# No CV folds — all 331 tiles used for training
# Hyperparameters identical to v7 (best CV model)
# Fixed epochs=100 (max best epoch from CV was 81, adding buffer)
# patience=0 disables early stopping — fixed epoch count used instead
# val points to same folder as train — monitored but not used for stopping
# optimizer=auto automatically selects AdamW lr=0.001667 (confirmed from v7 log)
# Output: results/final_summer_rtdetr/

from ultralytics import YOLO

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_YAML   = f"{DSS}/data/final_model_2cls/dataset_final.yaml"
OUTPUT_BASE = f"{DSS}/results/final_summer_rtdetr"

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
    patience      = 0,       # disabled — fixed epoch count
    save          = True,
    plots         = True,
    seed          = 42,
    flipud        = 0.5,
)

print("Final summer model training complete.")
print(f"Weights saved to: {OUTPUT_BASE}/final/weights/best.pt")