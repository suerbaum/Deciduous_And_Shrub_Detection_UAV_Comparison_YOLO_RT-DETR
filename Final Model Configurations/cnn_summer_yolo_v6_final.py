# cnn_summer_yolo_v6_final.py
#
# Purpose: Train FINAL YOLOv8m v6 model on ALL 331 summer tiles
# For direct comparison with RT-DETR v7 final model on test set
# Hyperparameters identical to v6 CV (best YOLO configuration)
# Fixed epochs=200, patience=0 (max best epoch from CV was 169)
# Data: final_model_2cls (331 tiles, same as RT-DETR final)

from ultralytics import YOLO

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_YAML   = f"{DSS}/data/final_model_2cls/dataset_final.yaml"
OUTPUT_BASE = f"{DSS}/results/final_summer_yolo_v6"

model = YOLO("yolov8m.pt")
model.train(
    data       = DATA_YAML,
    epochs     = 200,
    imgsz      = 640,
    batch      = 16,
    workers    = 8,
    device     = 0,
    project    = OUTPUT_BASE,
    name       = "final",
    exist_ok   = True,
    patience   = 0,
    save       = True,
    plots      = True,
    seed       = 42,
    flipud     = 0.5,
    copy_paste = 0.3,
    mosaic     = 0.0,
)

print("Final summer YOLOv8m v6 training complete.")
print(f"Weights: {OUTPUT_BASE}/final/weights/best.pt")