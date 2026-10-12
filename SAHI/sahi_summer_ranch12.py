# sahi_summer_ranch12.py
#
# Purpose: SAHI inference on Ranch 12 SUMMER orthomosaic
# Model: RT-DETR-L trained on 390 tiles
# Input: ranch12_summer_aligned.tif (6GB, 45904x39042px)
# Output: sahi_ranch12_summer.geojson

import os
import json
import rasterio
from rasterio.transform import xy as rasterio_xy
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

DSS = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"

RANCH       = "ranch12"
SEASON      = "summer"
WEIGHTS     = f"{DSS}/results/final_summer_rtdetr_390/final/weights/best.pt"
ORTHO_PATH  = f"{DSS}/data/orthomosaics/ranch12_summer_aligned.tif"
OUTPUT_DIR  = f"{DSS}/results/sahi_summer"
OUTPUT_FILE = f"{OUTPUT_DIR}/sahi_{RANCH}_{SEASON}.geojson"

SLICE_SIZE     = 640
OVERLAP_RATIO  = 0.2
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.5
CLASS_NAMES    = {0: "deciduous_tree", 1: "shrub"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(ORTHO_PATH):
    raise FileNotFoundError(f"Orthomosaic not found: {ORTHO_PATH}")

with rasterio.open(ORTHO_PATH) as src:
    transform = src.transform
    crs       = src.crs
    width     = src.width
    height    = src.height
print(f"{RANCH} {SEASON}: {width} x {height} pixels, CRS: {crs}")

print("Loading RT-DETR-L 390-tile model...")
detection_model = AutoDetectionModel.from_pretrained(
    model_type           = "ultralytics",
    model_path           = WEIGHTS,
    confidence_threshold = CONF_THRESHOLD,
    device               = "cuda:0",
)
print("Model loaded successfully")

print(f"Running SAHI inference (slice={SLICE_SIZE}px, overlap={OVERLAP_RATIO*100:.0f}%)...")
result = get_sliced_prediction(
    image                       = ORTHO_PATH,
    detection_model             = detection_model,
    slice_height                = SLICE_SIZE,
    slice_width                 = SLICE_SIZE,
    overlap_height_ratio        = OVERLAP_RATIO,
    overlap_width_ratio         = OVERLAP_RATIO,
    perform_standard_pred       = False,
    postprocess_type            = "NMM",
    postprocess_match_threshold = IOU_THRESHOLD,
    verbose                     = 1,
)
print(f"Final detections after NMM: {len(result.object_prediction_list)}")

features    = []
n_deciduous = 0
n_shrub     = 0

for pred in result.object_prediction_list:
    bbox       = pred.bbox
    class_id   = pred.category.id
    class_name = CLASS_NAMES.get(class_id, f"unknown_{class_id}")
    confidence = float(pred.score.value)

    utm_xmin, utm_ymax = rasterio_xy(transform, bbox.miny, bbox.minx)
    utm_xmax, utm_ymin = rasterio_xy(transform, bbox.maxy, bbox.maxx)
    utm_cx = (utm_xmin + utm_xmax) / 2
    utm_cy = (utm_ymin + utm_ymax) / 2

    if class_id == 0:
        n_deciduous += 1
    elif class_id == 1:
        n_shrub += 1

    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [utm_xmin, utm_ymin],
                [utm_xmax, utm_ymin],
                [utm_xmax, utm_ymax],
                [utm_xmin, utm_ymax],
                [utm_xmin, utm_ymin],
            ]]
        },
        "properties": {
            "class":      class_name,
            "class_id":   class_id,
            "confidence": round(confidence, 4),
            "px_xmin":    int(bbox.minx),
            "px_ymin":    int(bbox.miny),
            "px_xmax":    int(bbox.maxx),
            "px_ymax":    int(bbox.maxy),
            "utm_cx":     round(utm_cx, 2),
            "utm_cy":     round(utm_cy, 2),
            "ranch":      RANCH,
            "season":     SEASON,
        }
    })

geojson = {
    "type": "FeatureCollection",
    "crs": {"type": "name", "properties": {"name": str(crs)}},
    "features": features
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(geojson, f)

print(f"\n=== RESULTS: {RANCH.upper()} {SEASON.upper()} ===")
print(f"Total detections: {len(features)}")
print(f"deciduous_tree:   {n_deciduous}")
print(f"shrub:            {n_shrub}")
print(f"Saved: {OUTPUT_FILE}")