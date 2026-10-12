# sahi_inference_fall.py
#
# Purpose: Run SAHI inference on all three fall orthomosaics
# using the final RT-DETR-L fall model
#
# Identical to sahi_inference_summer.py except:
#   SEASON  = "fall"
#   WEIGHTS = final_fall_rtdetr weights
#   ORTHO_PATHS = fall orthomosaic paths
#
# Fall orthomosaics must be uploaded to DSS before running:
#   ortho_ranch5_fall_final_v2.tif
#   ranch9_fall_final.tif
#   ranch12_fall_final.tif

import os
import json
import numpy as np
import rasterio
from rasterio.transform import xy
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

DSS = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
SEASON  = "fall"
WEIGHTS = f"{DSS}/results/final_fall_rtdetr_390/final/weights/best.pt"

ORTHO_PATHS = {
    "ranch5":  f"{DSS}/data/orthomosaics/ortho_ranch5_fall_final_v2.tif",
    "ranch9":  f"{DSS}/data/orthomosaics/ranch9_fall_final.tif",
    "ranch12": f"{DSS}/data/orthomosaics/ranch12_fall_final.tif",
}

OUTPUT_DIR = f"{DSS}/results/sahi_{SEASON}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SLICE_SIZE     = 640
OVERLAP_RATIO  = 0.2
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.5
CLASS_NAMES    = {0: "deciduous_tree", 1: "shrub"}

print(f"Loading RT-DETR-L model ({SEASON})...")
detection_model = AutoDetectionModel.from_pretrained(
    model_type           = "ultralytics",
    model_path           = WEIGHTS,
    confidence_threshold = CONF_THRESHOLD,
    device               = "cuda:0",
)
print("Model loaded")

for ranch_name, ortho_path in ORTHO_PATHS.items():
    print(f"\n{'='*60}")
    print(f"Processing {ranch_name} {SEASON}...")
    print(f"{'='*60}")

    if not os.path.exists(ortho_path):
        print(f"WARNING: {ortho_path} not found — skipping")
        continue

    with rasterio.open(ortho_path) as src:
        transform = src.transform
        crs       = src.crs
        width     = src.width
        height    = src.height
        print(f"Size: {width} x {height} pixels")

    print(f"Running SAHI inference...")
    result = get_sliced_prediction(
        image                       = ortho_path,
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

    features  = []
    n_deciduous = 0
    n_shrub     = 0

    for pred in result.object_prediction_list:
        bbox  = pred.bbox
        utm_xmin, utm_ymax = xy(transform, bbox.miny, bbox.minx)
        utm_xmax, utm_ymin = xy(transform, bbox.maxy, bbox.maxx)
        class_id   = pred.category.id
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        confidence = pred.score.value

        if class_id == 0: n_deciduous += 1
        elif class_id == 1: n_shrub += 1

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [utm_xmin, utm_ymin], [utm_xmax, utm_ymin],
                    [utm_xmax, utm_ymax], [utm_xmin, utm_ymax],
                    [utm_xmin, utm_ymin],
                ]]
            },
            "properties": {
                "class": class_name, "class_id": class_id,
                "confidence": round(confidence, 4),
                "ranch": ranch_name, "season": SEASON,
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": str(crs)}},
        "features": features
    }

    output_path = os.path.join(OUTPUT_DIR, f"sahi_{ranch_name}_{SEASON}.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"Total: {len(features)} — deciduous={n_deciduous}, shrub={n_shrub}")
    print(f"Saved: {output_path}")

print(f"\nAll {SEASON} ranches complete. Results in: {OUTPUT_DIR}")