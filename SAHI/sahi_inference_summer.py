# sahi_inference_summer.py
#
# Purpose: Run SAHI (Sliced Aided Hyper Inference) on all three
# summer orthomosaics using the final RT-DETR-L model (390 tiles)
#
# Why SAHI:
#   Orthomosaics are 13GB+ images. Running the model directly would
#   downscale plants to 1-2px — invisible. SAHI slices the orthomosaic
#   into overlapping 640x640px tiles (same size as training), runs
#   inference on each tile, then merges detections removing duplicates
#   at tile boundaries using NMS. This gives accurate small-object
#   detection across the full landscape.
#
# To reproduce for fall imagery:
#   1. Change SEASON = "summer" to SEASON = "fall"
#   2. Change WEIGHTS to final_fall_rtdetr/final/weights/best.pt
#   3. Change ORTHO_PATHS to fall orthomosaic paths
#   4. Everything else stays identical
#
# Output per ranch:
#   - GeoJSON file with all detections + GPS coordinates
#   - Each detection has: class, confidence, bbox in pixel coords,
#     bbox in UTM coordinates
#   - Load directly in QGIS for visualization
#
# Runtime estimate: ~2-4 hours per ranch on V100 GPU
# Ranch 5 (13GB) will take longest

import os
import json
import numpy as np
import rasterio
from rasterio.transform import xy
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image

DSS = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"

# ── CONFIGURATION — change these for fall ─────────────────────────────────────
SEASON  = "summer"
WEIGHTS = f"{DSS}/results/final_summer_rtdetr_390/final/weights/best.pt"

# summer orthomosaic paths on DSS
ORTHO_PATHS = {
    "ranch5":  f"{DSS}/data/orthomosaics/ortho_ranch5_merged_v4.tif",
    "ranch9":  f"{DSS}/data/orthomosaics/ranch9_summer_aligned.tif",
    "ranch12": f"{DSS}/data/orthomosaics/ranch12_summer_aligned.tif",
}

# output folder
OUTPUT_DIR = f"{DSS}/results/sahi_{SEASON}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── SAHI settings ──────────────────────────────────────────────────────────────
# slice_size matches training tile size — critical for performance
# overlap_ratio 0.2 = 20% overlap between slices
# ensures plants at tile boundaries are not missed
# conf_threshold 0.25 = standard inference threshold
SLICE_SIZE     = 640
OVERLAP_RATIO  = 0.2
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.5   # NMS threshold for merging duplicate detections

# class names — must match training order
CLASS_NAMES = {0: "deciduous_tree", 1: "shrub"}

# ── load model ─────────────────────────────────────────────────────────────────
print(f"Loading RT-DETR-L model ({SEASON})...")
detection_model = AutoDetectionModel.from_pretrained(
    model_type          = "ultralytics",
    model_path          = WEIGHTS,
    confidence_threshold= CONF_THRESHOLD,
    device              = "cuda:0",  # GPU on cluster
)
print("Model loaded")

# ── process each ranch ─────────────────────────────────────────────────────────
for ranch_name, ortho_path in ORTHO_PATHS.items():
    print(f"\n{'='*60}")
    print(f"Processing {ranch_name} {SEASON}...")
    print(f"Orthomosaic: {ortho_path}")
    print(f"{'='*60}")

    if not os.path.exists(ortho_path):
        print(f"WARNING: orthomosaic not found — skipping {ranch_name}")
        continue

    # ── open orthomosaic to get geotransform ───────────────────────────────────
    # the geotransform converts pixel coordinates to UTM coordinates
    # needed to save detection locations as real GPS coordinates
    with rasterio.open(ortho_path) as src:
        transform = src.transform
        crs       = src.crs
        width     = src.width
        height    = src.height
        print(f"Size: {width} x {height} pixels")
        print(f"CRS: {crs}")

    # ── run SAHI sliced inference ──────────────────────────────────────────────
    # SAHI reads the orthomosaic, slices it into 640x640 tiles with 20% overlap
    # runs RT-DETR on each tile, merges results with NMS
    print(f"Running SAHI inference (slice={SLICE_SIZE}px, overlap={OVERLAP_RATIO})...")

    result = get_sliced_prediction(
        image                       = ortho_path,
        detection_model             = detection_model,
        slice_height                = SLICE_SIZE,
        slice_width                 = SLICE_SIZE,
        overlap_height_ratio        = OVERLAP_RATIO,
        overlap_width_ratio         = OVERLAP_RATIO,
        perform_standard_pred       = False,  # skip full-image prediction
        postprocess_type            = "NMM",  # Non-Maximum Merging
        postprocess_match_threshold = IOU_THRESHOLD,
        verbose                     = 1,
    )

    print(f"Total detections: {len(result.object_prediction_list)}")

    # ── convert detections to GeoJSON with UTM coordinates ────────────────────
    # each detection bounding box is converted from pixel to UTM coordinates
    # so detections can be displayed on a map in QGIS

    features = []
    n_deciduous = 0
    n_shrub     = 0

    for pred in result.object_prediction_list:
        # bounding box in pixel coordinates
        bbox  = pred.bbox
        x_min = bbox.minx
        y_min = bbox.miny
        x_max = bbox.maxx
        y_max = bbox.maxy

        # convert pixel coordinates to UTM using rasterio geotransform
        # xy() converts row/col to x/y UTM coordinates
        utm_xmin, utm_ymax = xy(transform, y_min, x_min)
        utm_xmax, utm_ymin = xy(transform, y_max, x_max)

        # center point in UTM
        utm_cx = (utm_xmin + utm_xmax) / 2
        utm_cy = (utm_ymin + utm_ymax) / 2

        class_id   = pred.category.id
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        confidence = pred.score.value

        if class_id == 0:
            n_deciduous += 1
        elif class_id == 1:
            n_shrub += 1

        # GeoJSON feature — polygon bounding box in UTM coordinates
        feature = {
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
                "class":       class_name,
                "class_id":    class_id,
                "confidence":  round(confidence, 4),
                "px_xmin":     int(x_min),
                "px_ymin":     int(y_min),
                "px_xmax":     int(x_max),
                "px_ymax":     int(y_max),
                "utm_cx":      round(utm_cx, 2),
                "utm_cy":      round(utm_cy, 2),
                "ranch":       ranch_name,
                "season":      SEASON,
            }
        }
        features.append(feature)

    # ── save GeoJSON ───────────────────────────────────────────────────────────
    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": str(crs)}
        },
        "features": features
    }

    output_path = os.path.join(
        OUTPUT_DIR, f"sahi_{ranch_name}_{SEASON}.geojson"
    )
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"\n=== {ranch_name.upper()} {SEASON.upper()} RESULTS ===")
    print(f"Total detections: {len(features)}")
    print(f"deciduous_tree:   {n_deciduous}")
    print(f"shrub:            {n_shrub}")
    print(f"Saved: {output_path}")

print(f"\nAll {SEASON} ranches complete.")
print(f"Results in: {OUTPUT_DIR}")
print("Download GeoJSON files and load in QGIS for visualization")