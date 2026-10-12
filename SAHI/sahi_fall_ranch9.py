# sahi_fall_ranch9.py
#
# Purpose: SAHI inference on Ranch 9 FALL orthomosaic
# Input: ranch9_fall_final.tif (8.2GB)

import os, json, rasterio
from rasterio.transform import xy as rasterio_xy
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

DSS = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"

RANCH,SEASON   = "ranch9","fall"
WEIGHTS        = f"{DSS}/results/final_fall_rtdetr_390/final/weights/best.pt"
ORTHO_PATH     = f"{DSS}/data/orthomosaics/ranch9_fall_final.tif"
OUTPUT_DIR     = f"{DSS}/results/sahi_fall"
OUTPUT_FILE    = f"{OUTPUT_DIR}/sahi_{RANCH}_{SEASON}.geojson"
SLICE_SIZE,OVERLAP_RATIO,CONF_THRESHOLD,IOU_THRESHOLD = 640,0.2,0.25,0.5
CLASS_NAMES    = {0:"deciduous_tree",1:"shrub"}

os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.exists(ORTHO_PATH):
    raise FileNotFoundError(f"Not found: {ORTHO_PATH}")

with rasterio.open(ORTHO_PATH) as src:
    transform,crs,width,height = src.transform,src.crs,src.width,src.height
print(f"{RANCH} {SEASON}: {width}x{height}px, CRS:{crs}")

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics", model_path=WEIGHTS,
    confidence_threshold=CONF_THRESHOLD, device="cuda:0")
print("Model loaded")

result = get_sliced_prediction(
    image=ORTHO_PATH, detection_model=detection_model,
    slice_height=SLICE_SIZE, slice_width=SLICE_SIZE,
    overlap_height_ratio=OVERLAP_RATIO, overlap_width_ratio=OVERLAP_RATIO,
    perform_standard_pred=False, postprocess_type="NMM",
    postprocess_match_threshold=IOU_THRESHOLD, verbose=1)
print(f"Detections: {len(result.object_prediction_list)}")

features,n_deciduous,n_shrub = [],[],[]
n_deciduous,n_shrub = 0,0
for pred in result.object_prediction_list:
    bbox = pred.bbox
    class_id = pred.category.id
    confidence = float(pred.score.value)
    utm_xmin,utm_ymax = rasterio_xy(transform,bbox.miny,bbox.minx)
    utm_xmax,utm_ymin = rasterio_xy(transform,bbox.maxy,bbox.maxx)
    if class_id==0: n_deciduous+=1
    elif class_id==1: n_shrub+=1
    features.append({"type":"Feature",
        "geometry":{"type":"Polygon","coordinates":[[
            [utm_xmin,utm_ymin],[utm_xmax,utm_ymin],
            [utm_xmax,utm_ymax],[utm_xmin,utm_ymax],[utm_xmin,utm_ymin]]]},
        "properties":{"class":CLASS_NAMES.get(class_id,f"unknown_{class_id}"),
            "class_id":class_id,"confidence":round(confidence,4),
            "utm_cx":round((utm_xmin+utm_xmax)/2,2),
            "utm_cy":round((utm_ymin+utm_ymax)/2,2),
            "ranch":RANCH,"season":SEASON}})

with open(OUTPUT_FILE,"w") as f:
    json.dump({"type":"FeatureCollection",
        "crs":{"type":"name","properties":{"name":str(crs)}},
        "features":features},f)
print(f"\n=== {RANCH.upper()} {SEASON.upper()} ===")
print(f"Total:{len(features)} deciduous={n_deciduous} shrub={n_shrub}")
print(f"Saved:{OUTPUT_FILE}")