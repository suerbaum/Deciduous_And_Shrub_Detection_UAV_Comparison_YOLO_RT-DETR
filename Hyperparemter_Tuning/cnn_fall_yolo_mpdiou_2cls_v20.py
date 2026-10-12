# cnn_fall_yolo_mpdiou_2cls_v20.py
#
# Purpose: Train YOLOv8m with MPDIoU loss on fall tiles, 2 classes
# Version 20: same MPDIoU approach as summer v20
# Output: results/fall_yolo_mpdiou_2cls_v20/

import torch
from ultralytics import YOLO
from ultralytics.utils.loss import BboxLoss
from ultralytics.models.yolo.detect import DetectionTrainer
import os

def mpdiou(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    d1 = (b1_x1 - b2_x1) ** 2 + (b1_y1 - b2_y1) ** 2
    d2 = (b1_x2 - b2_x2) ** 2 + (b1_y2 - b2_y2) ** 2
    return iou - (d1 + d2) / c2

class MPDIoUBboxLoss(BboxLoss):
    def iou_loss(self, pred_bboxes, target_bboxes, target_scores,
                 target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        return loss_iou, iou

class MPDIoUTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        if hasattr(model, 'criterion') and hasattr(model.criterion, 'bbox_loss'):
            model.criterion.bbox_loss = MPDIoUBboxLoss(
                model.criterion.bbox_loss.reg_max,
                model.criterion.bbox_loss.use_dfl
            )
        return model

DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_fall_2cls"
OUTPUT_BASE = f"{DSS}/results/fall_yolo_mpdiou_2cls_v20"

for fold in range(1, 6):
    print(f"========= Fold {fold} / 5 =========")
    model = YOLO("yolov8m.pt")
    model.train(
        data          = f"{DATA_BASE}/fold{fold}/dataset.yaml",
        epochs        = 200,
        imgsz         = 640,
        batch         = 16,
        workers       = 8,
        device        = 0,
        project       = OUTPUT_BASE,
        name          = f"fold{fold}",
        exist_ok      = True,
        patience      = 50,
        save          = True,
        plots         = True,
        seed          = 42,
        flipud        = 0.5,
        mosaic        = 0.0,
        copy_paste    = 0.3,
        warmup_epochs = 5,
        trainer       = MPDIoUTrainer,
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")