# cnn_summer_yolo_mpdiou_2cls_v20.py
#
# Purpose: Train YOLOv8m with MPDIoU loss on summer tiles, 2 classes
# Version 20: MPDIoU (Minimum Point Distance IoU) loss function
# MPDIoU improves bounding box regression for small objects by:
#   - incorporating minimum point distance between predicted and target boxes
#   - better handling of objects with extreme aspect ratios
#   - faster convergence for small object localization
# Reference: MPDIoU — A Loss for Efficient and Accurate Bounding Box
#            Regression (Siliang & Yong, arXiv:2307.07667)
# Implementation: custom trainer overriding bbox_iou in Ultralytics
# Uses same settings as best YOLOv8 model (v6)
# Pretrained weights: COCO (yolov8m.pt)
# CV: 5-fold spatially-aware (knndm)
# Output: results/summer_yolo_mpdiou_2cls_v20/

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.loss import BboxLoss
from ultralytics.models.yolo.detect import DetectionTrainer
import os

# ── MPDIoU implementation ──────────────────────────────────────────────────────

def mpdiou(box1, box2, eps=1e-7):
    """
    MPDIoU loss — Minimum Point Distance IoU
    Adds minimum point distance term to standard IoU
    Better for small objects with varying aspect ratios

    Args:
        box1: predicted boxes (x1, y1, x2, y2) normalized
        box2: target boxes (x1, y1, x2, y2) normalized
    Returns:
        MPDIoU loss value
    """
    # standard IoU computation
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    # intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    # MPDIoU additional terms — minimum point distances
    # d1: distance between top-left corners
    # d2: distance between bottom-right corners
    # normalized by diagonal of enclosing box
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps  # diagonal squared

    d1 = (b1_x1 - b2_x1) ** 2 + (b1_y1 - b2_y1) ** 2
    d2 = (b1_x2 - b2_x2) ** 2 + (b1_y2 - b2_y2) ** 2

    return iou - (d1 + d2) / c2


class MPDIoUBboxLoss(BboxLoss):
    """Custom BboxLoss using MPDIoU instead of default CIoU"""

    def iou_loss(self, pred_bboxes, target_bboxes, target_scores,
                 target_scores_sum, fg_mask):
        """Override IoU loss with MPDIoU"""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        return loss_iou, iou


class MPDIoUTrainer(DetectionTrainer):
    """Custom trainer that replaces bbox loss with MPDIoU"""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        # replace bbox loss with MPDIoU version
        if hasattr(model, 'criterion') and hasattr(model.criterion, 'bbox_loss'):
            model.criterion.bbox_loss = MPDIoUBboxLoss(
                model.criterion.bbox_loss.reg_max,
                model.criterion.bbox_loss.use_dfl
            )
        return model


# ── training loop ──────────────────────────────────────────────────────────────
DSS         = "/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz"
DATA_BASE   = f"{DSS}/data/cv_folds_2cls"
OUTPUT_BASE = f"{DSS}/results/summer_yolo_mpdiou_2cls_v20"

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
        trainer       = MPDIoUTrainer,  # use custom MPDIoU trainer
    )
    print(f"Fold {fold} complete.")

print("All folds complete.")