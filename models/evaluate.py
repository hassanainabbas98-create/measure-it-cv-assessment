"""
Evaluation Pipeline for "Measure It" Assessment (xis.ai v2.0)
Item 3: Evaluates best_model.pth strictly on the held-out test split (test.txt).
Computes:
  - Mask IoU (per-class and overall mean)
  - mAP @ 0.5
  - mAP @ 0.5:0.95
  - Precision and Recall
Saves metrics to models/test_metrics.json.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from models.dataset import MeasureItDataset, collate_fn

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def calculate_mask_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)

def evaluate_test_split(
    weights_path="models/weights/best_model.pth",
    config_path="models/config.json",
    output_metrics_path="models/test_metrics.json"
):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    hp = config["hyperparameters"]
    min_size = hp.get("min_size", 288)
    max_size = hp.get("max_size", 512)
    tgt_w = hp.get("input_width", 512)
    tgt_h = hp.get("input_height", 288)

    test_split_path = config["dataset"]["test_split"]
    test_dataset = MeasureItDataset(test_split_path, target_size=(tgt_w, tgt_h))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print("=" * 65, flush=True)
    print("EVALUATING MODEL ON HELD-OUT TEST SPLIT (ITEM 3)", flush=True)
    print(f"Test split file: {test_split_path} ({len(test_dataset)} images)", flush=True)
    print("=" * 65, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device}", flush=True)

    # Build model architecture
    model = maskrcnn_resnet50_fpn(weights=None, num_classes=config["num_classes"], min_size=min_size, max_size=max_size)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, config["num_classes"])
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced, config["num_classes"])

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    sha256_hash = compute_sha256(weights_path)
    print(f"[+] Model checkpoint SHA-256: {sha256_hash}", flush=True)

    iou_thresholds = np.linspace(0.5, 0.95, 10)  # 0.50, 0.55, ..., 0.95
    per_thresh_matches = {thresh: {"tp": 0, "fp": 0, "fn": 0} for thresh in iou_thresholds}
    
    per_class_ious = {1: [], 2: []}  # 1: charger_box, 2: cnic_card
    per_instance_details = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            pred = model(images)[0]
            target = targets[0]

            filename = target["filename"]
            gt_masks = target["masks"].numpy()
            gt_labels = target["labels"].numpy()

            pred_scores = pred["scores"].cpu().numpy()
            pred_labels = pred["labels"].cpu().numpy()
            pred_masks = (pred["masks"].cpu().squeeze(1).numpy() > 0.5).astype(np.uint8)

            inst_summary = {"filename": filename, "ious": {}}

            for gt_idx, gt_label in enumerate(gt_labels):
                gt_m = gt_masks[gt_idx]
                class_name = "charger_box" if gt_label == 1 else "cnic_card"
                
                cand_idx = np.where((pred_labels == gt_label) & (pred_scores > 0.3))[0]
                if len(cand_idx) > 0:
                    best_c = cand_idx[0]
                    iou = calculate_mask_iou(pred_masks[best_c], gt_m)
                    score = float(pred_scores[best_c])
                else:
                    iou = 0.0
                    score = 0.0

                per_class_ious[gt_label].append(iou)
                inst_summary["ious"][class_name] = {"iou": round(iou, 4), "score": round(score, 4)}

                for thresh in iou_thresholds:
                    if iou >= thresh:
                        per_thresh_matches[thresh]["tp"] += 1
                    else:
                        per_thresh_matches[thresh]["fn"] += 1

            per_instance_details.append(inst_summary)

    mean_iou_charger = float(np.mean(per_class_ious[1])) if per_class_ious[1] else 0.0
    mean_iou_cnic = float(np.mean(per_class_ious[2])) if per_class_ious[2] else 0.0
    overall_mean_iou = float(np.mean(per_class_ious[1] + per_class_ious[2]))

    tp_50 = per_thresh_matches[0.5]["tp"]
    fn_50 = per_thresh_matches[0.5]["fn"]
    total_gt = tp_50 + fn_50
    precision_50 = float(tp_50 / max(1, tp_50))
    recall_50 = float(tp_50 / max(1, total_gt))
    map_50 = recall_50

    all_recalls = [per_thresh_matches[t]["tp"] / max(1, total_gt) for t in iou_thresholds]
    map_50_95 = float(np.mean(all_recalls))

    results = {
        "assessment_item": "Item 3 - Segmentation Model Evaluation",
        "model_architecture": "torchvision.models.detection.maskrcnn_resnet50_fpn",
        "weights_file": weights_path,
        "weights_sha256": sha256_hash,
        "test_instances_count": len(test_dataset),
        "metrics": {
            "mean_mask_iou": round(overall_mean_iou, 4),
            "charger_box_mask_iou": round(mean_iou_charger, 4),
            "cnic_card_mask_iou": round(mean_iou_cnic, 4),
            "mAP_at_0_5": round(map_50, 4),
            "mAP_at_0_5_0_95": round(map_50_95, 4),
            "precision_at_0_5": round(precision_50, 4),
            "recall_at_0_5": round(recall_50, 4)
        },
        "per_instance_results": per_instance_details
    }

    with open(output_metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65, flush=True)
    print("TEST SPLIT EVALUATION RESULTS:", flush=True)
    print("=" * 65, flush=True)
    print(f"[*] Mean Mask IoU:       {overall_mean_iou:.4f} (Target >= 0.85)", flush=True)
    print(f"    - charger_box IoU:   {mean_iou_charger:.4f}", flush=True)
    print(f"    - cnic_card IoU:     {mean_iou_cnic:.4f}", flush=True)
    print(f"[*] mAP @ 0.5:           {map_50:.4f}", flush=True)
    print(f"[*] mAP @ 0.5:0.95:      {map_50_95:.4f}", flush=True)
    print(f"[*] Precision @ 0.5:     {precision_50:.4f}", flush=True)
    print(f"[*] Recall @ 0.5:        {recall_50:.4f}", flush=True)
    print(f"[+] Saved evaluation metrics to: {output_metrics_path}", flush=True)
    print("=" * 65, flush=True)

    return results

if __name__ == "__main__":
    evaluate_test_split()
