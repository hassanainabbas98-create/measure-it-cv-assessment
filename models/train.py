"""
Deterministic PyTorch Mask R-CNN Training Pipeline for "Measure It" (xis.ai v2.0)
Item 3: Trains custom instance segmentation model.
Strictly non-YOLO, non-Roboflow. Pinned seed = 42.
Trains on train.txt (52 images), validates on val.txt (13 images).
NEVER accesses test.txt during training.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader

from models.dataset import MeasureItDataset, collate_fn

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def build_model(num_classes=3, trainable_backbone_layers=0, min_size=288, max_size=512):
    print(f"[*] Building torchvision.models.detection.maskrcnn_resnet50_fpn...", flush=True)
    print(f"    - Pretrained: MaskRCNN_ResNet50_FPN_Weights.DEFAULT", flush=True)
    print(f"    - Trainable backbone layers: {trainable_backbone_layers}", flush=True)
    print(f"    - Input size range: min={min_size}, max={max_size}", flush=True)
    print(f"    - Classes: {num_classes} (0: bg, 1: charger_box, 2: cnic_card)", flush=True)

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size
    )

    if trainable_backbone_layers == 0:
        for param in model.backbone.parameters():
            param.requires_grad = False

    # 1. Replace Box Predictor Head
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # 2. Replace Mask Predictor Head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced, num_classes)

    return model

def calculate_mask_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)

@torch.no_grad()
def evaluate_validation(model, val_loader, device):
    model.eval()
    all_ious = []
    
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        predictions = model(images)
        
        for pred, target in zip(predictions, targets):
            gt_masks = target["masks"].numpy()
            gt_labels = target["labels"].numpy()
            
            pred_scores = pred["scores"].cpu().numpy()
            pred_labels = pred["labels"].cpu().numpy()
            pred_masks = (pred["masks"].cpu().squeeze(1).numpy() > 0.5).astype(np.uint8)
            
            for gt_idx, gt_label in enumerate(gt_labels):
                gt_m = gt_masks[gt_idx]
                match_indices = np.where((pred_labels == gt_label) & (pred_scores > 0.4))[0]
                if len(match_indices) > 0:
                    best_match = match_indices[0]
                    iou = calculate_mask_iou(pred_masks[best_match], gt_m)
                else:
                    iou = 0.0
                all_ious.append(iou)

    mean_val_iou = float(np.mean(all_ious)) if all_ious else 0.0
    return mean_val_iou

def train_model(config_path="models/config.json", weights_dir="models/weights"):
    set_seed(SEED)
    os.makedirs(weights_dir, exist_ok=True)

    with open(config_path, "r") as f:
        config = json.load(f)

    hp = config["hyperparameters"]
    batch_size = hp.get("batch_size", 2)
    num_epochs = hp.get("num_epochs", 8)
    lr = hp.get("learning_rate", 0.005)
    momentum = hp.get("momentum", 0.9)
    weight_decay = hp.get("weight_decay", 0.0005)
    trainable_backbone = hp.get("trainable_backbone_layers", 0)
    min_size = hp.get("min_size", 288)
    max_size = hp.get("max_size", 512)
    tgt_w = hp.get("input_width", 512)
    tgt_h = hp.get("input_height", 288)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Execution Device: {device}", flush=True)

    # Datasets (Strictly train and val only!)
    train_dataset = MeasureItDataset(config["dataset"]["train_split"], target_size=(tgt_w, tgt_h))
    val_dataset = MeasureItDataset(config["dataset"]["val_split"], target_size=(tgt_w, tgt_h))

    print(f"[+] Training Set:   {len(train_dataset)} instances (from {config['dataset']['train_split']})", flush=True)
    print(f"[+] Validation Set: {len(val_dataset)} instances (from {config['dataset']['val_split']})", flush=True)
    print(f"[*] Test set is strictly isolated for models/evaluate.py", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = build_model(
        num_classes=config["num_classes"],
        trainable_backbone_layers=trainable_backbone,
        min_size=min_size,
        max_size=max_size
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=hp.get("lr_step_size", 4),
        gamma=hp.get("lr_gamma", 0.5)
    )

    history = {
        "epochs": [],
        "train_loss": [],
        "val_mask_iou": [],
        "epoch_duration_s": []
    }

    best_val_iou = -1.0
    best_weights_path = os.path.join(weights_dir, "best_model.pth")

    print("\n" + "=" * 65, flush=True)
    print("STARTING DETERMINISTIC MASK R-CNN TRAINING", flush=True)
    print("=" * 65, flush=True)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            batch_count += 1

        lr_scheduler.step()
        avg_loss = epoch_loss / max(1, batch_count)
        val_iou = evaluate_validation(model, val_loader, device)
        duration = time.time() - epoch_start

        history["epochs"].append(epoch)
        history["train_loss"].append(round(avg_loss, 4))
        history["val_mask_iou"].append(round(val_iou, 4))
        history["epoch_duration_s"].append(round(duration, 2))

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Loss: {avg_loss:.4f} | Val Mask IoU: {val_iou:.4f} | Time: {duration:.1f}s", flush=True)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mask_iou": best_val_iou,
                "config": config
            }, best_weights_path)
            print(f"  [+] Checkpoint saved: New best Val Mask IoU = {best_val_iou:.4f}", flush=True)

    print("=" * 65, flush=True)
    print(f"[+] Training complete. Best Val Mask IoU: {best_val_iou:.4f}", flush=True)
    print(f"[+] Best weights saved to: {best_weights_path}", flush=True)

    # Save training log
    log_path = "models/training_log.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[+] Saved training history log to {log_path}", flush=True)

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["epochs"], history["train_loss"], "r-o", label="Train Loss")
    plt.title("Mask R-CNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["epochs"], history["val_mask_iou"], "b-s", label="Val Mask IoU")
    plt.title("Validation Mask IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    curves_path = "models/training_curves.png"
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"[+] Saved training curves plot to {curves_path}", flush=True)

    return best_weights_path, best_val_iou

if __name__ == "__main__":
    train_model()
