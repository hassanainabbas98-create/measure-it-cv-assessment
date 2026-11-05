"""
PyTorch Dataset for "Measure It" Assessment (xis.ai v2.0)
Loads undistorted images and instance segmentation ground-truth targets.
Strictly respects train/val/test split files.
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class MeasureItDataset(Dataset):
    def __init__(self, split_file, images_dir="dataset/images_undistorted", 
                 raw_labels_dir="dataset/annotations/raw_labels",
                 target_size=(512, 288)):
        self.images_dir = images_dir
        self.raw_labels_dir = raw_labels_dir
        self.target_size = target_size  # (width, height) for fast, stable CPU training
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        with open(split_file, "r") as f:
            self.filenames = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        base_name = os.path.splitext(filename)[0]
        
        # Load image
        img_path = os.path.join(self.images_dir, filename)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Load annotation
        json_path = os.path.join(self.raw_labels_dir, f"{base_name}.json")
        with open(json_path, "r") as f:
            ann_data = json.load(f)

        # Scale factors
        tgt_w, tgt_h = self.target_size
        sx = tgt_w / orig_w
        sy = tgt_h / orig_h

        resized_img = cv2.resize(img_rgb, (tgt_w, tgt_h))
        img_tensor = torch.as_tensor(resized_img.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        boxes = []
        labels = []
        masks = []
        areas = []

        for ann in ann_data["annotations"]:
            cat_id = ann["category_id"]
            pts = np.array(ann["points"], dtype=np.float32)
            # Scale points
            scaled_pts = pts.copy()
            scaled_pts[:, 0] *= sx
            scaled_pts[:, 1] *= sy

            # Compute scaled bounding box [x_min, y_min, x_max, y_max]
            x_min = float(np.min(scaled_pts[:, 0]))
            y_min = float(np.min(scaled_pts[:, 1]))
            x_max = float(np.max(scaled_pts[:, 0]))
            y_max = float(np.max(scaled_pts[:, 1]))

            # Guard against tiny/degenerated boxes
            if (x_max - x_min) < 1.0 or (y_max - y_min) < 1.0:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(cat_id)

            # Generate binary mask of shape (tgt_h, tgt_w)
            mask = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
            cv2.fillPoly(mask, [scaled_pts.astype(np.int32)], 1)
            masks.append(mask)

            area = cv2.contourArea(scaled_pts.astype(np.int32))
            areas.append(area)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "orig_size": torch.tensor([orig_h, orig_w]),
            "filename": filename
        }

        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))
