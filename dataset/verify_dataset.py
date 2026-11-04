"""
Dataset Verification Script for "Measure It" (xis.ai v2.0)
Audits Item 2: Verifies all 75 images, train/val/test splits, COCO annotations, and mask PNGs.
"""

import os
import json
import cv2
import numpy as np

def verify_dataset():
    print("=" * 60)
    print("AUDITING DATASET & ANNOTATIONS INTEGRITY (ITEM 2)")
    print("=" * 60)

    coco_path = "dataset/annotations/annotations_coco.json"
    if not os.path.exists(coco_path):
        raise FileNotFoundError(f"Missing {coco_path}")

    with open(coco_path, "r") as f:
        coco = json.load(f)

    total_images = len(coco["images"])
    total_annotations = len(coco["annotations"])
    print(f"[*] COCO Dataset: {total_images} images, {total_annotations} annotations")

    # 1. Audit Splits
    splits = {}
    for s in ["train", "val", "test"]:
        split_path = f"dataset/splits/{s}.txt"
        with open(split_path, "r") as f:
            splits[s] = [line.strip() for line in f if line.strip()]
        print(f"    - {s.upper()} split: {len(splits[s])} images")

    all_split_images = set(splits["train"] + splits["val"] + splits["test"])
    coco_images = {img["file_name"] for img in coco["images"]}

    assert len(all_split_images) == 75, f"Expected 75 total images, got {len(all_split_images)}"
    assert all_split_images == coco_images, "Mismatch between split files and COCO dataset!"
    print("[+] All 75 split images perfectly match COCO images.")

    # 2. Audit Categories and Geometry
    cat1_count = sum(1 for a in coco["annotations"] if a["category_id"] == 1)
    cat2_count = sum(1 for a in coco["annotations"] if a["category_id"] == 2)
    print(f"[+] Category 1 ('charger_box') count: {cat1_count} instances")
    print(f"[+] Category 2 ('cnic_card') count:   {cat2_count} instances")
    assert cat1_count == 75, f"Expected 75 charger_box instances, found {cat1_count}"
    assert cat2_count == 75, f"Expected 75 cnic_card instances, found {cat2_count}"

    for ann in coco["annotations"]:
        poly = ann["segmentation"][0]
        assert len(poly) == 8, f"Polygon must have 8 coordinates (4 corners), got {len(poly)}"
        bbox = ann["bbox"]
        assert bbox[2] > 0 and bbox[3] > 0, f"Invalid bbox: {bbox}"
        assert ann["area"] > 0, f"Invalid area: {ann['area']}"

    print("[+] All 150 polygons verified: valid 4-corner vertices, positive bounding boxes, non-zero areas.")

    # 3. Audit Semantic Mask PNGs
    mask_files = set(os.listdir("dataset/masks"))
    print(f"[*] Auditing semantic masks in dataset/masks/ ({len(mask_files)} files)...")
    for fn in all_split_images:
        mask_name = os.path.splitext(fn)[0] + ".png"
        assert mask_name in mask_files, f"Missing mask: {mask_name}"
        m = cv2.imread(os.path.join("dataset/masks", mask_name), cv2.IMREAD_GRAYSCALE)
        unique_vals = set(np.unique(m))
        assert 1 in unique_vals, f"Mask {mask_name} missing class 1 (charger_box)"
        assert 2 in unique_vals, f"Mask {mask_name} missing class 2 (cnic_card)"

    print("[+] All 75 mask PNGs verified containing class 1 (charger_box) and class 2 (cnic_card).")
    print("=" * 60)
    print("STATUS: DATASET VERIFICATION PASSED (AUDIT COMPLETE)")
    print("=" * 60)

if __name__ == "__main__":
    verify_dataset()
