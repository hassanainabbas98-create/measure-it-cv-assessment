"""
Interactive OpenCV Polygon Annotation Tool for "Measure It" (xis.ai v2.0)
Labels planar rectangular targets:
  - Category 1: Target Object ('charger_box', 117.0 x 82.0 mm)
  - Category 2: Reference Object ('cnic_card', 86.0 x 54.0 mm)
Outputs COCO-format JSON and semantic mask PNGs at native 4032x2268 resolution.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import cv2

CATEGORIES = [
    {"id": 1, "name": "charger_box", "supercategory": "object"},
    {"id": 2, "name": "cnic_card", "supercategory": "reference"}
]

class PolygonAnnotator:
    def __init__(self, image_dir="dataset/images_undistorted", 
                 output_dir="dataset/annotations",
                 masks_dir="dataset/masks",
                 disp_width=1280, disp_height=720):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.raw_labels_dir = os.path.join(output_dir, "raw_labels")
        self.masks_dir = masks_dir
        self.disp_width = disp_width
        self.disp_height = disp_height
        
        os.makedirs(self.raw_labels_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)

        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*.jpg")) +
            glob.glob(os.path.join(image_dir, "*.jpeg")) +
            glob.glob(os.path.join(image_dir, "*.png"))
        )

        self.current_clicks = []
        self.current_scale = 1.0
        self.window_name = "Measure It - Annotation Tool"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.current_clicks) < 8:
                orig_x = int(round(x / self.current_scale))
                orig_y = int(round(y / self.current_scale))
                self.current_clicks.append((orig_x, orig_y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click resets clicks for current image
            if len(self.current_clicks) > 0:
                self.current_clicks = []

    def get_annotated_filenames(self):
        saved_files = glob.glob(os.path.join(self.raw_labels_dir, "*.json"))
        return {os.path.splitext(os.path.basename(f))[0] for f in saved_files}

    def save_annotation(self, img_path, orig_shape):
        orig_h, orig_w = orig_shape
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]

        charger_box_pts = self.current_clicks[:4]
        cnic_card_pts = self.current_clicks[4:8]

        def get_ann_data(pts, cat_id):
            poly = np.array(pts, dtype=np.float32)
            # COCO segmentation format: flattened list [x1, y1, x2, y2, ...]
            seg = [coord for pt in pts for coord in pt]
            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            bbox = [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)]
            area = float(cv2.contourArea(poly.astype(np.int32)))
            return {
                "category_id": cat_id,
                "segmentation": [seg],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "points": pts
            }

        ann_charger = get_ann_data(charger_box_pts, 1)
        ann_cnic = get_ann_data(cnic_card_pts, 2)

        data = {
            "file_name": filename,
            "width": orig_w,
            "height": orig_h,
            "annotations": [ann_charger, ann_cnic]
        }

        # 1. Save per-image raw label JSON
        json_path = os.path.join(self.raw_labels_dir, f"{base_name}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # 2. Save class semantic mask (1 = charger_box, 2 = cnic_card)
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(charger_box_pts, dtype=np.int32)], 1)
        cv2.fillPoly(mask, [np.array(cnic_card_pts, dtype=np.int32)], 2)
        mask_path = os.path.join(self.masks_dir, f"{base_name}.png")
        cv2.imwrite(mask_path, mask)

        print(f"[+] Saved annotation & mask for: {filename}")

    def compile_coco_dataset(self):
        json_files = sorted(glob.glob(os.path.join(self.raw_labels_dir, "*.json")))
        images = []
        annotations = []
        ann_id = 1

        for img_id, jf in enumerate(json_files, 1):
            with open(jf, "r") as f:
                data = json.load(f)

            images.append({
                "id": img_id,
                "file_name": data["file_name"],
                "width": data["width"],
                "height": data["height"]
            })

            for ann in data["annotations"]:
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": ann["category_id"],
                    "segmentation": ann["segmentation"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": 0
                })
                ann_id += 1

        coco_data = {
            "info": {
                "description": "Measure It (xis.ai v2.0) Segmentation Dataset",
                "version": "1.0",
                "year": 2026
            },
            "categories": CATEGORIES,
            "images": images,
            "annotations": annotations
        }

        coco_path = os.path.join(self.output_dir, "annotations_coco.json")
        with open(coco_path, "w") as f:
            json.dump(coco_data, f, indent=2)

        print(f"\n[+] Successfully compiled COCO annotations: {coco_path}")
        print(f"    - Total Images: {len(images)}")
        print(f"    - Total Annotations: {len(annotations)}")
        return coco_path

    def run(self):
        if not self.image_paths:
            print(f"[-] No images found in {self.image_dir}")
            return

        annotated_set = self.get_annotated_filenames()
        print(f"[*] Found {len(self.image_paths)} total images in {self.image_dir}")
        print(f"[*] Already annotated: {len(annotated_set)} images")

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        idx = 0
        while idx < len(self.image_paths):
            img_path = self.image_paths[idx]
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]

            if base_name in annotated_set:
                idx += 1
                continue

            orig_img = cv2.imread(img_path)
            if orig_img is None:
                print(f"[-] Could not load {filename}, skipping.")
                idx += 1
                continue

            orig_h, orig_w = orig_img.shape[:2]
            scale_w = self.disp_width / orig_w
            scale_h = self.disp_height / orig_h
            self.current_scale = min(scale_w, scale_h)
            disp_w = int(orig_w * self.current_scale)
            disp_h = int(orig_h * self.current_scale)
            resized_base = cv2.resize(orig_img, (disp_w, disp_h))

            self.current_clicks = []

            while True:
                display = resized_base.copy()

                # Draw charger_box clicks (Green)
                c_pts = [
                    (int(round(pt[0] * self.current_scale)), int(round(pt[1] * self.current_scale)))
                    for pt in self.current_clicks[:4]
                ]
                for pt in c_pts:
                    cv2.circle(display, pt, 5, (0, 255, 0), -1)
                if len(c_pts) > 1:
                    for i in range(len(c_pts) - 1):
                        cv2.line(display, c_pts[i], c_pts[i + 1], (0, 255, 0), 2)
                if len(c_pts) == 4:
                    cv2.line(display, c_pts[3], c_pts[0], (0, 255, 0), 2)

                # Draw cnic_card clicks (Blue / Cyan)
                card_pts = [
                    (int(round(pt[0] * self.current_scale)), int(round(pt[1] * self.current_scale)))
                    for pt in self.current_clicks[4:8]
                ]
                for pt in card_pts:
                    cv2.circle(display, pt, 5, (255, 200, 0), -1)
                if len(card_pts) > 1:
                    for i in range(len(card_pts) - 1):
                        cv2.line(display, card_pts[i], card_pts[i + 1], (255, 200, 0), 2)
                if len(card_pts) == 4:
                    cv2.line(display, card_pts[3], card_pts[0], (255, 200, 0), 2)

                # Draw Top HUD Banner
                cv2.rectangle(display, (0, 0), (disp_w, 45), (30, 30, 30), -1)
                hud_text_img = f"[{idx + 1}/{len(self.image_paths)}] {filename}"
                cv2.putText(display, hud_text_img, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

                n_clicks = len(self.current_clicks)
                if n_clicks < 4:
                    stage_text = f"TARGET (charger_box): Click corner {n_clicks + 1}/4 (Green)"
                    color = (0, 255, 0)
                elif n_clicks < 8:
                    stage_text = f"REFERENCE (cnic_card): Click corner {n_clicks - 4 + 1}/4 (Cyan/Blue)"
                    color = (255, 200, 0)
                else:
                    stage_text = "READY! Press [SPACE]/[ENTER] to Save & Next | [r]: Reset"
                    color = (0, 255, 255)

                cv2.putText(display, stage_text, (disp_w - 600, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                # Bottom HUD controls
                cv2.rectangle(display, (0, disp_h - 25), (disp_w, disp_h), (20, 20, 20), -1)
                controls_text = "Controls: [Left-Click]: Corner | [u]: Undo | [r]/[Right-Click]: Reset | [SPACE]/[ENTER]: Save | [q]/[ESC]: Exit"
                cv2.putText(display, controls_text, (15, disp_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(20) & 0xFF

                if key in [ord('q'), 27]:  # 'q' or ESC
                    print("[*] Exit requested by user.")
                    cv2.destroyAllWindows()
                    self.compile_coco_dataset()
                    return

                elif key == ord('r'):  # Reset
                    self.current_clicks = []

                elif key == ord('u'):  # Undo last click
                    if len(self.current_clicks) > 0:
                        self.current_clicks.pop()

                elif key in [32, 13]:  # SPACE or ENTER
                    if len(self.current_clicks) == 8:
                        self.save_annotation(img_path, (orig_h, orig_w))
                        annotated_set.add(base_name)
                        idx += 1
                        break
                    else:
                        print(f"[-] Incomplete clicks: {len(self.current_clicks)}/8. Need 4 corners for charger_box and 4 for cnic_card.")

        cv2.destroyAllWindows()
        print("\n[+] All images in directory processed!")
        self.compile_coco_dataset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure It Interactive Annotation Tool")
    parser.add_argument("--image-dir", default="dataset/images_undistorted", help="Input image directory")
    parser.add_argument("--output-dir", default="dataset/annotations", help="COCO annotations output dir")
    parser.add_argument("--masks-dir", default="dataset/masks", help="Masks PNG output dir")
    parser.add_argument("--compile-only", action="store_true", help="Compile existing raw labels to COCO JSON")
    args = parser.parse_args()

    annotator = PolygonAnnotator(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        masks_dir=args.masks_dir
    )

    if args.compile_only:
        annotator.compile_coco_dataset()
    else:
        annotator.run()
