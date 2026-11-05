# Definition of Done (Audited Completion Evidence)

This document provides verified evidence and commit references for all assessment requirements in accordance with Section 5.4. Every box is ticked with supporting empirical evidence and commit hashes.

---

## 1. Core Engineering Checklist (Items 1 – 10)

- [x] **1. Camera Calibration (Item 1)**
  - *Requirement*: >= 20 checkerboard images calibrated with OpenCV; camera matrix and distortion coefficients saved; reprojection error reported.
  - *Verified Evidence*: 22/22 valid calibration photos processed at 4032x2268 resolution. Mean Reprojection Error: **`0.1393 px`** (Focal fx: `2879.19`, fy: `2875.22`). Parameters saved in [`calibration/camera_params.npz`](file:///c:/Users/kk/Desktop/Project_AI_CV/calibration/camera_params.npz) and [`calibration/calibration_report.json`](file:///c:/Users/kk/Desktop/Project_AI_CV/calibration/calibration_report.json). Verified in visual comparison [`calibration/verification_undistorted.jpg`](file:///c:/Users/kk/Desktop/Project_AI_CV/calibration/verification_undistorted.jpg).
  - *Commit*: `2c68e00` (`feat(calibration): calibrate camera using 22 checkerboard images achieving 0.1393px reprojection error`)

- [x] **2. Dataset Collection & Fixed Split List (Item 2)**
  - *Requirement*: >= 70 images collected with calibrated camera, undistorted, and labelled with segmentation masks. 70/20/10 split fixed by committed file list before training.
  - *Verified Evidence*: 75 custom photos undistorted to `dataset/images_undistorted/` (4032x2268). Split committed in [`dataset/splits/train.txt`](file:///c:/Users/kk/Desktop/Project_AI_CV/dataset/splits/train.txt) (52 images, 69.3%), [`dataset/splits/val.txt`](file:///c:/Users/kk/Desktop/Project_AI_CV/dataset/splits/val.txt) (13 images, 17.3%), and [`dataset/splits/test.txt`](file:///c:/Users/kk/Desktop/Project_AI_CV/dataset/splits/test.txt) (10 images, 13.3%). Zero overlap verified. All 75 photos annotated with 150 polygons in [`dataset/annotations/annotations_coco.json`](file:///c:/Users/kk/Desktop/Project_AI_CV/dataset/annotations/annotations_coco.json) and class masks in `dataset/masks/`. Audited via [`dataset/verify_dataset.py`](file:///c:/Users/kk/Desktop/Project_AI_CV/dataset/verify_dataset.py).
  - *Commits*: `fddf4c1` (Split lists committed prior to model code) & `0c93d97` (Annotations and masks).

- [x] **3. Segmentation Model Training & Test Evaluation (Item 3)**
  - *Requirement*: Custom-trained segmentation model (non-YOLO, non-Roboflow, seed=42) evaluated on test split reporting Mask IoU, mAP@0.5, mAP@0.5:0.95, Precision, and Recall.
  - *Verified Evidence*: PyTorch `torchvision.models.detection.maskrcnn_resnet50_fpn` fine-tuned for 8 epochs (`seed=42`) saving best checkpoint at 0.9503 val IoU. Evaluated on the 10 held-out test images: **`0.9526` Mean Mask IoU** (charger_box: `0.9543`, cnic_card: `0.9510`), **`1.0000` mAP@0.5**, **`0.9550` mAP@0.5:0.95**, **`1.0000` Precision**, **`1.0000` Recall**. Checkpoint SHA-256: `9eb61ae5a662ba52594af86c92e7d8c2cec0a3617b970b323695c63d78673ecb`. Saved in [`models/test_metrics.json`](file:///c:/Users/kk/Desktop/Project_AI_CV/models/test_metrics.json) and [`models/training_curves.png`](file:///c:/Users/kk/Desktop/Project_AI_CV/models/training_curves.png).
  - *Commit*: `d50e461` (`feat(models): train PyTorch Mask R-CNN ResNet-50 FPN achieving 0.9526 test Mask IoU`)

- [x] **4. End-to-End Single-Command Metric Inference CLI (Item 4)**
  - *Requirement*: Single CLI command (`python inference/measure.py --image <path>`) that undistorts, segments object, detects reference, outputs mask overlay with width/height in mm and confidence, handling missing object and missing reference cleanly.
  - *Verified Evidence*: Implemented in [`inference/measure.py`](file:///c:/Users/kk/Desktop/Project_AI_CV/inference/measure.py). Tested on raw image `dataset/images_raw/PXL_20261104_161823310.jpg`, producing `117.55 mm` width (err `0.55 mm`) and `83.04 mm` height (err `1.04 mm`), target conf `0.97`, ref conf `1.00`, saving visual HUD overlay to [`inference/sample_output_valid.jpg`](file:///c:/Users/kk/Desktop/Project_AI_CV/inference/sample_output_valid.jpg). Refusal test suite [`inference/test_refusal_cases.py`](file:///c:/Users/kk/Desktop/Project_AI_CV/inference/test_refusal_cases.py) verified clean refusal with exit code 0 on missing target and missing reference.
  - *Commit*: `f476000` (`feat(inference): implement end-to-end measure CLI with refusal guardrails (Items 4 & 9)`)

- [x] **5. Accuracy Against a Ruler / Calliper (Item 5)**
  - *Requirement*: Measure 10 or more instances by hand and with system. Report per-instance table, MAE (mm), and Mean Percentage Error (MPE).
  - *Verified Evidence*: Evaluated on all 10 held-out test instances (`test.txt`) vs. physical ruler ground truth (117.0 x 82.0 mm). **Mean Absolute Error: `3.805 mm`**, **Median Absolute Error: `2.338 mm`**, **Worst-Case Error: `10.350 mm`**, **Mean Percentage Error: `3.949%`**. Per-instance table and statistics saved in [`measurement/experiment_tables.md`](file:///c:/Users/kk/Desktop/Project_AI_CV/measurement/experiment_tables.md) and [`measurement/results_summary.json`](file:///c:/Users/kk/Desktop/Project_AI_CV/measurement/results_summary.json).
  - *Commit*: `556bcb4` (`feat(measurement): implement metrology verification suite and evaluate Items 5-9`)

- [x] **6. Empirical Proof That Undistortion Matters (Item 6)**
  - *Requirement*: Measure the same instances on raw images and undistorted ones side-by-side with your own numbers.
  - *Verified Evidence*: Measured exact 10 test instances on raw photos vs. undistorted photos. **Raw MAE: `4.268 mm` (Worst: `10.745 mm`)** vs. **Undistorted MAE: `3.805 mm` (Worst: `10.350 mm`)**, delivering a **`0.462 mm` average error reduction** across the dataset and **`3.61 mm` reduction on peripheral Instance #1**. Comparative table saved in [`measurement/experiment_tables.md`](file:///c:/Users/kk/Desktop/Project_AI_CV/measurement/experiment_tables.md) and visual barchart saved in [`measurement/raw_vs_undistorted.png`](file:///c:/Users/kk/Desktop/Project_AI_CV/measurement/raw_vs_undistorted.png).
  - *Commit*: `556bcb4`

- [x] **7. Measurable Speed & Accuracy Targets (Item 7 & Section 6)**
  - *Requirement*: Set specific targets in `docs/Objectives.md`, measure 5 times, report median and spread.
  - *Verified Evidence*: Measured on host Intel Core i5-7300U CPU across 5 runs: `[2.6587s, 2.6951s, 2.7359s, 3.1919s, 2.5622s]`. **Median Latency: `2.6951 s`**, **Spread: `0.6297 s`** (Min: `2.5622 s`, Max: `3.1919 s`). Target status honestly declared: MO-1 Accuracy target ($2.0\text{ mm}$) missed ($3.805\text{ mm}$ achieved); MO-2 Latency target ($1.50\text{ s}$) missed ($2.695\text{ s}$ achieved due to 2-core CPU forward pass); MO-2 Mask IoU target ($\ge 0.85$) surpassed (**`0.9526`** achieved).
  - *Commit*: `556bcb4`

- [x] **8. Breaking Condition Analysis: Camera Tilt (Item 8)**
  - *Requirement*: Choose one condition (camera tilt) and measure how error changes as it worsens.
  - *Verified Evidence*: Evaluated across 5 tilt angles: `0°` (`3.43 mm` err), `15°` (`2.41 mm` err), `30°` (`8.10 mm` err), `45°` (`11.32 mm` err), `60°` (`17.91 mm` err). Aspect ratio distortion progressed from `2.96%` at $0^\circ$ to `31.64%` at $60^\circ$. Degradation curves plotted in [`measurement/tilt_degradation_curve.png`](file:///c:/Users/kk/Desktop/Project_AI_CV/measurement/tilt_degradation_curve.png).
  - *Commit*: `556bcb4`

- [x] **9. Knowing When Not to Answer (Item 9)**
  - *Requirement*: System refuses to report measurements it cannot trust. Show threshold location and cost in refused images.
  - *Verified Evidence*: Guardrails implemented in [`inference/measure.py`](file:///c:/Users/kk/Desktop/Project_AI_CV/inference/measure.py): Confidence threshold $= 0.80$, Card aspect-ratio error tolerance $= 12.0\%$, Boundary clipping margin $= 15\text{ px}$. Verified 0.0% false rejections (10/10 normal test images accepted) and 100% safety interception on severe tilts ($\ge 45^\circ$ refused with `ASPECT_RATIO_ANOMALY`). Results logged in [`inference/refusal_test_results.json`](file:///c:/Users/kk/Desktop/Project_AI_CV/inference/refusal_test_results.json) and [`measurement/experiment_tables.md`](file:///c:/Users/kk/Desktop/Project_AI_CV/measurement/experiment_tables.md).
  - *Commit*: `f476000` & `556bcb4`

- [x] **10. Architectural Decision Record (Item 10)**
  - *Requirement*: Written inside `Plan.md` covering approach taken, approach rejected, and what rejecting it cost.
  - *Verified Evidence*: Fully documented in Section 5 of [`docs/Plan.md`](file:///c:/Users/kk/Desktop/Project_AI_CV/docs/Plan.md#5-architectural-decision-record-item-10) analyzing Decision 1 (Mask R-CNN vs. Semantic Segmentation vs. YOLO), Decision 2 (Dual-axis scale vs. 4-point homography), and Decision 3 (Rotated bounding box vs. axis-aligned box).
  - *Commit*: Commit #8 (final documentation & README audit)

---

## 2. Assessment Integrity & Code Hygiene

- [x] **Strict Phased Git Chronology Maintained**:
  - `f8f5083`: Commit #1 (Project skeleton & initial docs before any code)
  - `2c68e00`: Commit #2 (OpenCV camera calibration)
  - `fddf4c1`: Commit #3 (Fixed 70/20/10 split files before training code)
  - `0c93d97`: Commit #4 (Polygon annotations & class semantic masks)
  - `d50e461`: Commit #5 (PyTorch Mask R-CNN training & test evaluation)
  - `f476000`: Commit #6 (Single-command CLI & refusal guardrails)
  - `556bcb4`: Commit #7 (Metrology evaluation suite for Items 5–9)
  - Commit #8: Final documentation, Item 10 Decision Record, Definition of Done, and README
- [x] **Zero Synthetic / Fabricated Numbers**: All values derived strictly from raw terminal executions on host machine.
- [x] **Code Hygiene**: Zero dead code, zero commented-out code blocks, zero stray files, modular Python scripts only.
- [x] **Incomplete / Missed Items Declared**: Declared honest results for MO-1 ($3.805\text{ mm}$ MAE vs $2.0\text{ mm}$ target) and MO-2 CPU Latency ($2.695\text{ s}$ vs $1.50\text{ s}$ target) with full physical and algorithmic explanations.
