# Measure It — Industrial Segmentation & Metric Metrology (xis.ai v2.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11-green.svg)](https://opencv.org/)
[![Branch: dev-test01](https://img.shields.io/badge/git%20branch-dev--test01-brightgreen.svg)]()

Production-grade monocular metric measurement pipeline developed for the **"Measure It"** Technical Assessment by **xis.ai** (Version 2.0).

---

## 1. System & Hardware Specifications

As required by Section 6 of the assessment specification, all timings, calibration outputs, and model metrics were produced on the following environment:

| Specification Field | Recorded System Details |
| :--- | :--- |
| **Operating System** | Microsoft Windows 10 Pro (Version 10.0.19045) |
| **CPU** | Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz (2 Cores, 4 Threads) |
| **Local GPU** | Intel(R) HD Graphics 620 (Local execution on CPU) |
| **RAM** | 7.86 GB Visible Physical Memory (8 GB physical) |
| **Camera Model** | Google Pixel 6 Pro (Main wide sensor, locked 1x optical zoom) |
| **Native Resolution** | 4032 x 2268 pixels (locked focus, exposure, and white balance) |
| **Calibration Target** | 8x10 squares = 7x9 inner corners (`(7, 9)` in OpenCV), square size = 20.0 mm |

---

## 2. Dataset Summary

- **Target Object**: Rectangular Charger Box (rigid cardboard packaging box). Ground-truth measured with physical ruler/caliper: **`Width = 117.0 mm, Height = 82.0 mm`**.
  - *Why chosen*: Rigid rectangular geometry with crisp straight edges and matte surfaces, enabling unambiguous ground-truth verification.
- **Reference Object**: Standard CNIC ID Card (rigid PVC card, sensitive text masked with tape while keeping boundary edges fully intact) placed in the same plane as the target. Ground-truth: **`Width = 86.0 mm, Height = 54.0 mm`** (aspect ratio = `1.5926`).
- **Calibration Images**: 22 checkerboard photos in `calibration/images/`.
- **Dataset Images**: 75 custom photos in `dataset/images_undistorted/` (4032x2268).
- **Dataset Split**: Pinned deterministic 70/20/10 partition (`seed = 42`):
  - **Train split**: 52 images (69.3%) — [`dataset/splits/train.txt`](file:///c:/Users/kk/Desktop/Project_AI_CV/dataset/splits/train.txt)
  - **Val split**: 13 images (17.3%) — [`dataset/splits/val.txt`](file:///c:/Users/kk/Desktop/Project_AI_CV/dataset/splits/val.txt)
  - **Test split**: 10 images (13.3%) — [`dataset/splits/test.txt`](file:///c:/Users/kk/Desktop/Project_AI_CV/dataset/splits/test.txt)
  - *Audit*: Mutually exclusive, zero cross-set leakage verified by [`dataset/verify_dataset.py`](file:///c:/Users/kk/Desktop/Project_AI_CV/dataset/verify_dataset.py).
- **Labelling Tool**: Custom in-house interactive OpenCV tool ([`dataset/annotate_tool.py`](file:///c:/Users/kk/Desktop/Project_AI_CV/dataset/annotate_tool.py)) producing native COCO polygons and pixel-accurate semantic masks.
- **Tilt Breaking Condition Set**: 5 photos in `measurement/tilt_images_undistorted/` ($0^\circ, 15^\circ, 30^\circ, 45^\circ, 60^\circ$).

---

## 3. Quickstart & Installation

```powershell
# 1. Clone repository and checkout dev-test01 branch
git clone <repo_url>
cd Project_AI_CV
git checkout dev-test01

# 2. Create virtual environment (Python 3.10 recommended)
py -3.10 -m venv .venv
.\.venv\Scripts\activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 4. How to Reproduce Every Stage

Work down the pipeline step-by-step:

### Stage 1: Camera Calibration (Item 1)
Computes camera intrinsic matrix $K$, distortion coefficients $D$, and reprojection error from 22 checkerboard images:
```powershell
python calibration/calibrate.py
```
- **Output**: Achieves **`0.1393 px`** mean reprojection error (well below the $0.5\text{ px}$ target). Saves [`calibration/camera_params.npz`](file:///c:/Users/kk/Desktop/Project_AI_CV/calibration/camera_params.npz), [`calibration/calibration_report.json`](file:///c:/Users/kk/Desktop/Project_AI_CV/calibration/calibration_report.json), and [`calibration/verification_undistorted.jpg`](file:///c:/Users/kk/Desktop/Project_AI_CV/calibration/verification_undistorted.jpg).

### Stage 2: Dataset Undistortion & Split Generation (Item 2)
Undistorts raw images using calibrated parameters and generates deterministic splits:
```powershell
# Undistort raw photos
python calibration/undistort_dataset.py

# Verify / generate deterministic splits (seed=42)
python dataset/create_splits.py

# Audit annotations and masks integrity
python dataset/verify_dataset.py
```

### Stage 3: Model Training & Test Evaluation (Item 3)
Trains PyTorch Mask R-CNN ResNet-50 FPN on CPU (pinned `seed=42`, strictly non-YOLO, non-Roboflow) and evaluates the held-out test split:
```powershell
# Train model (trains on train.txt, validates on val.txt, never touches test.txt)
python models/train.py

# Evaluate checkpoint on held-out test split
python models/evaluate.py
```
- **Test Metrics**: **`0.9526` Mean Mask IoU** (Target $\ge 0.85$), **`1.0000` mAP@0.5**, **`0.9550` mAP@0.5:0.95**, **`1.0000` Precision**, **`1.0000` Recall**. Saved in [`models/test_metrics.json`](file:///c:/Users/kk/Desktop/Project_AI_CV/models/test_metrics.json).

### Stage 4: Single-Command Metric Measurement CLI (Item 4 & Item 9)
Takes any photo, undistorts it, segments both objects, calculates the millimetre-per-pixel scale, and outputs width/height with refusal guardrails:
```powershell
# Run measurement on a sample raw test image
python inference/measure.py --image dataset/images_raw/PXL_20261104_161823310.jpg --output inference/sample_output.jpg

# Run refusal guardrail test suite (verifies Cases 1 - 4)
python inference/test_refusal_cases.py
```

### Stage 5: Metrology Verification Suite (Items 5, 6, 7, 8, & 9)
Executes accuracy benchmarking, raw vs. undistorted comparison, runtime latency profiling, and camera tilt breaking tests:
```powershell
python measurement/evaluate_all.py
```

---

## 5. Summary of Experimental Results

| Assessment Item | Metric / Target | Real Empirical Result | Status |
| :--- | :--- | :---: | :---: |
| **Item 1: Calibration** | Reprojection Error | **0.1393 px** across 22 images | **MET** |
| **Item 3: Segmentation** | Test Split Mask IoU ($\ge 0.85$) | **0.9526** (mAP@0.5: **1.0000**) | **MET** |
| **Item 4: CLI Command** | End-to-end measure CLI | Functional with 4 refusal guards | **MET** |
| **Item 5: Calliper Accuracy** | MAE $\le 2.0\text{ mm}$, MPE $\le 2.5\%$ | **MAE: 3.805 mm** (Median: **2.338 mm**) | **MISSED (Honest report)** |
| **Item 6: Undistortion Proof** | Error reduction from calibration | **-0.462 mm** MAE improvement (up to -3.61 mm) | **PROVEN** |
| **Item 7: Speed Benchmark** | Median Latency $\le 1.50\text{ s}$ | **Median: 2.6951 s** (Spread: 0.6297 s) | **MISSED (CPU limitation)** |
| **Item 8: Breaking Condition** | Camera Tilt Degradation | Spikes from 3.43 mm (0°) to 17.91 mm (60°) | **PROVEN** |
| **Item 9: Refusal Guardrails** | False Rejection vs. Safety | 0.0% false rejections; 100% tilt rejections | **MET** |
| **Item 10: Decision Record** | Documented inside Plan | Complete in `docs/Plan.md` Section 5 | **MET** |

---

## 6. Model Weights & SHA-256 Verification

In strict compliance with Section 5.5 of the assessment:

- **Local Weights Path**: `models/weights/best_model.pth`
- **File Size**: 244.7 MB (244,767,070 bytes)
- **SHA-256 Checksum**:
  ```text
  9eb61ae5a662ba52594af86c92e7d8c2cec0a3617b970b323695c63d78673ecb
  ```
- **Verification Command**:
  ```powershell
  Get-FileHash models/weights/best_model.pth -Algorithm SHA256
  ```
- **Weights Release Asset**: [`best_model.pth` (v1.0.0 Direct Download)](https://github.com/hassanainabbas98-create/measure-it-cv-assessment/releases/download/v1.0.0/best_model.pth)
- **GitHub Release Page**: [v1.0.0 Release Notes & Assets](https://github.com/hassanainabbas98-create/measure-it-cv-assessment/releases/tag/v1.0.0)
- **Deterministic Reproduction**: Model weights can also be reproduced deterministically by running `python models/train.py` (`seed = 42`).

---

## 7. Loom 5-Minute Technical Defense Quick Reference (Section 7.2)

- **K1: Which file and line turns pixels into millimetres? What must be true about the scene?**
  - **File & Lines**: [`inference/measure.py` (Lines 124–125)](file:///c:/Users/kk/Desktop/Project_AI_CV/inference/measure.py#L124-L125) inside `convert_pixels_to_mm()`:
    ```python
    width_mm = obj_width_px * scale_mm_per_px
    height_mm = obj_height_px * scale_mm_per_px
    ```
    where `scale_mm_per_px = 0.5 * ((86.0 / ref_width_px) + (54.0 / ref_height_px))`.
  - **Scene Requirements**: (1) Strict coplanarity (both objects rest on the exact same planar depth $Z$); (2) Orthogonal optical axis (minimal camera tilt $<10^\circ$); (3) Uniform distance to sensor.
- **K2: Where in your code is undistortion applied, and how much did skipping it change your measurement error?**
  - **File & Line**: [`inference/measure.py` (Line 51)](file:///c:/Users/kk/Desktop/Project_AI_CV/inference/measure.py#L51) inside `undistort_image()`:
    ```python
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    ```
  - **Our Own Empirical Numbers**: Skipping undistortion increased Mean Absolute Error from **`3.805 mm`** to **`4.268 mm`** (a direct **`0.462 mm`** degradation across the dataset), with edge/peripheral instances like Instance #1 degrading by **`3.61 mm`** due to uncorrected barrel distortion ($k_1 = +0.118, k_2 = -0.398$).
- **K3: Name one thing in your own work you would reverse, and what changing it would cost.**
  - **Answer**: I would replace the isotropic single-scale factor with full **4-point planar perspective homography** (`cv2.findHomography` / `cv2.warpPerspective`). Rejecting homography cost us measurement accuracy when the camera is moderately tilted: at $30^\circ$ tilt, error rose to `8.10 mm` before the refusal threshold triggered at $45^\circ$. Adopting homography would rectify perspective foreshortening on tilted planes, but would cost increased sensitivity to reference corner point extraction noise.
- **K4: Name one thing you did not understand, and what you did about it.**
  - **Answer**: Initially, training Mask R-CNN ResNet-50 FPN on CPU took ~10 minutes per epoch due to gradient backpropagation through deep residual layers. I researched torchvision backbone freezing mechanisms, locked all earlier backbone feature stages (`trainable_backbone_layers = 0`), and downsampled the internal training transform to $512 \times 288$. This reduced epoch duration by over 60%, allowing clean convergence in under 4 minutes with 0.9503 validation Mask IoU.

---

## 8. Documentation Index

The required three documents under `docs/`:
1. [`docs/Plan.md`](file:///c:/Users/kk/Desktop/Project_AI_CV/docs/Plan.md): Project schedule, execution timeline, and Section 5 Architectural Decision Record (Item 10).
2. [`docs/Objectives.md`](file:///c:/Users/kk/Desktop/Project_AI_CV/docs/Objectives.md): System hardware profile, MO-1 & MO-2 specification tables, and full empirical tables (Items 1, 3, 5, 6, 7, 8, 9).
3. [`docs/Definition_of_Done.md`](file:///c:/Users/kk/Desktop/Project_AI_CV/docs/Definition_of_Done.md): Audited checklist ticked with numerical evidence, file links, and Git commit SHAs.
