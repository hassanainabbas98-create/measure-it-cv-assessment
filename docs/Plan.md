# Project Plan: "Measure It" (v2.0)

## 1. Project Overview & Scope
- **Project**: "Measure It" AI / Computer Vision Assessment (xis.ai)
- **Objective**: Develop an industrial-grade metric measurement pipeline that photographs a physical target object (Rectangular Charger Box: 117.0 mm × 82.0 mm) alongside a planar reference object (Standard CNIC ID Card: 86.0 mm × 54.0 mm), removes lens distortion via camera calibration, segments the object with a custom-trained PyTorch Mask R-CNN model, and converts pixel outlines to accurate millimetre dimensions with refusal guardrails.
- **Git Branch**: `dev-test01`
- **Model Architecture**: PyTorch `torchvision.models.detection.maskrcnn_resnet50_fpn` (strictly non-YOLO, non-Roboflow, trained with deterministic `seed = 42`).

---

## 2. Hardware & Environment Specifications
- **Operating System**: Microsoft Windows 10 Pro (Build 10.0.19045)
- **CPU**: Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz (2 Cores, 4 Logical Processors)
- **Local GPU**: Intel(R) HD Graphics 620 (Local execution: calibration, inference, evaluation)
- **Cloud Training Environment**: Google Colab NVIDIA Tesla T4 GPU (16 GB VRAM) for model fine-tuning with deterministic seed `42`
- **RAM**: 7.86 GB (8 GB physical)
- **Camera**: Google Pixel 6 Pro (Main wide sensor, locked at 1x optical zoom, resolution verified ~4080×3072, locked focus/exposure)
- **Calibration Target**: Printed checkerboard on rigid backing, 8×10 squares = 7×9 inner corners (`(7, 9)` in OpenCV), square size = 20.0 mm.

---

## 3. Execution Schedule & 8-Hour Time Breakdown

| Phase | Description & Deliverables | Planned Time | Target Commit |
| :--- | :--- | :--- | :--- |
| **Phase 1: Setup & Docs** | Git branch initialization (`dev-test01`), directory structure, `.gitignore`, initial `Plan.md`, `Objectives.md`, `Definition_of_Done.md`. | 30 mins | Commit #1 |
| **Phase 2: Calibration** | Collect ≥20 checkerboard images, write `calibration/calibrate.py`, compute reprojection error, export camera matrix & distortion coefficients (`calibration/camera_params.json`). | 45 mins | Commit #2 |
| **Phase 3: Dataset & Splits** | Acquire ≥70 object photos, undistort images, generate segmentation masks, create and commit fixed 70/20/10 split files (`dataset/splits/train.txt`, `val.txt`, `test.txt`). | 75 mins | Commit #3 |
| **Phase 4: Segmentation Model** | Develop `models/train.py` (PyTorch Mask R-CNN ResNet-50 FPN, seed=42) and `models/evaluate.py`. Train model and evaluate test split (Mask IoU, mAP@0.5, mAP@0.5:0.95, Precision, Recall). | 90 mins *(train wait excluded)* | Commit #4 |
| **Phase 5: Metric Inference** | Develop `inference/measure.py` CLI: image undistortion, Mask R-CNN inference, reference detection, scale factor calculation, millimetre output, refusal guardrails (missing object/reference, low confidence). | 60 mins | Commit #5 |
| **Phase 6: Measurement Experiments** | Execute validation suite: 10+ instances vs ruler/caliper (Item 5), raw vs undistorted error proof (Item 6), latency benchmark (Item 7), camera tilt breaking test (Item 8: 0°, 15°, 30°, 45°). | 90 mins | Commit #6 |
| **Phase 7: Final Documentation** | Complete Decision Record (Item 10), populate actual numbers in `Objectives.md` and `Definition_of_Done.md`, complete `README.md` (with SHA-256 weight hash), verify zero dead code. | 60 mins | Commit #7 |
| **Buffer** | Verification, Loom demo dry run, pull request preparation. | 30 mins | - |
| **Total** | | **480 mins (8.0 hrs)** | |

---

## 4. Prioritization & Fallback Strategy
- **Critical Floor (Items 1–4)**: Must be fully operational before downstream tasks begin.
  1. OpenCV camera calibration with saved parameters & reprojection error.
  2. 70+ custom images with committed 70/20/10 split lists prior to training.
  3. Non-YOLO/Roboflow trained segmentation model with documented test-split metrics.
  4. End-to-end `inference/measure.py` CLI with refusal logic for missing/ambiguous targets.
- **Secondary Tier (Items 5–7)**: Caliper ground-truth evaluation, raw vs. undistorted empirical proof, speed/accuracy benchmark.
- **Tertiary Tier (Items 8–9)**: Camera tilt breaking condition analysis and refusal threshold sensitivity.
- **Explicitly Skipped / Out of Scope**: Multi-view stereo reconstruction, dynamic SLAM, heavy web UI serving. Focus remains solely on monocular metric accuracy under controlled planar reference constraints.

---

## 5. Decision Record (Item 10 Placeholder)

> [!NOTE]
> This section will be finalized in Phase 7 upon completion of real empirical runs.

- **Decision 1: Segmentation Architecture**
  - *Approach Taken*: PyTorch torchvision Mask R-CNN (`maskrcnn_resnet50_fpn`) with pretrained backbone fine-tuned locally/colab with pinned `seed = 42`.
  - *Approach Rejected*: Ultralytics YOLOv8-seg / Roboflow hosted models.
  - *What Rejecting It Cost*: [TBD_AFTER_RUN - e.g., Higher training compute requirements and longer inference latency compared to lightweight single-stage detectors, offset by compliance with assessment rules and granular control over anchor proposals and loss weighting].
- **Decision 2: Reference Marker Selection**
  - *Approach Taken*: Planar Standard CNIC ID Card (86.0 mm × 54.0 mm) with taped sensitive fields placed in the object plane.
  - *Approach Rejected*: ArUco fiducial marker / circular coin.
  - *What Rejecting It Cost*: [TBD_AFTER_RUN - e.g., Requires explicit contour/aspect-ratio detection or dual-class segmentation rather than off-the-shelf cv2.aruco dictionary lookups, but represents realistic industrial packaging measurement where fiducials are unavailable].
- **Decision 3: Dimension Extraction Geometry**
  - *Approach Taken*: Minimum-area rotated bounding rectangle (`cv2.minAreaRect`) applied on the segmented binary mask contour to measure true oriented length and width irrespective of object orientation.
  - *Approach Rejected*: Axis-aligned bounding box (`cv2.boundingRect`).
  - *What Rejecting It Cost*: [TBD_AFTER_RUN - e.g., Sensitivity to small contour corner segmentation noise, which is mitigated via morphological smoothing, but eliminates severe axis-alignment orientation errors].
