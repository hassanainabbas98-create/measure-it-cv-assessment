# Project Plan: "Measure It" (v2.0)

## 1. Project Overview & Scope
- **Project**: "Measure It" AI / Computer Vision Assessment (xis.ai)
- **Objective**: Develop an industrial-grade metric measurement pipeline that photographs a physical target object (Rectangular Charger Box: 117.0 mm x 82.0 mm) alongside a planar reference object (Standard CNIC ID Card: 86.0 mm x 54.0 mm), removes lens distortion via camera calibration, segments the object with a custom-trained PyTorch Mask R-CNN model, and converts pixel outlines to accurate millimetre dimensions with refusal guardrails.
- **Git Branch**: `dev-test01`
- **Model Architecture**: PyTorch `torchvision.models.detection.maskrcnn_resnet50_fpn` (strictly non-YOLO, non-Roboflow, trained with deterministic `seed = 42`).

---

## 2. Hardware & Environment Specifications
- **Operating System**: Microsoft Windows 10 Pro (Version 10.0.19045)
- **CPU**: Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz (2 Cores, 4 Logical Processors)
- **Execution Device**: Local CPU runtime
- **RAM**: 7.86 GB Visible Physical Memory
- **Camera**: Google Pixel 6 Pro (Main wide sensor, locked 1x optical zoom, verified 4032x2268 resolution, locked focus and exposure)
- **Calibration Target**: Printed checkerboard on rigid backing, 8x10 squares = 7x9 inner corners (`(7, 9)` in OpenCV), physical square size = 20.0 mm.

---

## 3. Execution Schedule & 8-Hour Time Breakdown

| Phase | Description & Deliverables | Planned Time | Actual Time | Status & Commit |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 1: Setup & Docs** | Git branch initialization (`dev-test01`), directory structure, `.gitignore`, initial `Plan.md`, `Objectives.md`, `Definition_of_Done.md`. | 30 mins | 20 mins | **Completed** (`f8f5083`) |
| **Phase 2: Calibration** | Collect >=20 checkerboard images, write `calibration/calibrate.py`, compute reprojection error, export camera matrix & distortion coefficients (`calibration/camera_params.npz`). | 45 mins | 35 mins | **Completed** (`2c68e00`) |
| **Phase 3: Dataset & Splits** | Acquire 75 object photos, undistort images, generate segmentation masks via `dataset/annotate_tool.py`, commit fixed 70/20/10 split lists (`dataset/splits/train.txt`, `val.txt`, `test.txt`). | 75 mins | 60 mins | **Completed** (`fddf4c1`, `0c93d97`) |
| **Phase 4: Segmentation Model** | Develop `models/train.py` (PyTorch Mask R-CNN ResNet-50 FPN, seed=42) and `models/evaluate.py`. Train model and evaluate test split (Mask IoU, mAP@0.5, mAP@0.5:0.95, Precision, Recall). | 90 mins | 50 mins | **Completed** (`d50e461`) |
| **Phase 5: Metric Inference** | Develop `inference/measure.py` CLI: image undistortion, Mask R-CNN inference, reference detection, scale factor calculation, millimetre output, refusal guardrails. | 60 mins | 45 mins | **Completed** (`f476000`) |
| **Phase 6: Measurement Experiments** | Execute validation suite: 10 instances vs ruler (Item 5), raw vs undistorted error proof (Item 6), latency benchmark (Item 7), camera tilt breaking test (Item 8: 0 deg to 60 deg). | 90 mins | 40 mins | **Completed** (`556bcb4`) |
| **Phase 7: Final Documentation** | Complete Decision Record (Item 10), populate actual numbers in `Objectives.md` and `Definition_of_Done.md`, complete `README.md` (with SHA-256 weight hash), verify zero dead code. | 60 mins | 35 mins | **Completed** (Commit #8) |
| **Buffer** | Verification, Loom demo dry run, pull request preparation. | 30 mins | 20 mins | **Completed** |
| **Total** | | **480 mins (8.0 hrs)** | **305 mins (5.1 hrs)** | **On Schedule** |

---

## 4. What Changed During Execution (Section 5.3)
As required by Section 5.3 ("A plan that matches what happened is good. A plan that changed, with a note saying why, is better"):
1. **Local CPU Training instead of Cloud Offload**: The initial plan provided a fallback to Google Colab T4 for model training if local CPU training proved too slow. By freezing the early ImageNet-pretrained ResNet-50 backbone layers (`trainable_backbone_layers = 0`) and training the RPN, box predictor, and FPN mask head at `512x288` internal resolution, full convergence was achieved locally on the dual-core Intel Core i5 CPU in only 8 epochs (~3.5 minutes total training time, reaching 0.9503 validation Mask IoU). This eliminated external cloud dependencies and ensured 100% self-contained local reproducibility.
2. **Annotation Ergonomics**: Instead of using external web services (CVAT/Roboflow), we engineered an in-house interactive OpenCV tool ([`dataset/annotate_tool.py`](file:///c:/Users/kk/Desktop/Project_AI_CV/dataset/annotate_tool.py)) that scaled display resolution to screen size while mapping mouse clicks back to native `4032x2268` coordinates. This allowed rapid, accurate ground-truth labeling of all 75 images and 5 tilt images in under 15 minutes.

---

## 5. Architectural Decision Record (Item 10)

### Decision 1: Instance Segmentation Architecture (Two-Stage Mask R-CNN vs. Semantic Segmentation vs. YOLO)
- **Approach Taken**: Torchvision `torchvision.models.detection.maskrcnn_resnet50_fpn` with Feature Pyramid Networks and RoIAlign, fine-tuned locally with pinned random seed (`seed = 42`).
- **Approach Rejected**: 
  1. *Ultralytics YOLOv8-seg / Roboflow*: Explicitly rejected to strictly comply with Section 3 (#3) and Section 9 (automatic rejection criteria).
  2. *Lightweight Semantic Segmentation (e.g., U-Net or MobileNetV3-DeepLabV3)*: Rejected because metric metrology requires separating distinct physical object instances with associated confidence probabilities, bounding box proposals, and orientation contours.
- **What Rejecting It Cost**:
  - *Inference Latency*: Rejecting lightweight single-stage semantic segmentation cost us approximately 1.19 seconds of CPU latency (`2.695s` median observed vs. our aggressive `1.50s` target).
  - *RoIAlign Mask Quantization*: Mask R-CNN generates binary masks via a $28 \times 28$ RoIAlign grid before bilinear upsampling, introducing subtle edge smoothing around sharp polygon corners when projected back to `4032x2268` native pixels.

### Decision 2: Metric Conversion Geometry (Dual-Axis Reference Scale vs. Full 4-Point Homography)
- **Approach Taken**: Isotropic dual-axis reference scaling (`scale_mm_per_px = 0.5 * ((86.0 / ref_w_px) + (54.0 / ref_h_px))`) extracted from the reference card's minimum-area bounding rectangle ([`cv2.minAreaRect`](file:///c:/Users/kk/Desktop/Project_AI_CV/inference/measure.py#L93)), coupled with an active `12%` aspect-ratio tilt refusal guardrail ([`inference/measure.py`](file:///c:/Users/kk/Desktop/Project_AI_CV/inference/measure.py#L225)).
- **Approach Rejected**: Full 4-point planar perspective homography (`cv2.findHomography` / `cv2.warpPerspective`).
- **What Rejecting It Cost** *(Loom Question K3)*:
  - *Tilt Sensitivity*: An isotropic single-scale factor assumes the optical axis is orthogonal to the scene normal. Rejecting full planar homography cost us measurement accuracy at moderate tilt angles: at $30^\circ$ tilt, measurement error increased to `8.10 mm` (an $8.19\%$ aspect ratio distortion) before our Item 9 refusal threshold triggered at $45^\circ$ (`11.32 mm` error). 
  - *Why this trade-off was chosen*: Planar homography requires identifying the exact 4 corresponding ordered corners of the reference card, which is vulnerable to corner occlusion or slight segmentation noise. Dual-axis `minAreaRect` is substantially more robust to mask noise in standard top-down industrial camera inspection setups, while the aspect-ratio guardrail safely refuses tilted images when the planar assumption is violated.

### Decision 3: Contour Dimension Fitting Geometry (Rotated MinAreaRect vs. Axis-Aligned Bounding Box)
- **Approach Taken**: Minimum-area rotated bounding rectangle ([`cv2.minAreaRect`](file:///c:/Users/kk/Desktop/Project_AI_CV/inference/measure.py#L93)) on the largest contour of the instance mask.
- **Approach Rejected**: Axis-aligned bounding box (`cv2.boundingRect`).
- **What Rejecting It Cost**:
  - *Contour Corner Sensitivity*: `minAreaRect` is sensitive to small contour protrusions or outlier mask pixels, which can alter the bounding box orientation angle by a few degrees. This was mitigated by taking the largest external contour and applying sub-pixel averaging.
  - *Why this was necessary*: Axis-aligned bounding boxes suffer massive geometric errors (up to $\sqrt{2} - 1 \approx 41.4\%$ inflation) whenever an object is rotated relative to the camera frame coordinates. `minAreaRect` guarantees rotation-invariant true Euclidean dimensions.
