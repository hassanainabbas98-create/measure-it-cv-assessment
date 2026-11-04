# Definition of Done

This document defines the audited completion criteria for the "Measure It" technical assessment (xis.ai v2.0). 
Every checklist item must be verified with concrete evidence (e.g., commit SHA, raw measurement number, or reproduction log). In accordance with Section 5.4, a box ticked without supporting evidence is considered unticked.

---

## 1. Core Engineering Checklist

- [ ] **1. Camera Calibration (Item 1)**
  - *Requirement*: ≥20 checkerboard images captured and calibrated with OpenCV. Camera matrix and distortion coefficients saved to a persistent JSON/YAML file. Reprojection error calculated and reported.
  - *Evidence*: `[TBD_AFTER_RUN - Calibration image count: N, Reprojection Error: X.XX px, Saved file: calibration/camera_params.json]`

- [ ] **2. Dataset Collection & Fixed Split List (Item 2)**
  - *Requirement*: ≥70 images of target object collected with the calibrated camera, undistorted, and labelled with segmentation masks. Fixed 70/20/10 split committed prior to any model training code.
  - *Evidence*: `[TBD_AFTER_RUN - Total images: N, Train count: N, Val count: N, Test count: N, Split Commit SHA: <hash>]`

- [ ] **3. Segmentation Model Training & Test Evaluation (Item 3)**
  - *Requirement*: Custom-trained segmentation model (Mask R-CNN ResNet-50 FPN, non-YOLO, non-Roboflow, seed=42) evaluated strictly on the committed test split. Reports Mask IoU, mAP@0.5, mAP@0.5:0.95, Precision, and Recall.
  - *Evidence*: `[TBD_AFTER_RUN - Model architecture: Mask R-CNN ResNet-50 FPN, Test Mask IoU: X.XX, mAP@0.5: X.XX, mAP@0.5:0.95: X.XX, Precision: X.XX, Recall: X.XX, Checkpoint SHA-256: <hash>]`

- [ ] **4. End-to-End Single-Command Metric Inference CLI (Item 4)**
  - *Requirement*: Single CLI command (`python -m inference.measure --image <path>`) that undistorts input, detects target object mask, finds reference object, computes scale factor, and outputs overlaid image with width (mm), height (mm), and confidence score. Cleanly handles two failure modes: (a) object not found, (b) reference not found.
  - *Evidence*: `[TBD_AFTER_RUN - CLI command reproduction log, sample overlay path, exit codes for missing target/reference]`

- [ ] **5. Physical Accuracy Validation vs. Caliper/Ruler (Item 5)**
  - *Requirement*: 10+ held-out instances measured with physical ruler/caliper vs. system. Detailed per-instance table reporting Mean Absolute Error (MAE in mm) and Mean Percentage Error (MPE).
  - *Evidence*: `[TBD_AFTER_RUN - Evaluated instances: N, MAE: X.XX mm, MPE: X.XX%, Raw output file: measurement/results_caliper.json]`

- [ ] **6. Empirical Proof of Undistortion Value (Item 6)**
  - *Requirement*: Exact same 10+ instances evaluated on raw (distorted) images vs. undistorted images. Side-by-side comparative error table with quantitative explanation of distortion impact.
  - *Evidence*: `[TBD_AFTER_RUN - Raw MAE: X.XX mm vs. Undistorted MAE: X.XX mm, Delta: X.XX mm, Explanation in docs/Objectives.md]`

- [ ] **7. Measurable Speed / Accuracy Objectives (Item 7 & Section 6)**
  - *Requirement*: Specific targets set in `docs/Objectives.md` (MO-1: MAE $\le 2.0\text{ mm}$, MPE $\le 2.5\%$; MO-2: Test Mask IoU $\ge 0.85$, median inference time $\le 1.50\text{ s}$ over 5 runs). Measured and reported with median and spread.
  - *Evidence*: `[TBD_AFTER_RUN - MO-1 status: MET/MISSED with numbers; MO-2 status: MET/MISSED with median X.XX s, spread X.XX s]`

- [ ] **8. Breaking Condition Analysis: Camera Tilt (Item 8)**
  - *Requirement*: Quantitative evaluation of measurement error as camera tilt angle varies (0°, 15°, 30°, 45° relative to normal). Justification provided for failure dynamics.
  - *Evidence*: `[TBD_AFTER_RUN - Tilt error curve table: 0° (X.XX mm), 15° (X.XX mm), 30° (X.XX mm), 45° (X.XX mm)]`

- [ ] **9. Refusal Guardrails & Trust Thresholds (Item 9)**
  - *Requirement*: System refuses to report measurements when confidence is below trust threshold or when plane tilt / aspect ratio distortion is detected. Threshold trade-off quantified (rejection rate vs false measurement avoidance).
  - *Evidence*: `[TBD_AFTER_RUN - Confidence threshold: X.XX, Rejection trigger metrics: low mask score, missing reference, aspect-ratio anomaly]`

- [ ] **10. Architectural Decision Record (Item 10)**
  - *Requirement*: Documented inside `docs/Plan.md`. Records approaches taken, approaches rejected (e.g. YOLO/Roboflow, axis-aligned boxes), and what rejecting them cost.
  - *Evidence*: `[TBD_AFTER_RUN - Reference to docs/Plan.md Section 5 with full trade-off analysis]`

---

## 2. Assessment Integrity & Code Hygiene

- [ ] **Strict Commit Chronology Maintained**:
  - [ ] Commit #1: Initial plan, objectives, definition of done, directory skeleton (prior to any code).
  - [ ] Split list committed prior to model training code.
  - [ ] Distinct commits for calibration, dataset/splits, model training, CLI inference, measurement suite, and final documentation.
- [ ] **Zero Synthetic / Fabricated Data**: All numbers originate directly from script outputs on real photos.
- [ ] **Code Hygiene**: Zero dead code, zero commented-out code blocks, zero stray files, no messy notebooks in the repository root.
- [ ] **Incomplete Items Declared**: Any unfinished tasks explicitly disclosed with rationale.
  - *Unfinished items log*: `[TBD_AFTER_RUN - None or explicitly listed]`
