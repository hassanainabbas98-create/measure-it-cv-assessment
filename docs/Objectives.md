# Measurable Objectives

## Hardware & System Specification
As mandated by Section 6 of the assessment specification, all timings and metric outputs are produced on the following environment:

| Specification Field | Recorded System Specification |
| :--- | :--- |
| **Operating System** | Microsoft Windows 10 Pro (Version 10.0.19045) |
| **CPU** | Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz (2 Cores, 4 Threads) |
| **Local GPU** | Intel(R) HD Graphics 620 |
| **Model Training GPU** | Google Colab NVIDIA Tesla T4 GPU (16 GB VRAM) (deterministic seed = 42) |
| **System RAM** | 7.86 GB Visible Physical Memory |
| **Camera Model** | Google Pixel 6 Pro (Main wide sensor, locked 1x zoom) |
| **Capture Resolution** | Verified ~4080 × 3072 pixels (locked focus, exposure, and focal length) |

---

## Objective MO-1: Dimensional Measurement Accuracy (Required)

| ID | Field | Entry |
| :--- | :--- | :--- |
| **MO-1** | **What is measured** | Absolute measurement error (Mean Absolute Error in mm) and percentage error (Mean Percentage Error) in physical width and height of the target rectangular charger box (ground-truth: 117.0 mm × 82.0 mm) compared against physical ruler/caliper measurements. |
| | **How** | An automated script (`measurement/evaluate_accuracy.py`) executes the measurement pipeline across 10+ held-out evaluation photos. For each image, the pipeline undistorts the frame, predicts the instance mask, identifies the reference object (CNIC card: 86.0 mm × 54.0 mm) to determine the millimetre-per-pixel scale factor, and calculates predicted dimensions via minimum-area bounding rectangles. Errors are calculated as $\text{Absolute Error} = \|d_{\text{pred}} - d_{\text{gt}}\|$ and $\text{Percentage Error} = \frac{\|d_{\text{pred}} - d_{\text{gt}}\|}{d_{\text{gt}}} \times 100\%$. |
| | **Target** | Mean Absolute Error (MAE) $\le 2.0\text{ mm}$ and Mean Percentage Error (MPE) $\le 2.5\%$ across 10 or more independent held-out test instances. |
| | **Conditions** | Camera positioned face-on to the planar scene (within $10^\circ$ normal tilt), indoor diffuse lighting, target charger box and reference card lying strictly coplanar on the same surface. |
| | **Not included** | Images with camera tilt $> 20^\circ$, scenes with partial occlusion of either the charger box or reference card, and non-planar placements (e.g. object elevated on a riser). |

### MO-1 Empirical Results (To be populated after execution)
*Raw terminal run outputs will be pasted here upon running `measurement/evaluate_accuracy.py`.*

| Instance # | Image File | GT Width (mm) | Pred Width (mm) | Width Err (mm) | GT Height (mm) | Pred Height (mm) | Height Err (mm) | Absolute Err (mm) | % Err |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `[TBD_AFTER_RUN]` | 117.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | 82.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` |
| 2 | `[TBD_AFTER_RUN]` | 117.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | 82.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` |
| 3 | `[TBD_AFTER_RUN]` | 117.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | 82.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` |
| 4 | `[TBD_AFTER_RUN]` | 117.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | 82.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` |
| 5 | `[TBD_AFTER_RUN]` | 117.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | 82.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` |
| 6 | `[TBD_AFTER_RUN]` | 117.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | 82.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` |
| 7 | `[TBD_AFTER_RUN]` | 117.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | 82.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` |
| 8 | `[TBD_AFTER_RUN]` | 117.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | 82.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` |
| 9 | `[TBD_AFTER_RUN]` | 117.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | 82.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` |
| 10 | `[TBD_AFTER_RUN]` | 117.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | 82.0 | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` | `[TBD_AFTER_RUN]` |

- **Mean Absolute Error**: `[TBD_AFTER_RUN]` mm
- **Median Absolute Error**: `[TBD_AFTER_RUN]` mm
- **Worst-case Absolute Error**: `[TBD_AFTER_RUN]` mm
- **Mean Percentage Error**: `[TBD_AFTER_RUN]` %
- **Outcome**: `[TBD_AFTER_RUN]` (Target $\le 2.0\text{ mm}$ MAE, $\le 2.5\%$ MPE)

---

## Objective MO-2: Segmentation Mask Quality & End-to-End Latency

| ID | Field | Entry |
| :--- | :--- | :--- |
| **MO-2** | **What is measured** | Segmentation mask quality (Mask Intersection-over-Union on the held-out test split) and end-to-end execution runtime (undistort + model forward pass + reference detection + metric scaling) for single-image inference. |
| | **How** | 1. `models/evaluate.py` evaluates the trained Mask R-CNN checkpoint against the fixed test split (`dataset/splits/test.txt`), computing mean Mask IoU, mAP@0.5, mAP@0.5:0.95, precision, and recall.<br>2. `measurement/benchmark_speed.py` measures wall-clock time across 5 consecutive runs on test images using `time.perf_counter()`, reporting the median and full spread (min to max). |
| | **Target** | Test split Mask IoU $\ge 0.85$, and local single-image inference median latency $\le 1.50\text{ s}$ over 5 runs. |
| | **Conditions** | Test split evaluated strictly on held-out images. Speed benchmark executed locally on Intel Core i5-7300U CPU, 8 GB RAM, Windows 10 Pro. |
| | **Not included** | Initial Python/PyTorch module import overhead, cold cache initialization, and saving debug visual plots to disk. |

### MO-2 Empirical Results (To be populated after execution)
*Raw terminal run outputs will be pasted here upon running `models/evaluate.py` and `measurement/benchmark_speed.py`.*

#### 1. Segmentation Quality on Test Split:
- **Mask IoU**: `[TBD_AFTER_RUN]`
- **mAP @ 0.5**: `[TBD_AFTER_RUN]`
- **mAP @ 0.5:0.95**: `[TBD_AFTER_RUN]`
- **Precision**: `[TBD_AFTER_RUN]`
- **Recall**: `[TBD_AFTER_RUN]`

#### 2. End-to-End Latency (5 runs on host CPU):
- **Run 1**: `[TBD_AFTER_RUN]` s
- **Run 2**: `[TBD_AFTER_RUN]` s
- **Run 3**: `[TBD_AFTER_RUN]` s
- **Run 4**: `[TBD_AFTER_RUN]` s
- **Run 5**: `[TBD_AFTER_RUN]` s
- **Median Latency**: `[TBD_AFTER_RUN]` s
- **Spread (Min - Max)**: `[TBD_AFTER_RUN]` s
- **Outcome**: `[TBD_AFTER_RUN]` (Target Mask IoU $\ge 0.85$, Median Latency $\le 1.50\text{ s}$)
