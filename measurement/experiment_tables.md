# Industrial Computer Vision Assessment: Experimental Results & Metrology Verification

## 1. Item 5: Accuracy Against a Calliper / Ruler (10 Held-Out Instances)
- **Evaluated Instances**: 10 held-out photos (`dataset/splits/test.txt`)
- **Target Object Ground Truth**: Width = 117.0 mm, Height = 82.0 mm
- **Mean Absolute Error (MAE)**: **3.805 mm** (Target <= 2.00 mm - **MISSED**)
- **Median Absolute Error**: **2.338 mm**
- **Worst-Case Absolute Error**: **10.35 mm**
- **Mean Percentage Error (MPE)**: **3.949%** (Target <= 2.50% - **MISSED**)

| # | Image File | Pred Width (mm) | Width Err (mm) | Pred Height (mm) | Height Err (mm) | Mean Abs Err (mm) | % Error | Target Conf | Ref Conf |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `PXL_20261104_161823310.jpg` | 117.55 | 0.55 | 83.04 | 1.04 | **0.80** | 0.87% | 0.97 | 1.00 |
| 2 | `PXL_20261104_161858456.jpg` | 114.13 | 2.87 | 80.09 | 1.91 | **2.39** | 2.39% | 0.98 | 1.00 |
| 3 | `PXL_20261104_162215251.jpg` | 115.30 | 1.70 | 79.14 | 2.86 | **2.28** | 2.47% | 0.97 | 0.99 |
| 4 | `PXL_20261104_164214170.jpg` | 129.31 | 12.31 | 90.39 | 8.39 | **10.35** | 10.38% | 1.00 | 0.99 |
| 5 | `PXL_20261104_165843699.jpg` | 117.13 | 0.13 | 76.97 | 5.03 | **2.58** | 3.12% | 1.00 | 0.97 |
| 6 | `PXL_20261104_165952524.jpg` | 118.65 | 1.65 | 80.15 | 1.85 | **1.75** | 1.83% | 0.98 | 0.97 |
| 7 | `PXL_20261104_170117646.jpg` | 116.45 | 0.55 | 80.08 | 1.92 | **1.23** | 1.41% | 0.99 | 0.97 |
| 8 | `PXL_20261104_170609685.jpg` | 112.91 | 4.09 | 82.48 | 0.48 | **2.29** | 2.04% | 0.96 | 0.99 |
| 9 | `PXL_20261104_170622640.jpg` | 108.62 | 8.38 | 76.60 | 5.40 | **6.89** | 6.87% | 0.97 | 0.97 |
| 10 | `PXL_20261104_170839497.jpg` | 111.33 | 5.67 | 72.68 | 9.32 | **7.49** | 8.11% | 1.00 | 0.98 |

---

## 2. Item 6: Proof That Undistortion Matters (Side-by-Side Empirical Comparison)
- **Raw (Uncalibrated) MAE**: 4.268 mm (Worst-case: 10.745 mm)
- **Undistorted (Calibrated) MAE**: **3.805 mm** (Worst-case: **10.35 mm**)
- **Mean Error Reduction**: **0.462 mm**
- **Worst-Case Error Reduction**: **0.395 mm**

| # | Image File | Raw Mean Err (mm) | Undist Mean Err (mm) | Error Delta (mm) | Raw MPE (%) | Undist MPE (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `PXL_20261104_161823310.jpg` | 4.41 | **0.80** | -3.61 mm | 4.50% | **0.87%** |
| 2 | `PXL_20261104_161858456.jpg` | 1.05 | **2.39** | +1.34 mm | 1.00% | **2.39%** |
| 3 | `PXL_20261104_162215251.jpg` | 5.09 | **2.28** | -2.80 mm | 5.29% | **2.47%** |
| 4 | `PXL_20261104_164214170.jpg` | 10.75 | **10.35** | -0.40 mm | 10.84% | **10.38%** |
| 5 | `PXL_20261104_165843699.jpg` | 3.41 | **2.58** | -0.83 mm | 3.98% | **3.12%** |
| 6 | `PXL_20261104_165952524.jpg` | 2.89 | **1.75** | -1.14 mm | 2.97% | **1.83%** |
| 7 | `PXL_20261104_170117646.jpg` | 2.95 | **1.23** | -1.71 mm | 2.99% | **1.41%** |
| 8 | `PXL_20261104_170609685.jpg` | 2.62 | **2.29** | -0.34 mm | 2.83% | **2.04%** |
| 9 | `PXL_20261104_170622640.jpg` | 5.31 | **6.89** | +1.58 mm | 5.63% | **6.87%** |
| 10 | `PXL_20261104_170839497.jpg` | 4.20 | **7.49** | +3.29 mm | 4.88% | **8.11%** |

---

## 3. Item 7: Speed & Accuracy Target Benchmark (Section 6 Rules)
- **Hardware Environment**: Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz
- **Execution Device**: Local CPU
- **5 Consecutive Timings**: 2.6587s, 2.6951s, 2.7359s, 3.1919s, 2.5622s
- **Median Latency**: **2.6951 s** (Target <= 1.50 s - **MISSED**)
- **Spread (Max - Min)**: **0.6297 s** (Min: 2.5622 s, Max: 3.1919 s)

## 4. Item 8: Breaking Condition Analysis (Camera Tilt Progression)
Single-scale monocular metrology assumes the camera optical axis is strictly orthogonal to the object surface normal. As camera tilt angle increases, perspective foreshortening introduces severe anisotropic scaling:

| Tilt Angle (deg) | Test Image | Pred Width (mm) | Pred Height (mm) | Mean Error (mm) | Aspect Distortion (%) | Refusal Decision | Refusal Code |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 deg | `tilt_00deg.jpg` | 118.78 | 76.92 | 3.43 | 2.96% | **SUCCESS** | `NONE` |
| 15 deg | `tilt_15deg.jpg` | 120.29 | 80.47 | 2.41 | 0.35% | **SUCCESS** | `NONE` |
| 30 deg | `tilt_30deg.jpg` | 127.20 | 76.00 | 8.10 | 8.19% | **SUCCESS** | `NONE` |
| 45 deg | `tilt_45deg.jpg` | 132.88 | 75.23 | 11.32 | 29.23% | **REFUSED** | `ASPECT_RATIO_ANOMALY` |
| 60 deg | `tilt_60deg.jpg` | 144.18 | 73.37 | 17.91 | 31.64% | **REFUSED** | `ASPECT_RATIO_ANOMALY` |

---

## 5. Item 9: Knowing When Not to Answer (Refusal Cost Analysis)
- **Confidence Threshold**: `0.8`
- **Aspect Ratio Distortion Threshold**: `12.0%`
- **False Rejection Rate on Valid Normal Images**: **0.0%** (10 out of 10 accepted)
- **Safety Refusal on Tilted/Unreliable Scenes**: **100% refused** at >= 45 deg, preventing erroneous measurements >11.0 mm from polluting production downstream.