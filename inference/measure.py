"""
End-to-End Metric Measurement CLI for "Measure It" (xis.ai v2.0)
Items 4 & 9: Undistort -> Segment -> Detect Reference -> Scale to Millimetres with Refusal Guardrails.

LOOM EXPLANATION REFERENCES:
- Line turning pixels into millimetres (Loom Question K1): See function `convert_pixels_to_mm()`
- Line applying lens undistortion (Loom Question K2): See function `undistort_image()`
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import numpy as np
import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Ground-truth reference and object dimensions in millimetres
REF_WIDTH_MM = 86.0    # CNIC ID Card width (long side)
REF_HEIGHT_MM = 54.0   # CNIC ID Card height (short side)
REF_ASPECT_RATIO = REF_WIDTH_MM / REF_HEIGHT_MM  # 1.59259

TARGET_WIDTH_MM = 117.0  # Charger Box width (long side)
TARGET_HEIGHT_MM = 82.0  # Charger Box height (short side)

CONFIDENCE_THRESHOLD = 0.80
BORDER_MARGIN_PX = 15
MAX_ASPECT_ERROR = 0.12  # 12% max aspect ratio distortion before refusal


# ==============================================================================
# LOOM QUESTION K2: UNDISTORTION FUNCTION
# Where in code is undistortion applied? Exactly here at cv2.undistort().
# ==============================================================================
def undistort_image(image, params_path="calibration/camera_params.npz"):
    """
    Applies calibrated camera intrinsic matrix K and distortion coefficients D.
    Loom Question K2 reference: Undistortion is applied right at the input boundary
    before any neural inference or geometric measurement occurs.
    """
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Calibration parameters not found at {params_path}")
    
    calib_data = np.load(params_path)
    camera_matrix = calib_data["camera_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]
    
    # LINE BELOW APPLIES LENS UNDISTORTION (LOOM QUESTION K2):
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    return undistorted


def load_model(weights_path="models/weights/best_model.pth", config_path="models/config.json", device="cpu"):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    hp = config["hyperparameters"]
    min_size = hp.get("min_size", 288)
    max_size = hp.get("max_size", 512)

    model = maskrcnn_resnet50_fpn(weights=None, num_classes=config["num_classes"], min_size=min_size, max_size=max_size)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, config["num_classes"])
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced, config["num_classes"])

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def extract_oriented_dimensions(binary_mask):
    """
    Extracts oriented Euclidean width and height from binary mask using cv2.minAreaRect.
    Avoids axis-aligned box inflation and handles arbitrary planar object rotation.
    Returns: long_side_px, short_side_px, box_points_int
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, None

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    (cx, cy), (w, h), angle = rect

    long_side = max(w, h)
    short_side = min(w, h)

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    return float(long_side), float(short_side), box


def touches_image_boundary(binary_mask, margin=BORDER_MARGIN_PX):
    """Checks if segmented mask reaches within margin pixels of image borders."""
    h, w = binary_mask.shape[:2]
    # Check top, bottom, left, right border bands
    if np.any(binary_mask[:margin, :]) or np.any(binary_mask[-margin:, :]):
        return True
    if np.any(binary_mask[:, :margin]) or np.any(binary_mask[:, -margin:]):
        return True
    return False


# ==============================================================================
# LOOM QUESTION K1: PIXEL-TO-MILLIMETRE CONVERSION
# Which file and line turns pixels into millimetres? What must be true about the scene?
# Scene requirements:
#   1. Strict planarity: Target and reference must lie coplanar on the same surface.
#   2. Perpendicular optical axis: Camera must view the scene face-on (minimal tilt).
#   3. Uniform scale: Single-scale assumption holds only when distance to camera is uniform.
# ==============================================================================
def convert_pixels_to_mm(obj_width_px, obj_height_px, ref_width_px, ref_height_px):
    """
    Calculates metric scale factor (mm/px) from reference object and scales target dimensions.
    """
    # Dual-axis scale estimation averaged to reduce directional discretization noise:
    scale_w = REF_WIDTH_MM / max(1e-6, ref_width_px)
    scale_h = REF_HEIGHT_MM / max(1e-6, ref_height_px)
    scale_mm_per_px = 0.5 * (scale_w + scale_h)

    # --------------------------------------------------------------------------
    # LINES BELOW CONVERT PIXELS TO MILLIMETRES (LOOM QUESTION K1):
    width_mm = obj_width_px * scale_mm_per_px
    height_mm = obj_height_px * scale_mm_per_px
    # --------------------------------------------------------------------------

    return width_mm, height_mm, scale_mm_per_px


def measure_image(
    image_path,
    output_path=None,
    skip_undistort=False,
    confidence_thresh=CONFIDENCE_THRESHOLD,
    max_aspect_error=MAX_ASPECT_ERROR,
    model=None,
    config=None
):
    if not os.path.exists(image_path):
        return {
            "status": "ERROR",
            "message": f"Image file not found: {image_path}"
        }

    raw_img = cv2.imread(image_path)
    if raw_img is None:
        return {
            "status": "ERROR",
            "message": f"Could not decode image at {image_path}"
        }

    orig_h, orig_w = raw_img.shape[:2]

    # 1. Undistortion stage
    if skip_undistort:
        processed_img = raw_img.copy()
        undistort_applied = False
    else:
        processed_img = undistort_image(raw_img)
        undistort_applied = True

    # 2. Model inference
    if model is None or config is None:
        model, config = load_model()

    tgt_w = config["hyperparameters"].get("input_width", 512)
    tgt_h = config["hyperparameters"].get("input_height", 288)

    img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    resized_input = cv2.resize(img_rgb, (tgt_w, tgt_h))
    tensor_input = torch.as_tensor(resized_input.transpose(2, 0, 1), dtype=torch.float32) / 255.0

    with torch.no_grad():
        predictions = model([tensor_input])[0]

    pred_scores = predictions["scores"].cpu().numpy()
    pred_labels = predictions["labels"].cpu().numpy()
    pred_masks = predictions["masks"].cpu().squeeze(1).numpy()

    # Find top prediction for Target Object (Class 1: charger_box)
    target_idx = np.where(pred_labels == 1)[0]
    best_target_score = float(pred_scores[target_idx[0]]) if len(target_idx) > 0 else 0.0

    # Find top prediction for Reference Object (Class 2: cnic_card)
    ref_idx = np.where(pred_labels == 2)[0]
    best_ref_score = float(pred_scores[ref_idx[0]]) if len(ref_idx) > 0 else 0.0

    # --------------------------------------------------------------------------
    # ITEM 4 & ITEM 9 REFUSAL GUARDRAILS
    # --------------------------------------------------------------------------

    # Guardrail 1: Target object missing or below confidence threshold
    if len(target_idx) == 0 or best_target_score < confidence_thresh:
        return {
            "status": "REFUSED",
            "refusal_code": "TARGET_NOT_FOUND",
            "message": f"REFUSED: Target object (charger_box) not found or confidence ({best_target_score:.3f}) below threshold ({confidence_thresh})",
            "target_confidence": best_target_score,
            "reference_confidence": best_ref_score,
            "undistort_applied": undistort_applied
        }

    # Guardrail 2: Reference object missing or below confidence threshold
    if len(ref_idx) == 0 or best_ref_score < confidence_thresh:
        return {
            "status": "REFUSED",
            "refusal_code": "REFERENCE_NOT_FOUND",
            "message": f"REFUSED: Reference object (cnic_card) not found or confidence ({best_ref_score:.3f}) below threshold ({confidence_thresh})",
            "target_confidence": best_target_score,
            "reference_confidence": best_ref_score,
            "undistort_applied": undistort_applied
        }

    # Resize masks back to full resolution
    target_mask_small = pred_masks[target_idx[0]] > 0.5
    ref_mask_small = pred_masks[ref_idx[0]] > 0.5

    target_mask_full = cv2.resize(target_mask_small.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    ref_mask_full = cv2.resize(ref_mask_small.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Guardrail 3: Boundary clipping / partial occlusion
    if touches_image_boundary(target_mask_full) or touches_image_boundary(ref_mask_full):
        return {
            "status": "REFUSED",
            "refusal_code": "BOUNDARY_CLIPPING",
            "message": "REFUSED: Object or reference partially occluded/clipped at frame boundary",
            "target_confidence": best_target_score,
            "reference_confidence": best_ref_score,
            "undistort_applied": undistort_applied
        }

    # Extract pixel dimensions
    obj_w_px, obj_h_px, obj_box = extract_oriented_dimensions(target_mask_full)
    ref_w_px, ref_h_px, ref_box = extract_oriented_dimensions(ref_mask_full)

    if ref_w_px <= 0 or ref_h_px <= 0 or obj_w_px <= 0 or obj_h_px <= 0:
        return {
            "status": "REFUSED",
            "refusal_code": "DEGENERATE_MASK",
            "message": "REFUSED: Degenerate contour extracted from instance mask",
            "target_confidence": best_target_score,
            "reference_confidence": best_ref_score
        }

    # Guardrail 4: Aspect ratio distortion (Perspective tilt check)
    observed_ref_aspect = ref_w_px / max(1e-6, ref_h_px)
    aspect_rel_error = abs(observed_ref_aspect - REF_ASPECT_RATIO) / REF_ASPECT_RATIO

    if aspect_rel_error > max_aspect_error:
        return {
            "status": "REFUSED",
            "refusal_code": "ASPECT_RATIO_ANOMALY",
            "message": f"REFUSED: Reference card aspect ratio error ({aspect_rel_error*100:.1f}%) exceeds tolerance ({max_aspect_error*100:.1f}%). Severe perspective tilt or occlusion detected.",
            "observed_aspect": round(observed_ref_aspect, 4),
            "expected_aspect": round(REF_ASPECT_RATIO, 4),
            "aspect_error_pct": round(aspect_rel_error * 100, 2),
            "target_confidence": best_target_score,
            "reference_confidence": best_ref_score,
            "undistort_applied": undistort_applied
        }

    # 4. Pixel to millimetre metric calculation
    width_mm, height_mm, scale_mm_per_px = convert_pixels_to_mm(obj_w_px, obj_h_px, ref_w_px, ref_h_px)

    result = {
        "status": "SUCCESS",
        "image": os.path.basename(image_path),
        "undistort_applied": undistort_applied,
        "measurements": {
            "target_width_mm": round(width_mm, 2),
            "target_height_mm": round(height_mm, 2),
            "ground_truth_width_mm": TARGET_WIDTH_MM,
            "ground_truth_height_mm": TARGET_HEIGHT_MM,
            "error_width_mm": round(abs(width_mm - TARGET_WIDTH_MM), 2),
            "error_height_mm": round(abs(height_mm - TARGET_HEIGHT_MM), 2),
            "scale_mm_per_px": round(scale_mm_per_px, 5)
        },
        "confidences": {
            "target_confidence": round(best_target_score, 4),
            "reference_confidence": round(best_ref_score, 4)
        },
        "pixel_geometry": {
            "target_width_px": round(obj_w_px, 1),
            "target_height_px": round(obj_h_px, 1),
            "reference_width_px": round(ref_w_px, 1),
            "reference_height_px": round(ref_h_px, 1)
        }
    }

    # 5. Visual overlay generation
    overlay = processed_img.copy()

    # Draw semi-transparent masks
    # Green overlay for charger_box
    green_layer = np.zeros_like(overlay)
    green_layer[target_mask_full > 0] = (0, 220, 0)
    # Cyan overlay for cnic_card
    cyan_layer = np.zeros_like(overlay)
    cyan_layer[ref_mask_full > 0] = (255, 200, 0)

    mask_combined = (green_layer > 0) | (cyan_layer > 0)
    overlay[mask_combined] = cv2.addWeighted(overlay, 0.65, green_layer + cyan_layer, 0.35, 0)[mask_combined]

    # Draw oriented bounding boxes
    if obj_box is not None:
        cv2.drawContours(overlay, [obj_box], 0, (0, 255, 0), 4)
    if ref_box is not None:
        cv2.drawContours(overlay, [ref_box], 0, (255, 200, 0), 4)

    # Draw HUD Banner on top
    banner_h = 100
    cv2.rectangle(overlay, (0, 0), (orig_w, banner_h), (25, 25, 25), -1)

    hud_line1 = f"TARGET: Width = {width_mm:.1f} mm | Height = {height_mm:.1f} mm  (Conf: {best_target_score:.2f})"
    hud_line2 = f"REFERENCE: CNIC Card (Conf: {best_ref_score:.2f}) | Scale: {scale_mm_per_px:.4f} mm/px | Undistort: {'ON' if undistort_applied else 'OFF'}"

    cv2.putText(overlay, hud_line1, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, hud_line2, (30, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 100), 2, cv2.LINE_AA)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, overlay)
        result["overlay_path"] = output_path

    return result


def main():
    parser = argparse.ArgumentParser(description="Measure It - Monocular Metric Measurement CLI")
    parser.add_argument("--image", required=True, help="Path to input photo")
    parser.add_argument("--output", default=None, help="Path to save annotated visual overlay image")
    parser.add_argument("--skip-undistort", action="store_true", help="Skip camera undistortion (for Item 6 comparison)")
    parser.add_argument("--confidence-thresh", type=float, default=CONFIDENCE_THRESHOLD, help="Refusal confidence threshold")
    parser.add_argument("--max-aspect-error", type=float, default=MAX_ASPECT_ERROR, help="Max reference aspect ratio distortion")
    args = parser.parse_args()

    result = measure_image(
        image_path=args.image,
        output_path=args.output,
        skip_undistort=args.skip_undistort,
        confidence_thresh=args.confidence_thresh,
        max_aspect_error=args.max_aspect_error
    )

    print("\n" + "=" * 65)
    print("MEASURE IT - INDUSTRIAL METRIC MEASUREMENT RESULT")
    print("=" * 65)
    print(json.dumps(result, indent=2))
    print("=" * 65)


if __name__ == "__main__":
    main()
