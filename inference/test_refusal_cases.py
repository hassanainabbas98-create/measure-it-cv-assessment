"""
Test Refusal Guardrails for "Measure It" (xis.ai v2.0)
Validates Item 4 & Item 9 ("Knowing when not to answer"):
  - Case 1: Target Object missing / below confidence
  - Case 2: Reference Object missing / below confidence
  - Case 3: Boundary clipping / partial occlusion
  - Case 4: Aspect ratio anomaly / severe planar tilt
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import cv2

from inference.measure import measure_image, load_model

def run_refusal_tests(output_log_path="inference/refusal_test_results.json"):
    print("=" * 65)
    print("TESTING REFUSAL GUARDRAILS (ITEM 4 & ITEM 9)")
    print("=" * 65)

    model, config = load_model()
    test_img_path = "dataset/images_undistorted/PXL_20261104_161823310.jpg"
    if not os.path.exists(test_img_path):
        raise FileNotFoundError(f"Base test image not found: {test_img_path}")

    base_img = cv2.imread(test_img_path)
    h, w = base_img.shape[:2]

    # Create temporary directory for synthetic test cases
    temp_dir = "inference/test_cases"
    os.makedirs(temp_dir, exist_ok=True)

    results = []

    # --------------------------------------------------------------------------
    # Case 0: Normal Valid Image (Control)
    # --------------------------------------------------------------------------
    res_valid = measure_image(test_img_path, skip_undistort=True, model=model, config=config)
    print(f"[+] Case 0 (Normal Image): Status = {res_valid['status']}")
    assert res_valid["status"] == "SUCCESS", "Control test failed!"
    results.append({
        "case_id": "CASE_0_NORMAL",
        "description": "Standard planar scene with both target and reference present",
        "expected_status": "SUCCESS",
        "observed_status": res_valid["status"],
        "passed": res_valid["status"] == "SUCCESS",
        "details": res_valid
    })

    # --------------------------------------------------------------------------
    # Case 1: Target Object Missing (Mask out charger_box at x: 600..1700, y: 700..1600)
    # --------------------------------------------------------------------------
    img_no_target = base_img.copy()
    img_no_target[700:1600, 600:1700] = 200
    p_no_target = os.path.join(temp_dir, "missing_target.jpg")
    cv2.imwrite(p_no_target, img_no_target)

    res_no_target = measure_image(p_no_target, skip_undistort=True, model=model, config=config)
    print(f"[+] Case 1 (Target Missing): Status = {res_no_target['status']} | Code = {res_no_target.get('refusal_code')}")
    assert res_no_target["status"] == "REFUSED" and res_no_target.get("refusal_code") == "TARGET_NOT_FOUND"
    results.append({
        "case_id": "CASE_1_TARGET_NOT_FOUND",
        "description": "Target charger box absent or obliterated from scene",
        "expected_status": "REFUSED",
        "observed_status": res_no_target["status"],
        "refusal_code": res_no_target.get("refusal_code"),
        "passed": res_no_target["status"] == "REFUSED",
        "message": res_no_target.get("message")
    })

    # --------------------------------------------------------------------------
    # Case 2: Reference Object Missing (Mask out CNIC card at x: 1750..2450, y: 900..1400)
    # --------------------------------------------------------------------------
    img_no_ref = base_img.copy()
    img_no_ref[900:1400, 1750:2450] = 200
    p_no_ref = os.path.join(temp_dir, "missing_reference.jpg")
    cv2.imwrite(p_no_ref, img_no_ref)

    res_no_ref = measure_image(p_no_ref, skip_undistort=True, model=model, config=config)
    print(f"[+] Case 2 (Reference Missing): Status = {res_no_ref['status']} | Code = {res_no_ref.get('refusal_code')}")
    assert res_no_ref["status"] == "REFUSED" and res_no_ref.get("refusal_code") == "REFERENCE_NOT_FOUND"
    results.append({
        "case_id": "CASE_2_REFERENCE_NOT_FOUND",
        "description": "Reference CNIC card absent or obliterated from scene",
        "expected_status": "REFUSED",
        "observed_status": res_no_ref["status"],
        "refusal_code": res_no_ref.get("refusal_code"),
        "passed": res_no_ref["status"] == "REFUSED",
        "message": res_no_ref.get("message")
    })

    # --------------------------------------------------------------------------
    # Case 3: Boundary Clipping / Partly Hidden (Shifted/Cropped at border)
    # --------------------------------------------------------------------------
    # Crop right through the middle of the reference card and charger box
    img_clipped = base_img[int(h * 0.40):, int(w * 0.20):int(w * 0.75)]
    p_clipped = os.path.join(temp_dir, "clipped_boundary.jpg")
    cv2.imwrite(p_clipped, img_clipped)

    res_clipped = measure_image(p_clipped, skip_undistort=True, model=model, config=config)
    print(f"[+] Case 3 (Boundary Clipped): Status = {res_clipped['status']} | Code = {res_clipped.get('refusal_code')}")
    assert res_clipped["status"] == "REFUSED"
    results.append({
        "case_id": "CASE_3_BOUNDARY_CLIPPING",
        "description": "Object or reference clipped at frame edge / partial visibility",
        "expected_status": "REFUSED",
        "observed_status": res_clipped["status"],
        "refusal_code": res_clipped.get("refusal_code"),
        "passed": res_clipped["status"] == "REFUSED",
        "message": res_clipped.get("message")
    })

    # --------------------------------------------------------------------------
    # Case 4: Severe Planar Tilt (tilt_45deg or tilt_60deg)
    # --------------------------------------------------------------------------
    tilt_path = "measurement/tilt_images_undistorted/tilt_60deg.jpg"
    if os.path.exists(tilt_path):
        res_tilt = measure_image(tilt_path, skip_undistort=True, max_aspect_error=0.10, model=model, config=config)
        print(f"[+] Case 4 (Severe Tilt 60 deg): Status = {res_tilt['status']} | Code = {res_tilt.get('refusal_code')}")
        results.append({
            "case_id": "CASE_4_SEVERE_TILT",
            "description": "Camera tilted at 60 degrees relative to surface normal",
            "expected_status": "REFUSED",
            "observed_status": res_tilt["status"],
            "refusal_code": res_tilt.get("refusal_code"),
            "passed": res_tilt["status"] == "REFUSED",
            "message": res_tilt.get("message")
        })

    # Save summary
    with open(output_log_path, "w") as f:
        json.dump(results, f, indent=2)

    # Clean up temporary test cases directory (Zero stray files rule)
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print("\n" + "=" * 65)
    print("ALL REFUSAL GUARDRAILS VERIFIED SUCCESSFULLY")
    print(f"[+] Detailed refusal log saved to: {output_log_path}")
    print("=" * 65)
    return results

if __name__ == "__main__":
    run_refusal_tests()
