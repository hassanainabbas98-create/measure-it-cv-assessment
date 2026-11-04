"""
Camera Calibration Module for "Measure It" Assessment (xis.ai v2.0)
Item 1: OpenCV Camera Calibration using 7x9 inner-corner checkerboard.
Physical square size: 20.0 mm.
"""

import os
import glob
import json
import numpy as np
import cv2

CHECKERBOARD_SIZES = [(7, 9), (9, 7)]
SQUARE_SIZE_MM = 20.0
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate_camera(image_dir="calibration/images", output_dir="calibration"):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + 
                         glob.glob(os.path.join(image_dir, "*.jpeg")) + 
                         glob.glob(os.path.join(image_dir, "*.png")))
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No calibration images found in {image_dir}")

    print(f"[*] Found {len(image_paths)} calibration images in {image_dir}")
    
    objpoints = []  # 3D points in real world space (mm)
    imgpoints = []  # 2D points in image plane (pixels)
    successful_images = []
    failed_images = []
    
    image_size = None

    for idx, path in enumerate(image_paths, 1):
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"[-] Could not read image: {filename}")
            failed_images.append(filename)
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])  # (width, height)
            print(f"[*] Native Image Resolution: {image_size[0]} x {image_size[1]}")

        found = False
        corners = None
        used_pattern = None

        # Try standard pattern sizes
        for pattern_size in CHECKERBOARD_SIZES:
            cols, rows = pattern_size
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, detected_corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
            
            # Fallback to Sector-Based detector if classic detector fails
            if not ret and hasattr(cv2, "findChessboardCornersSB"):
                ret, detected_corners = cv2.findChessboardCornersSB(gray, (cols, rows), cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                found = True
                corners = detected_corners
                used_pattern = pattern_size
                break

        if found:
            # Prepare 3D coordinates based on verified pattern orientation
            cols, rows = used_pattern
            objp = np.zeros((rows * cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQUARE_SIZE_MM

            # Refine corner locations to sub-pixel accuracy
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
            objpoints.append(objp)
            imgpoints.append(corners_subpix)
            successful_images.append(filename)
            print(f"  [+] Image {idx:02d}/{len(image_paths):02d} ({filename}): Detected {used_pattern[0]}x{used_pattern[1]} corners")
        else:
            print(f"  [-] Image {idx:02d}/{len(image_paths):02d} ({filename}): Corner detection failed")
            failed_images.append(filename)

    valid_count = len(objpoints)
    print(f"\n[*] Calibration Summary: {valid_count} / {len(image_paths)} images successfully detected")
    if valid_count < 20:
        raise ValueError(f"Requirement failed: Assessment requires >= 20 valid calibration images, but found {valid_count}.")

    print("[*] Running cv2.calibrateCamera...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # Compute per-image and overall reprojection errors
    total_error = 0.0
    per_image_errors = {}
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        per_image_errors[successful_images[i]] = float(error)
        total_error += error

    mean_reprojection_error = float(total_error / len(objpoints))
    print(f"[+] Camera calibration complete!")
    print(f"    - Mean Reprojection Error: {mean_reprojection_error:.4f} pixels")
    print(f"    - Focal Length fx: {camera_matrix[0,0]:.2f}, fy: {camera_matrix[1,1]:.2f}")
    print(f"    - Principal Point cx: {camera_matrix[0,2]:.2f}, cy: {camera_matrix[1,2]:.2f}")
    print(f"    - Distortion Coeffs (k1, k2, p1, p2, k3): {dist_coeffs.ravel()[:5]}")

    # Save to camera_params.npz
    npz_path = os.path.join(output_dir, "camera_params.npz")
    np.savez(
        npz_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        mean_reprojection_error=mean_reprojection_error
    )
    print(f"[+] Saved parameters to: {npz_path}")

    # Save human-readable calibration report JSON
    report = {
        "assessment_item": "Item 1 - Camera Calibration",
        "camera_model": "Google Pixel 6 Pro (1x locked zoom)",
        "image_resolution": {"width": image_size[0], "height": image_size[1]},
        "checkerboard": {
            "inner_corners": [7, 9],
            "square_size_mm": SQUARE_SIZE_MM
        },
        "image_counts": {
            "total_images": len(image_paths),
            "successful_images": valid_count,
            "failed_images": len(failed_images)
        },
        "mean_reprojection_error_px": round(mean_reprojection_error, 4),
        "camera_matrix_K": camera_matrix.tolist(),
        "distortion_coefficients_D": dist_coeffs.ravel().tolist(),
        "per_image_reprojection_error_px": per_image_errors
    }
    json_path = os.path.join(output_dir, "calibration_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Saved calibration report to: {json_path}")

    # Generate side-by-side visual verification image (raw vs. undistorted)
    sample_img_path = os.path.join(image_dir, successful_images[0])
    raw_img = cv2.imread(sample_img_path)
    
    # Generate optimal new camera matrix for full field-of-view visualization
    h, w = raw_img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(raw_img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Downscale for visual side-by-side comparison image
    disp_w = 960
    disp_h = int(h * (disp_w / w))
    raw_small = cv2.resize(raw_img, (disp_w, disp_h))
    undist_small = cv2.resize(undistorted_img, (disp_w, disp_h))

    # Draw straight reference guide lines across the checkerboard edges to visualize distortion removal
    for y_line in np.linspace(disp_h * 0.2, disp_h * 0.8, 5, dtype=int):
        cv2.line(raw_small, (0, y_line), (disp_w, y_line), (0, 0, 255), 2)
        cv2.line(undist_small, (0, y_line), (disp_w, y_line), (0, 255, 0), 2)

    cv2.putText(raw_small, "RAW (Distorted) with straight red reference", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(undist_small, f"UNDISTORTED (Error: {mean_reprojection_error:.3f} px)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    comparison = np.hstack([raw_small, undist_small])
    vis_path = os.path.join(output_dir, "verification_undistorted.jpg")
    cv2.imwrite(vis_path, comparison)
    print(f"[+] Saved visual verification image to: {vis_path}")

    return mean_reprojection_error, valid_count

if __name__ == "__main__":
    calibrate_camera()
