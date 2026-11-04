"""
Undistort Dataset Images for "Measure It" Assessment (xis.ai v2.0)
Applies calibrated camera matrix K and distortion coefficients D to raw photos.
"""

import os
import glob
import numpy as np
import cv2

def load_camera_parameters(params_path="calibration/camera_params.npz"):
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Calibration parameters not found at {params_path}. Run calibration/calibrate.py first.")
    
    data = np.load(params_path)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    print(f"[+] Loaded calibration parameters from {params_path}")
    print(f"    - fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}, cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
    return camera_matrix, dist_coeffs

def undistort_directory(input_dir, output_dir, camera_matrix, dist_coeffs):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) +
                         glob.glob(os.path.join(input_dir, "*.jpeg")) +
                         glob.glob(os.path.join(input_dir, "*.png")))
    
    if len(image_paths) == 0:
        print(f"[-] No images found in {input_dir}")
        return 0

    print(f"[*] Undistorting {len(image_paths)} images from '{input_dir}' to '{output_dir}'...")
    count = 0
    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [-] Failed to read: {filename}")
            continue

        h, w = img.shape[:2]
        # Undistort keeping intrinsic geometry intact
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
        cv2.imwrite(out_path, undistorted)
        count += 1
        if idx % 15 == 0 or idx == len(image_paths):
            print(f"    [+] Processed {idx}/{len(image_paths)}: {filename}")

    print(f"[+] Successfully undistorted {count} images in {output_dir}\n")
    return count

def main():
    K, D = load_camera_parameters()
    
    # 1. Undistort primary dataset
    raw_dataset_dir = "dataset/images_raw"
    undist_dataset_dir = "dataset/images_undistorted"
    if os.path.exists(raw_dataset_dir):
        undistort_directory(raw_dataset_dir, undist_dataset_dir, K, D)

    # 2. Undistort tilt test images (Item 8)
    tilt_raw_dir = "measurement/tilt_images"
    tilt_undist_dir = "measurement/tilt_images_undistorted"
    if os.path.exists(tilt_raw_dir):
        undistort_directory(tilt_raw_dir, tilt_undist_dir, K, D)

if __name__ == "__main__":
    main()
