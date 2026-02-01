"""
evaluate_stabilization.py - Evaluate stabilization and archive logs for comparison
"""
import cv2
import numpy as np
import os
import shutil
from glob import glob
from datetime import datetime

def create_gradient(img):
    # Safe grayscale conversion
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        gray = img[:,:,0]
    else:
        gray = img
        
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def get_image_time(filename):
    # Filename format: YYYY-MM-DD_HH-MM-SS.jpg
    try:
        basename = os.path.basename(filename)
        time_str = basename.split('_')[1].split('.')[0] # HH-MM-SS
        h, m, s = map(int, time_str.split('-'))
        return h * 3600 + m * 60 + s
    except:
        return None

def find_closest_image(folder, target_hour):
    files = sorted(glob(os.path.join(folder, "*.jpg"))) + sorted(glob(os.path.join(folder, "*.JPG")))
    if not files:
        return None
        
    target_seconds = target_hour * 3600
    closest_file = None
    min_diff = float('inf')
    
    for f in files:
        seconds = get_image_time(f)
        if seconds is None:
            continue
        diff = abs(seconds - target_seconds)
        if diff < min_diff:
            min_diff = diff
            closest_file = f
            
    return closest_file

def calculate_shift(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    
    if img1 is None or img2 is None:
        return None
        
    # Resize for speed and robustness
    scale = 0.5
    img1 = cv2.resize(img1, None, fx=scale, fy=scale)
    img2 = cv2.resize(img2, None, fx=scale, fy=scale)
    
    # 1. Check Rotation (ECC)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
    
    try:
        (cc, warp_matrix) = cv2.findTransformECC(gray1, gray2, warp_matrix, warp_mode, criteria)
        rot_deg = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0]) * 180 / np.pi
    except:
        rot_deg = 0.0
        
    # 2. Check Translation (Phase Correlation)
    h, w = gray1.shape
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
    gray2_derot = cv2.warpAffine(gray2, M, (w, h))
    
    grad1 = create_gradient(img1) 
    grad2 = create_gradient(gray2_derot)
    
    hann = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    shift, response = cv2.phaseCorrelate(np.float32(grad1)*hann, np.float32(grad2)*hann)
    dx, dy = -shift[0], -shift[1]
    
    dx /= scale
    dy /= scale
    
    return dx, dy, rot_deg

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfolder", "-f", help="Subfolder name inside output directory")
    args = parser.parse_args()

    base_dir = os.path.join("output", args.subfolder) if args.subfolder else "output"
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' not found.")
        return

    # === New Archiving Logic ===
    # 1. Create directory based on timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_root = "evaluation_reports"
    report_dir = os.path.join(archive_root, timestamp)
    os.makedirs(report_dir, exist_ok=True)
    
    output_file = os.path.join(report_dir, "evaluation_result.txt")
    
    # Context manager or helper for logging
    def log(msg):
        print(msg)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            
    log(f"=== Stabilization Evaluation Report ===")
    log(f"Date: {timestamp}")
    log(f"Report Location: {report_dir}\n")

    # 2. Backup relevant logs and code
    log("[Backing up logs and code]")
    
    # Define prefixes to look for (standard, and subfolder-specific if provided)
    prefixes = [""]
    if args.subfolder:
        prefixes.insert(0, f"{args.subfolder}_")
        
    log_names = ["full_log.txt", "analysis_log.json", "refine_log.json"]
    files_to_backup = ["timelapse_stabilizer.py"]
    
    # Resolve log paths
    for name in log_names:
        found = False
        for prefix in prefixes:
            candidate = os.path.join(base_dir, f"{prefix}{name}")
            if os.path.exists(candidate):
                files_to_backup.append(candidate)
                found = True
                break
        if not found:
            # Add missing entry for logging
            files_to_backup.append(os.path.join(base_dir, name))
    
    for src in files_to_backup:
        if os.path.exists(src):
            basename = os.path.basename(src)
            # Add 'backup_' prefix to python files to distinguish them
            if src.endswith(".py"):
                basename = f"backup_{basename}"
            
            dst = os.path.join(report_dir, basename)
            shutil.copy2(src, dst)
            log(f"  Matched: {os.path.basename(src)} -> {basename}")
        else:
            log(f"  Missing: {os.path.basename(src)}")
    log("-" * 30 + "\n")

    days = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    if not days:
        log("No output folders found.")
        return

    # Select 5 dates (First, Last, and 3 evenly spaced in between)
    num_samples = 5
    if len(days) <= num_samples:
        selected_days = days
    else:
        indices = np.linspace(0, len(days)-1, num_samples, dtype=int)
        selected_days = [days[i] for i in indices]
        
    log(f"Selected Dates for Evaluation: {selected_days}")
    
    # Define time targets: Morning (~09:00), Lunch (~13:00), Evening (~18:00)
    times = [
        ("Morning", 9), 
        ("Lunch", 13), 
        ("Evening", 18)
    ]
    
    for time_name, target_hour in times:
        log(f"\n[{time_name}] (Target: {target_hour:02d}:00)")
        log(f"{'Date':<15} {'dX (px)':>10} {'dY (px)':>10} {'Rot (deg)':>10}")
        log("-" * 50)
        
        # Reference day is the first one
        ref_day = selected_days[0]
        ref_path = find_closest_image(os.path.join(base_dir, ref_day), target_hour)
        
        if not ref_path:
            log(f" Reference image missing for {ref_day}")
            continue
            
        log(f"{ref_day:<15} {'REF':>10} {'REF':>10} {'REF':>10}")
        
        for day in selected_days[1:]:
            curr_path = find_closest_image(os.path.join(base_dir, day), target_hour)
            if not curr_path:
                log(f"{day:<15} {'MISSING':>32}")
                continue
                
            res = calculate_shift(ref_path, curr_path)
            if res:
                dx, dy, rot = res
                log(f"{day:<15} {dx:10.2f} {dy:10.2f} {rot:10.4f}")
            else:
                 log(f"{day:<15} {'FAIL':>32}")

    log(f"\nSaved full report to: {output_file}")

if __name__ == "__main__":
    main()
