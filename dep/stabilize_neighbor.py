"""
[ARCHIVED]
Reason: Old prototype for neighbor alignment
Date: 2026-01-31
"""

"""
=============================================================================
util/stabilize_neighbor.py - 이웃 정렬 테스트 스크립트 (개발용)
=============================================================================

이웃 프레임 간 Phase Correlation 정합의 원형 테스트 스크립트.
메인 스크립트(stabilize_phase.py)에 통합되어 더 이상 사용하지 않음.

참고용으로 보관.
=============================================================================
"""
import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

def get_images(input_dir, ext='jpg'):
    patterns = [f"*.{ext}", f"*.{ext.upper()}"]
    images = []
    for pattern in patterns:
        images.extend(glob(os.path.join(input_dir, pattern)))
    return sorted(images)

def create_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def phase_correlation(ref, mov):
    f1 = np.float32(ref)
    f2 = np.float32(mov)
    
    h, w = f1.shape
    hann_row = np.hanning(h)
    hann_col = np.hanning(w)
    hann_2d = np.outer(hann_row, hann_col).astype(np.float32)
    
    f1 = f1 * hann_2d
    f2 = f2 * hann_2d
    
    shift, response = cv2.phaseCorrelate(f1, f2)
    return -shift[0], -shift[1], response

def stabilize_neighbor(input_dir, output_dir, ext='jpg'):
    image_paths = get_images(input_dir, ext)
    if not image_paths:
        print("No images found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    n = len(image_paths)
    
    print(f"Stabilizing {n} frames using Neighbor Alignment (Accumulated)...")
    
    # Process first frame
    first_path = image_paths[0]
    prev_img = cv2.imread(first_path)
    h, w = prev_img.shape[:2]
    
    # Save first frame as is
    out_name = os.path.basename(first_path)
    cv2.imwrite(os.path.join(output_dir, out_name), prev_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    
    prev_edge = create_edge(prev_img)
    scale = 0.5
    prev_edge_small = cv2.resize(prev_edge, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Accumulated transform (dx, dy)
    acc_dx = 0.0
    acc_dy = 0.0
    
    full_log = []
    outliers = []
    
    # Log first frame
    full_log.append(f"{os.path.basename(first_path)}\tdx=0.0\tdy=0.0\tresponse=1.000\tstatus=REF")
    
    for i in tqdm(range(1, n)):
        curr_path = image_paths[i]
        filename = os.path.basename(curr_path)
        curr_img = cv2.imread(curr_path)
        if curr_img is None: continue
        
        curr_edge = create_edge(curr_img)
        curr_edge_small = cv2.resize(curr_edge, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # Calculate offset from prev to curr
        dx, dy, response = phase_correlation(prev_edge_small, curr_edge_small)
        
        dx /= scale
        dy /= scale
        
        status = "OK"
        # Outlier rejection (simple)
        if abs(dx) > 50 or abs(dy) > 50 or response < 0.03:
            status = "OUTLIER"
            outliers.append(f"{filename}\tdx={dx:.1f}, dy={dy:.1f}, resp={response:.3f}")
            print(f"  Skipped {filename}: dx={dx:.1f}, dy={dy:.1f}, resp={response:.3f}")
            dx = 0
            dy = 0
        
        full_log.append(f"{filename}\tdx={dx:.1f}\tdy={dy:.1f}\tresponse={response:.3f}\tstatus={status}")
        
        # Accumulate transforms
        acc_dx += dx
        acc_dy += dy
        
        # Apply accumulated transform
        M = np.float32([[1, 0, acc_dx], [0, 1, acc_dy]])
        aligned = cv2.warpAffine(curr_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Save
        cv2.imwrite(os.path.join(output_dir, filename), aligned, [cv2.IMWRITE_JPEG_QUALITY, 98])
        
        # Update prev for next iteration
        prev_edge_small = curr_edge_small

    # Save logs
    with open(os.path.join(output_dir, "full_log.txt"), "w", encoding="utf-8") as f:
        f.write("Filename\tdx\tdy\tResponse\tStatus\n")
        f.write("\n".join(full_log))
        
    if outliers:
        with open(os.path.join(output_dir, "outliers.log"), "w", encoding="utf-8") as f:
            f.write("# Outliers (Neighbor Alignment)\n")
            f.write("# Threshold: shift > 50 or response < 0.03\n\n")
            f.write("\n".join(outliers))

# Run for 2026-01-01
stabilize_neighbor("input/2026-01-01", "output/neighbor_test_0101")
