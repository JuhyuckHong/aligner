"""
[ARCHIVED]
Reason: Integrated into stabilize_phase.py (Late/Early Correction)
Date: 2026-01-31
"""

"""
=============================================================================
util/refine_day_alignment.py - 날짜별 드리프트 후처리 보정 (단독 실행용)
=============================================================================

이미 보정된 이미지(output/)에서 날짜 간 드리프트를 추가로 보정.
메인 스크립트(stabilize_phase.py)에 통합되어 있으나, 단독 실행도 가능.

사용법:
  python util/refine_day_alignment.py

입출력:
  - 입력: output/[날짜]/
  - 출력: output_refined/[날짜]/ + combined_refined.mp4
=============================================================================
"""
import cv2
import numpy as np
import os
import subprocess
from glob import glob
from tqdm import tqdm
import random

def create_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    return cv2.Canny(blurred, 50, 150)

def phase_correlation(ref, mov):
    f1 = np.float32(ref)
    f2 = np.float32(mov)
    h, w = f1.shape
    hann = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    shift, resp = cv2.phaseCorrelate(f1 * hann, f2 * hann)
    return -shift[0], -shift[1], resp

def get_day_samples(folder_path, n_samples=10):
    """Get random samples from daytime hours (avoid dark hours)"""
    images = sorted(glob(os.path.join(folder_path, "*.jpg")))
    if not images:
        return []
    # Pick from middle 50% (roughly daytime)
    mid_range = images[len(images)//4 : 3*len(images)//4]
    if len(mid_range) < n_samples:
        return mid_range
    return random.sample(mid_range, n_samples)

def calculate_day_offset(folder1, folder2, n_samples=10, scale=0.5):
    """Calculate median offset between two days using random sampling"""
    samples1 = get_day_samples(folder1, n_samples)
    samples2 = get_day_samples(folder2, n_samples)
    
    if not samples1 or not samples2:
        return 0, 0, 0
    
    dx_list = []
    dy_list = []
    
    for s1 in samples1:
        img1 = cv2.imread(s1)
        edge1 = create_edge(img1)
        e1 = cv2.resize(edge1, None, fx=scale, fy=scale)
        
        for s2 in samples2:
            img2 = cv2.imread(s2)
            edge2 = create_edge(img2)
            e2 = cv2.resize(edge2, None, fx=scale, fy=scale)
            
            dx, dy, resp = phase_correlation(e1, e2)
            dx /= scale
            dy /= scale
            
            # Only use good matches
            if resp > 0.05:
                dx_list.append(dx)
                dy_list.append(dy)
    
    if not dx_list:
        return 0, 0, 0
    
    # Use median for robustness
    median_dx = np.median(dx_list)
    median_dy = np.median(dy_list)
    n_valid = len(dx_list)
    
    return median_dx, median_dy, n_valid

def main():
    input_base = "output"
    output_base = "output_refined"
    
    # Find all day folders
    folders = sorted([d for d in os.listdir(input_base) 
                      if os.path.isdir(os.path.join(input_base, d)) and d != 'logs'])
    
    print(f"Found {len(folders)} day folders: {folders}")
    
    # Step 1: Calculate day-to-day offsets
    print("\n=== Calculating Day-Level Offsets ===")
    day_offsets = {folders[0]: (0.0, 0.0)}  # First day is reference
    
    cumulative_dx = 0.0
    cumulative_dy = 0.0
    
    for i in range(1, len(folders)):
        prev_folder = os.path.join(input_base, folders[i-1])
        curr_folder = os.path.join(input_base, folders[i])
        
        # Calculate offset: how much should curr move to align with prev?
        dx, dy, n_valid = calculate_day_offset(prev_folder, curr_folder, n_samples=30)
        
        cumulative_dx += dx
        cumulative_dy += dy
        day_offsets[folders[i]] = (cumulative_dx, cumulative_dy)
        
        print(f"  {folders[i-1]} -> {folders[i]}: dx={dx:.1f}, dy={dy:.1f} (from {n_valid} pairs)")
        print(f"    Cumulative: dx={cumulative_dx:.1f}, dy={cumulative_dy:.1f}")
    
    # Step 2: Apply offsets to all images
    print("\n=== Applying Day-Level Corrections ===")
    os.makedirs(output_base, exist_ok=True)
    
    for folder in folders:
        input_folder = os.path.join(input_base, folder)
        output_folder = os.path.join(output_base, folder)
        os.makedirs(output_folder, exist_ok=True)
        
        offset_dx, offset_dy = day_offsets[folder]
        images = sorted(glob(os.path.join(input_folder, "*.jpg")))
        
        print(f"  {folder}: applying dx={offset_dx:.1f}, dy={offset_dy:.1f} to {len(images)} images")
        
        for img_path in tqdm(images, desc=f"    {folder}", leave=False):
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
            M = np.float32([[1, 0, offset_dx], [0, 1, offset_dy]])
            corrected = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            output_path = os.path.join(output_folder, os.path.basename(img_path))
            cv2.imwrite(output_path, corrected, [cv2.IMWRITE_JPEG_QUALITY, 98])
    
    # Step 3: Create video
    print("\n=== Creating Video ===")
    all_images = []
    for folder in folders:
        folder_path = os.path.join(output_base, folder)
        images = sorted(glob(os.path.join(folder_path, "*.jpg")))
        all_images.extend(images)
    
    print(f"Total frames: {len(all_images)}")
    
    video_path = os.path.join(output_base, "combined_refined.mp4")
    list_file = os.path.join(output_base, "ffmpeg_list.txt")
    fps = 30
    crf = 23
    
    with open(list_file, "w") as f:
        for img in all_images:
            f.write(f"file '{os.path.abspath(img).replace(chr(92), '/')}'\n")
            f.write(f"duration {1/fps}\n")
    
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
           "-vf", "scale=-2:1080",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", str(crf), "-preset", "medium", "-r", str(fps),
           video_path]
    
    subprocess.run(cmd, capture_output=True)
    os.remove(list_file)
    
    print(f"\n✓ Done! Video: {video_path}")
    print(f"  Compare with output/combined_all.mp4")

if __name__ == "__main__":
    main()
