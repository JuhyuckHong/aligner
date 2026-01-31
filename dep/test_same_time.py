"""
[ARCHIVED]
Reason: Test script for comparing same-time images across days.
Date: 2026-01-31
"""

"""
같은 시간대 직접 비교 테스트
전날 18:00 vs 다음날 18:00 (조명 동일)
"""
import cv2
import numpy as np
import sys
import os
from glob import glob

def create_edge(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    return cv2.Canny(blurred, 50, 150)

def phase_correlation(ref, mov):
    f1 = np.float32(ref)
    f2 = np.float32(mov)
    h, w = f1.shape
    hann = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    shift, response = cv2.phaseCorrelate(f1 * hann, f2 * hann)
    return -shift[0], -shift[1], response

def find_image_at_time(folder, hour, minute=0):
    """특정 시간의 이미지 찾기"""
    pattern = os.path.join(folder, f"*_{hour:02d}-{minute:02d}-*.jpg")
    files = glob(pattern)
    if files:
        return files[0]
    # 근처 시간 찾기
    for m in range(0, 60, 6):
        pattern = os.path.join(folder, f"*_{hour:02d}-{m:02d}-*.jpg")
        files = glob(pattern)
        if files:
            return files[0]
    return None

def same_time_offset(folder1, folder2, hour=18, scale=0.5):
    """같은 시간 이미지 비교"""
    img1_path = find_image_at_time(folder1, hour)
    img2_path = find_image_at_time(folder2, hour)
    
    if not img1_path or not img2_path:
        print(f"Cannot find {hour}:00 images")
        return 0, 0, 0, None, None
    
    print(f"Comparing:")
    print(f"  {os.path.basename(img1_path)}")
    print(f"  {os.path.basename(img2_path)}")
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    edge1 = create_edge(img1)
    edge2 = create_edge(img2)
    
    if scale != 1.0:
        edge1_s = cv2.resize(edge1, None, fx=scale, fy=scale)
        edge2_s = cv2.resize(edge2, None, fx=scale, fy=scale)
    else:
        edge1_s, edge2_s = edge1, edge2
    
    dx, dy, resp = phase_correlation(edge1_s, edge2_s)
    dx /= scale
    dy /= scale
    
    return dx, dy, resp, img1_path, img2_path

def apply_and_save(mov_path, dx, dy, output_path):
    img = cv2.imread(mov_path)
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(output_path, aligned, [cv2.IMWRITE_JPEG_QUALITY, 98])
    return output_path

if __name__ == "__main__":
    folder1 = "input/2026-01-02"
    folder2 = "input/2026-01-03"
    hour = 18
    
    if len(sys.argv) >= 3:
        folder1 = sys.argv[1]
        folder2 = sys.argv[2]
    if len(sys.argv) >= 4:
        hour = int(sys.argv[3])
    
    print(f"=== Same Time Comparison ({hour}:00) ===")
    print(f"Folder 1: {folder1}")
    print(f"Folder 2: {folder2}")
    print()
    
    dx, dy, resp, img1_path, img2_path = same_time_offset(folder1, folder2, hour)
    
    print()
    print(f"Offset: dx={dx:.2f}, dy={dy:.2f}")
    print(f"Response: {resp:.4f}")
    
    if img1_path and img2_path:
        # 다음날 아침 이미지에 offset 적용해서 저녁과 비교
        morning_pattern = os.path.join(folder2, "*_06-00-*.jpg")
        morning_files = glob(morning_pattern)
        if morning_files:
            morning_img = morning_files[0]
            os.makedirs("temp", exist_ok=True)
            aligned_path = "temp/same_time_aligned.jpg"
            apply_and_save(morning_img, dx, dy, aligned_path)
            
            night_pattern = os.path.join(folder1, "*_18-00-*.jpg")
            night_files = glob(night_pattern)
            night_img = night_files[0] if night_files else img1_path
            
            print()
            print(f">>> 검증: python util/manual_align_gui.py --ref {night_img} --mov {aligned_path}")
