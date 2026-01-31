"""
ORB Feature Matching 기반 정합
- 조명 변화에 더 강건한 특징점 매칭
"""
import cv2
import numpy as np
import sys
import os
from glob import glob

def orb_match_offset(ref_path, mov_path, max_features=1000):
    """
    ORB + RANSAC으로 translation 추정
    """
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    mov = cv2.imread(mov_path, cv2.IMREAD_GRAYSCALE)
    
    # ORB detector
    orb = cv2.ORB_create(nfeatures=max_features)
    
    kp1, des1 = orb.detectAndCompute(ref, None)
    kp2, des2 = orb.detectAndCompute(mov, None)
    
    if des1 is None or des2 is None:
        return 0, 0, 0, False
    
    # Brute force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if len(matches) < 10:
        print(f"  Too few matches: {len(matches)}")
        return 0, 0, len(matches), False
    
    # Sort by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Get matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    
    # RANSAC to find translation
    # Simple approach: median of differences
    diffs = pts1 - pts2
    
    # RANSAC-like: use only consistent matches
    dx_list = diffs[:, 0]
    dy_list = diffs[:, 1]
    
    # Remove outliers using IQR
    def remove_outliers(data):
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        mask = (data >= q1 - 1.5*iqr) & (data <= q3 + 1.5*iqr)
        return data[mask]
    
    dx_clean = remove_outliers(dx_list)
    dy_clean = remove_outliers(dy_list)
    
    if len(dx_clean) < 5 or len(dy_clean) < 5:
        dx = np.median(dx_list)
        dy = np.median(dy_list)
    else:
        dx = np.median(dx_clean)
        dy = np.median(dy_clean)
    
    return dx, dy, len(matches), True

def find_image_at_time(folder, hour):
    for m in range(0, 60, 6):
        pattern = os.path.join(folder, f"*_{hour:02d}-{m:02d}-*.jpg")
        files = glob(pattern)
        if files:
            return files[0]
    return None

def apply_and_save(mov_path, dx, dy, output_path):
    img = cv2.imread(mov_path)
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(output_path, aligned, [cv2.IMWRITE_JPEG_QUALITY, 98])

if __name__ == "__main__":
    # 밤 vs 아침 직접 비교!
    ref_path = "input/2026-01-02/2026-01-02_18-00-00.jpg"
    mov_path = "input/2026-01-03/2026-01-03_06-00-00.jpg"
    
    if len(sys.argv) >= 3:
        ref_path = sys.argv[1]
        mov_path = sys.argv[2]
    
    print(f"=== ORB Feature Matching ===")
    print(f"Ref: {os.path.basename(ref_path)}")
    print(f"Mov: {os.path.basename(mov_path)}")
    print()
    
    dx, dy, n_matches, success = orb_match_offset(ref_path, mov_path)
    
    print(f"Matches: {n_matches}")
    print(f"Offset: dx={dx:.3f}, dy={dy:.3f}")
    
    if success:
        os.makedirs("temp", exist_ok=True)
        apply_and_save(mov_path, dx, dy, "temp/orb_aligned.jpg")
        print()
        print(f">>> 검증: python util/manual_align_gui.py --ref {ref_path} --mov temp/orb_aligned.jpg")
