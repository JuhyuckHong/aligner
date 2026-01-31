"""
[ARCHIVED]
Reason: Test script for Day Bridging strategy (Noon-to-Noon).
Date: 2026-01-31
"""

"""
낮 시간대 브릿지 테스트
- 전날 낮 vs 다음날 낮 비교 (조명 동일!)
"""
import cv2
import numpy as np
import sys
import os

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

def day_bridge_offset(folder1, folder2, target_hours=[10, 11, 12, 13, 14], scale=0.5):
    """
    낮 시간대 브릿지: 같은 시간대끼리 비교 후 중앙값
    """
    from glob import glob
    
    dx_list, dy_list, resp_list = [], [], []
    
    for hour in target_hours:
        # 해당 시간대 파일 찾기
        pattern1 = os.path.join(folder1, f"*_{hour:02d}-*-*.jpg")
        pattern2 = os.path.join(folder2, f"*_{hour:02d}-*-*.jpg")
        
        files1 = sorted(glob(pattern1))
        files2 = sorted(glob(pattern2))
        
        if not files1 or not files2:
            continue
        
        # 중간 파일 선택
        img1_path = files1[len(files1)//2]
        img2_path = files2[len(files2)//2]
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            continue
        
        edge1 = create_edge(img1)
        edge2 = create_edge(img2)
        
        if scale != 1.0:
            edge1 = cv2.resize(edge1, None, fx=scale, fy=scale)
            edge2 = cv2.resize(edge2, None, fx=scale, fy=scale)
        
        dx, dy, resp = phase_correlation(edge1, edge2)
        dx /= scale
        dy /= scale
        
        if resp > 0.4:  # 신뢰도 높은 매칭만
            dx_list.append(dx)
            dy_list.append(dy)
            resp_list.append(resp)
            print(f"  {hour:02d}:00 - dx={dx:.2f}, dy={dy:.2f}, resp={resp:.3f}")
    
    if not dx_list:
        return 0, 0, 0
    
    # 중앙값
    final_dx = np.median(dx_list)
    final_dy = np.median(dy_list)
    avg_resp = np.mean(resp_list)
    
    return final_dx, final_dy, avg_resp

def apply_offset(img_path, dx, dy, output_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(output_path, aligned, [cv2.IMWRITE_JPEG_QUALITY, 98])
    return output_path

if __name__ == "__main__":
    folder1 = "input/2026-01-02"
    folder2 = "input/2026-01-03"
    
    if len(sys.argv) >= 3:
        folder1 = sys.argv[1]
        folder2 = sys.argv[2]
    
    print(f"=== Day Bridge (Same Time Comparison) ===")
    print(f"Folder 1: {folder1}")
    print(f"Folder 2: {folder2}")
    print()
    
    dx, dy, resp = day_bridge_offset(folder1, folder2)
    
    print()
    print(f"=== Final (Median) ===")
    print(f"Offset: dx={dx:.2f}, dy={dy:.2f}")
    print(f"Avg Response: {resp:.4f}")
    
    # 저녁→아침 이미지에 offset 적용해서 검증
    night_img = os.path.join(folder1, sorted([f for f in os.listdir(folder1) if "18-00" in f])[0]) if os.path.isdir(folder1) else None
    morning_img = os.path.join(folder2, sorted([f for f in os.listdir(folder2) if "06-00" in f])[0]) if os.path.isdir(folder2) else None
    
    if night_img and morning_img and os.path.exists(night_img) and os.path.exists(morning_img):
        os.makedirs("temp", exist_ok=True)
        aligned_path = "temp/bridge_aligned.jpg"
        apply_offset(morning_img, dx, dy, aligned_path)
        print()
        print(f">>> Aligned morning image: {aligned_path}")
        print(f">>> 검증: python util/manual_align_gui.py --ref {night_img} --mov {aligned_path}")
