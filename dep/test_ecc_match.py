"""
[ARCHIVED]
Reason: Test script for ECC alignment (Translation/Affine). Found to be too slow/unstable.
Date: 2026-01-31
"""

"""
ECC 기반 정밀 매칭 테스트
- Phase correlation보다 sub-pixel 정밀도가 높음
"""
import cv2
import numpy as np
import sys
import os
from glob import glob

def ecc_align(ref_path, mov_path, scale=0.5):
    """
    ECC (Enhanced Correlation Coefficient)로 정밀 정합
    Translation만 추정
    """
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    mov = cv2.imread(mov_path, cv2.IMREAD_GRAYSCALE)
    
    if scale != 1.0:
        ref_s = cv2.resize(ref, None, fx=scale, fy=scale)
        mov_s = cv2.resize(mov, None, fx=scale, fy=scale)
    else:
        ref_s, mov_s = ref, mov
    
    # Translation only (2x3 matrix)
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
    
    try:
        cc, warp_matrix = cv2.findTransformECC(mov_s, ref_s, warp_matrix, warp_mode, criteria)
        
        dx = warp_matrix[0, 2] / scale
        dy = warp_matrix[1, 2] / scale
        
        return dx, dy, cc, True
    except Exception as e:
        print(f"ECC failed: {e}")
        return 0, 0, 0, False

def find_image_at_time(folder, hour, minute=0):
    pattern = os.path.join(folder, f"*_{hour:02d}-{minute:02d}-*.jpg")
    files = glob(pattern)
    if files:
        return files[0]
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
    folder1 = "input/2026-01-02"
    folder2 = "input/2026-01-03"
    hour = 18
    
    if len(sys.argv) >= 3:
        folder1 = sys.argv[1]
        folder2 = sys.argv[2]
    if len(sys.argv) >= 4:
        hour = int(sys.argv[3])
    
    print(f"=== ECC Precision Matching ({hour}:00) ===")
    
    img1_path = find_image_at_time(folder1, hour)
    img2_path = find_image_at_time(folder2, hour)
    
    if not img1_path or not img2_path:
        print("Cannot find images")
        sys.exit(1)
    
    print(f"Ref: {os.path.basename(img1_path)}")
    print(f"Mov: {os.path.basename(img2_path)}")
    print()
    
    # 여러 스케일로 테스트
    for scale in [0.25, 0.5, 1.0]:
        dx, dy, cc, success = ecc_align(img1_path, img2_path, scale=scale)
        if success:
            print(f"Scale {scale}: dx={dx:.3f}, dy={dy:.3f}, cc={cc:.4f}")
    
    # Full resolution 결과 사용
    dx, dy, cc, success = ecc_align(img1_path, img2_path, scale=1.0)
    
    if success:
        print()
        print(f"=== Final (Full Resolution) ===")
        print(f"Offset: dx={dx:.3f}, dy={dy:.3f}")
        print(f"Correlation: {cc:.4f}")
        
        # 아침 이미지에 적용
        morning_files = glob(os.path.join(folder2, "*_06-00-*.jpg"))
        if morning_files:
            os.makedirs("temp", exist_ok=True)
            apply_and_save(morning_files[0], dx, dy, "temp/ecc_aligned.jpg")
            
            night_files = glob(os.path.join(folder1, "*_18-00-*.jpg"))
            night_img = night_files[0] if night_files else img1_path
            
            print()
            print(f">>> 검증: python util/manual_align_gui.py --ref {night_img} --mov temp/ecc_aligned.jpg")
