"""
[ARCHIVED]
Reason: Old test script
Date: 2026-01-31
"""

"""
=============================================================================
util/test_outlier_align.py - 아웃라이어 정렬 테스트 (개발용)
=============================================================================

특정 아웃라이어 이미지에 수동 오프셋을 적용하여 결과 확인.
개발 중 알고리즘 검증에 사용.
=============================================================================
"""
import cv2
import numpy as np
import os

def test_align(filename, dx, dy, subfolder="2026-01-01"):
    input_path = os.path.join("input", subfolder, filename)
    output_dir = "temp/test_outlier"
    os.makedirs(output_dir, exist_ok=True)
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Failed to load: {input_path}")
        return

    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    output_path = os.path.join(output_dir, f"aligned_{filename}")
    cv2.imwrite(output_path, aligned)
    print(f"Aligned: {output_path} (dx={dx}, dy={dy})")

# Test cases
test_align("2026-01-01_06-24-00.jpg", -4.5, -1.4)
test_align("2026-01-01_06-30-00.jpg", -4.3, -13.6)
test_align("2026-01-01_06-42-00.jpg", -5.2, 0.6)
test_align("2026-01-01_07-00-00.jpg", -5.0, 0.8)
