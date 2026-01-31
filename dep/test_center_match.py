"""
[ARCHIVED]
Reason: Test script for Center-only Phase Correlation.
Date: 2026-01-31
"""

"""
중심부 25% Edge 비교 테스트
- 밤→아침 조도 차이가 큰 상황에서 중심부만 비교
"""
import cv2
import numpy as np
import sys
import os

def create_edge(img):
    """Canny edge 추출"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    return cv2.Canny(blurred, 50, 150)

def extract_center(img, ratio=0.5):
    """
    이미지 중심부 추출
    ratio=0.5 → 가로세로 50%씩 = 면적 25%
    """
    h, w = img.shape[:2]
    
    margin_h = int(h * (1 - ratio) / 2)
    margin_w = int(w * (1 - ratio) / 2)
    
    center = img[margin_h:h-margin_h, margin_w:w-margin_w]
    return center

def phase_correlation(ref, mov):
    """Phase correlation으로 offset 계산"""
    f1 = np.float32(ref)
    f2 = np.float32(mov)
    
    h, w = f1.shape
    hann = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    
    shift, response = cv2.phaseCorrelate(f1 * hann, f2 * hann)
    return -shift[0], -shift[1], response

def center_edge_offset(img1_path, img2_path, center_ratio=0.5, scale=0.5):
    """
    중심부 edge 기반 offset 계산
    
    Args:
        img1_path: 기준 이미지 (전날 밤)
        img2_path: 이동 이미지 (다음날 아침)
        center_ratio: 중심부 비율 (0.5 = 면적 25%)
        scale: 처리 스케일
    
    Returns:
        dx, dy, response
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Error: Cannot load images")
        return 0, 0, 0
    
    # Edge 추출
    edge1 = create_edge(img1)
    edge2 = create_edge(img2)
    
    # 중심부 추출
    center1 = extract_center(edge1, center_ratio)
    center2 = extract_center(edge2, center_ratio)
    
    # 스케일 다운
    if scale != 1.0:
        center1 = cv2.resize(center1, None, fx=scale, fy=scale)
        center2 = cv2.resize(center2, None, fx=scale, fy=scale)
    
    # Phase correlation
    dx, dy, response = phase_correlation(center1, center2)
    
    # 스케일 보정
    dx /= scale
    dy /= scale
    
    return dx, dy, response

def apply_offset(img_path, dx, dy, output_path):
    """offset 적용한 이미지 저장"""
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    cv2.imwrite(output_path, aligned, [cv2.IMWRITE_JPEG_QUALITY, 98])
    return output_path

def visualize_center_region(img_path, center_ratio=0.5, output_path=None):
    """중심부 영역 시각화"""
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    margin_h = int(h * (1 - center_ratio) / 2)
    margin_w = int(w * (1 - center_ratio) / 2)
    
    # 중심부 표시
    cv2.rectangle(img, (margin_w, margin_h), (w-margin_w, h-margin_h), (0, 255, 0), 3)
    
    # 정보 표시
    area_percent = center_ratio * center_ratio * 100
    cv2.putText(img, f"Center {area_percent:.0f}%", (margin_w + 10, margin_h + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, img)
    
    return img

# === 메인 테스트 ===
if __name__ == "__main__":
    # 테스트 이미지
    img1_path = "input/2026-01-01/2026-01-01_18-00-00.jpg"  # 전날 저녁
    img2_path = "input/2026-01-02/2026-01-02_06-00-00.jpg"  # 다음날 아침
    
    if len(sys.argv) >= 3:
        img1_path = sys.argv[1]
        img2_path = sys.argv[2]
    
    print(f"=== Center 25% Edge Matching ===")
    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    print()
    
    # 중심부 25% (ratio=0.5 → 가로세로 50%씩 = 면적 25%)
    dx, dy, response = center_edge_offset(img1_path, img2_path, center_ratio=0.5, scale=0.5)
    
    print(f"Offset: dx={dx:.2f}, dy={dy:.2f}")
    print(f"Response: {response:.4f}")
    print()
    
    # Align된 이미지 저장
    os.makedirs("temp", exist_ok=True)
    aligned_path = "temp/center_aligned.jpg"
    apply_offset(img2_path, dx, dy, aligned_path)
    print(f"Aligned image saved: {aligned_path}")
    
    # 시각화
    vis_path = "temp/center_region.jpg"
    visualize_center_region(img1_path, center_ratio=0.5, output_path=vis_path)
    print(f"Center region visualization: {vis_path}")
    
    print()
    print(">>> 다음 명령으로 정합 검증:")
    print(f"python util/manual_align_gui.py --ref {img1_path} --mov {aligned_path}")
