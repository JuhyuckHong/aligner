"""
=============================================================================
util/grid_match.py - 격자 기반 밝기 유사 영역 매칭
=============================================================================

밤→아침 전환처럼 조도 차이가 큰 경우를 위한 특수 매칭 알고리즘.

알고리즘:
  1. 두 이미지를 NxN 격자로 분할
  2. 각 격자의 평균 밝기 계산
  3. 밝기 차이가 작은 격자 쌍만 선택
  4. 선택된 격자들에서 edge + phase correlation 수행
  5. 결과를 가중 평균 (response 기반)

사용 예:
  from util.grid_match import grid_based_offset
  dx, dy, confidence = grid_based_offset(night_img, morning_img)
=============================================================================
"""
import cv2
import numpy as np


def calculate_grid_brightness(img, grid_size=9):
    """
    이미지를 grid_size x grid_size 격자로 나누고 각 격자의 평균 밝기 반환
    
    Returns:
        brightness: (grid_size, grid_size) 배열, 각 격자의 평균 밝기 (0-255)
        cells: 각 격자의 (y_start, y_end, x_start, x_end) 좌표
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    h, w = gray.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    brightness = np.zeros((grid_size, grid_size), dtype=np.float32)
    cells = []
    
    for i in range(grid_size):
        row_cells = []
        for j in range(grid_size):
            y_start = i * cell_h
            y_end = (i + 1) * cell_h if i < grid_size - 1 else h
            x_start = j * cell_w
            x_end = (j + 1) * cell_w if j < grid_size - 1 else w
            
            cell = gray[y_start:y_end, x_start:x_end]
            brightness[i, j] = np.mean(cell)
            row_cells.append((y_start, y_end, x_start, x_end))
        cells.append(row_cells)
    
    return brightness, cells


def find_similar_brightness_cells(brightness1, brightness2, threshold=30, min_brightness=20):
    """
    두 이미지의 밝기 격자에서 유사한 밝기를 가진 셀 쌍 찾기
    
    Args:
        brightness1: 첫 번째 이미지의 격자별 밝기
        brightness2: 두 번째 이미지의 격자별 밝기
        threshold: 밝기 차이 허용 임계값
        min_brightness: 최소 밝기 (너무 어두운 영역 제외)
    
    Returns:
        valid_cells: [(i, j), ...] 유사한 밝기를 가진 격자 인덱스 리스트
    """
    grid_size = brightness1.shape[0]
    valid_cells = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            b1 = brightness1[i, j]
            b2 = brightness2[i, j]
            
            # 밝기 차이가 threshold 이하이고, 둘 다 최소 밝기 이상
            if abs(b1 - b2) <= threshold and b1 >= min_brightness and b2 >= min_brightness:
                valid_cells.append((i, j))
    
    return valid_cells


def create_edge(img):
    """Canny edge 추출"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    return cv2.Canny(blurred, 50, 150)


def phase_correlation_cell(cell1, cell2):
    """단일 격자에 대한 phase correlation"""
    f1 = np.float32(cell1)
    f2 = np.float32(cell2)
    
    h, w = f1.shape
    if h < 10 or w < 10:
        return 0, 0, 0
    
    hann = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    
    shift, response = cv2.phaseCorrelate(f1 * hann, f2 * hann)
    return -shift[0], -shift[1], response


def grid_based_offset(img1, img2, grid_size=9, brightness_threshold=30, 
                       min_brightness=20, min_valid_cells=5, scale=0.5):
    """
    격자 기반 밝기 유사 영역 매칭으로 offset 계산
    
    Args:
        img1: 첫 번째 이미지 (예: 전날 밤)
        img2: 두 번째 이미지 (예: 다음날 아침)
        grid_size: 격자 크기 (9 = 9x9 = 81개 격자)
        brightness_threshold: 밝기 차이 허용 임계값
        min_brightness: 최소 밝기 (너무 어두운 셀 제외)
        min_valid_cells: 최소 유효 셀 개수
        scale: 처리 스케일
    
    Returns:
        dx, dy: 오프셋
        confidence: 신뢰도 (0-1, 유효 셀 비율)
        n_valid: 유효 셀 개수
    """
    # 스케일 조정
    if scale != 1.0:
        img1 = cv2.resize(img1, None, fx=scale, fy=scale)
        img2 = cv2.resize(img2, None, fx=scale, fy=scale)
    
    # 격자별 밝기 계산
    brightness1, cells1 = calculate_grid_brightness(img1, grid_size)
    brightness2, cells2 = calculate_grid_brightness(img2, grid_size)
    
    # 유사한 밝기 격자 찾기
    valid_cells = find_similar_brightness_cells(
        brightness1, brightness2, brightness_threshold, min_brightness
    )
    
    if len(valid_cells) < min_valid_cells:
        return 0, 0, 0, 0
    
    # Edge 변환
    edge1 = create_edge(img1)
    edge2 = create_edge(img2)
    
    # 유효 격자들에서 phase correlation 수행
    dx_list = []
    dy_list = []
    resp_list = []
    
    for (i, j) in valid_cells:
        y1, y2, x1, x2 = cells1[i][j]
        
        cell_edge1 = edge1[y1:y2, x1:x2]
        cell_edge2 = edge2[y1:y2, x1:x2]
        
        dx, dy, resp = phase_correlation_cell(cell_edge1, cell_edge2)
        
        if resp > 0.05:  # 최소 response threshold
            dx_list.append(dx)
            dy_list.append(dy)
            resp_list.append(resp)
    
    if not dx_list:
        return 0, 0, 0, 0
    
    # Response 가중 평균 (또는 중앙값)
    # 방법 1: 가중 평균
    # total_resp = sum(resp_list)
    # final_dx = sum(d * r for d, r in zip(dx_list, resp_list)) / total_resp
    # final_dy = sum(d * r for d, r in zip(dy_list, resp_list)) / total_resp
    
    # 방법 2: 중앙값 (더 robust)
    final_dx = np.median(dx_list) / scale
    final_dy = np.median(dy_list) / scale
    
    confidence = len(dx_list) / (grid_size * grid_size)
    
    return final_dx, final_dy, confidence, len(dx_list)


def visualize_grid_match(img1, img2, grid_size=9, brightness_threshold=30, 
                          min_brightness=20, output_path=None):
    """
    격자 매칭 결과 시각화 (디버깅용)
    
    유효 격자는 녹색, 무효 격자는 빨간색으로 표시
    """
    brightness1, cells1 = calculate_grid_brightness(img1, grid_size)
    brightness2, cells2 = calculate_grid_brightness(img2, grid_size)
    
    valid_cells = find_similar_brightness_cells(
        brightness1, brightness2, brightness_threshold, min_brightness
    )
    valid_set = set(valid_cells)
    
    # 시각화 이미지 생성
    vis1 = img1.copy()
    vis2 = img2.copy()
    
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2, x1, x2 = cells1[i][j]
            
            if (i, j) in valid_set:
                color = (0, 255, 0)  # 녹색: 유효
                thickness = 2
            else:
                color = (0, 0, 255)  # 빨간색: 무효
                thickness = 1
            
            cv2.rectangle(vis1, (x1, y1), (x2, y2), color, thickness)
            cv2.rectangle(vis2, (x1, y1), (x2, y2), color, thickness)
            
            # 밝기 표시
            cv2.putText(vis1, f"{brightness1[i,j]:.0f}", (x1+5, y1+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(vis2, f"{brightness2[i,j]:.0f}", (x1+5, y1+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # 가로로 합치기
    combined = np.hstack([vis1, vis2])
    
    # 정보 추가
    info_text = f"Valid cells: {len(valid_cells)}/{grid_size*grid_size} ({100*len(valid_cells)/(grid_size*grid_size):.1f}%)"
    cv2.putText(combined, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, combined)
    
    return combined, valid_cells


# === 테스트 코드 ===
if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) < 3:
        print("Usage: python grid_match.py <night_image> <morning_image>")
        print("Example: python grid_match.py output/2026-01-01/235900.jpg output/2026-01-02/060000.jpg")
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("Error: Image file not found")
        sys.exit(1)
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    print()
    
    # 격자 매칭 수행
    dx, dy, confidence, n_valid = grid_based_offset(img1, img2, grid_size=9)
    
    print(f"=== Grid-based Matching Result ===")
    print(f"Offset: dx={dx:.2f}, dy={dy:.2f}")
    print(f"Valid cells: {n_valid}/81")
    print(f"Confidence: {confidence:.2%}")
    
    # 시각화
    vis_path = "grid_match_result.jpg"
    visualize_grid_match(img1, img2, output_path=vis_path)
    print(f"\nVisualization saved: {vis_path}")
