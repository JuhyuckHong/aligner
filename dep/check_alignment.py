"""
[ARCHIVED]
Reason: Old prototype for alignment checking
Date: 2026-01-31
"""

"""
=============================================================================
util/check_alignment.py - 정합 결과 검증 스크립트 (개발용)
=============================================================================

특정 이미지 쌍의 Phase Correlation 결과를 확인하는 디버깅 도구.
개발 중 알고리즘 검증에 사용.
=============================================================================
"""
import cv2
import numpy as np
import os

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

def check_pair(ref_name, mov_name, folder="2026-01-01"):
    base_path = os.path.join("input", folder)
    ref_path = os.path.join(base_path, ref_name)
    mov_path = os.path.join(base_path, mov_name)
    
    print(f"Ref: {ref_name}")
    print(f"Mov: {mov_name}")
    
    ref = cv2.imread(ref_path)
    mov = cv2.imread(mov_path)
    
    if ref is None or mov is None:
        print("Failed to load images.")
        return

    # Edge creation
    ref_edge = create_edge(ref)
    mov_edge = create_edge(mov)
    
    # Downscale for Phase Correlation (same as main script)
    scale = 0.5
    ref_small = cv2.resize(ref_edge, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mov_small = cv2.resize(mov_edge, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Calculate
    dx, dy, response = phase_correlation(ref_small, mov_small)
    
    # Rescale dx, dy
    dx /= scale
    dy /= scale
    
    print(f"\n[Result]")
    print(f"dx: {dx:.2f}")
    print(f"dy: {dy:.2f}")
    print(f"response: {response:.5f}")
    print("-" * 30)

def get_global_ref_from_log():
    log_path = "output/outliers.log"
    if not os.path.exists(log_path):
        print("Log file not found.")
        return None
        
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Global Reference:" in line:
                # Format: # Global Reference: input/folder/file.jpg
                parts = line.split("Global Reference:")
                if len(parts) > 1:
                    ref_path = parts[1].strip()
                    # Ensure path is correct (relative to current script execution)
                    if not os.path.exists(ref_path):
                        # Try prepending 'input/' if missing or adjusting
                        if os.path.exists(os.path.join("input", os.path.basename(os.path.dirname(ref_path)), os.path.basename(ref_path))):
                             ref_path = os.path.join("input", os.path.basename(os.path.dirname(ref_path)), os.path.basename(ref_path))
                    return ref_path
    return None

def check_alignment_with_log_ref():
    ref_path = get_global_ref_from_log()
    if not ref_path or not os.path.exists(ref_path):
        print(f"Could not find valid Global Reference from log: {ref_path}")
        # Fallback manual ref if needed
        # ref_path = "input/2026-01-20/2026-01-20_13-00-00.jpg" 
        return

    print(f"Global Reference: {ref_path}")
    
    targets = [
        ("2026-01-01", "2026-01-01_06-12-00.jpg"),
        ("2026-01-01", "2026-01-01_06-18-00.jpg")
    ]
    
    ref_img = cv2.imread(ref_path)
    ref_edge = create_edge(ref_img)
    scale = 0.5
    ref_small = cv2.resize(ref_edge, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    for folder, filename in targets:
        target_path = os.path.join("input", folder, filename)
        if not os.path.exists(target_path):
            print(f"Target not found: {target_path}")
            continue
            
        mov_img = cv2.imread(target_path)
        mov_edge = create_edge(mov_img)
        mov_small = cv2.resize(mov_edge, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        dx, dy, response = phase_correlation(ref_small, mov_small)
        dx /= scale
        dy /= scale
        
        print(f"\n[Target: {filename}]")
        print(f"dx: {dx:.2f}, dy: {dy:.2f}")
        print(f"response: {response:.5f}")
        
        # Apply alignment and save to temp
        h, w = mov_img.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(mov_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        output_dir = "temp/check_alignment"
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"aligned_{filename}")
        cv2.imwrite(out_path, aligned)
        print(f"Saved aligned image: {out_path}")

if __name__ == "__main__":
    check_alignment_with_log_ref()
