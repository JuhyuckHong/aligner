"""
[ARCHIVED]
Reason: Test script for SIFT+RANSAC. Effective but slower than Gradient PC.
Date: 2026-01-31
"""

"""
SIFT Feature Matching - 조명 변화에 가장 강건
"""
import cv2
import numpy as np
import sys
import os

def sift_match_offset(ref_path, mov_path):
    """
    SIFT + RANSAC으로 translation 추정
    """
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    mov = cv2.imread(mov_path, cv2.IMREAD_GRAYSCALE)
    
    # Histogram equalization for better matching
    ref = cv2.equalizeHist(ref)
    mov = cv2.equalizeHist(mov)
    
    # SIFT detector
    sift = cv2.SIFT_create(nfeatures=2000)
    
    kp1, des1 = sift.detectAndCompute(ref, None)
    kp2, des2 = sift.detectAndCompute(mov, None)
    
    print(f"  Keypoints: ref={len(kp1)}, mov={len(kp2)}")
    
    if des1 is None or des2 is None:
        return 0, 0, 0, False
    
    # FLANN matcher (faster for SIFT)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    print(f"  Good matches: {len(good_matches)}")
    
    if len(good_matches) < 10:
        return 0, 0, len(good_matches), False
    
    # Get matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # RANSAC to find homography/affine
    # For translation only, use estimateAffinePartial2D
    M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    
    if M is None:
        return 0, 0, len(good_matches), False
    
    # Extract translation
    dx = M[0, 2]
    dy = M[1, 2]
    scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
    angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
    
    n_inliers = np.sum(inliers) if inliers is not None else 0
    
    print(f"  Inliers: {n_inliers}")
    print(f"  Scale: {scale:.4f}, Angle: {angle:.3f}°")
    
    return dx, dy, n_inliers, True

def apply_and_save(mov_path, dx, dy, output_path):
    img = cv2.imread(mov_path)
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(output_path, aligned, [cv2.IMWRITE_JPEG_QUALITY, 98])

if __name__ == "__main__":
    ref_path = "input/2026-01-02/2026-01-02_18-00-00.jpg"
    mov_path = "input/2026-01-03/2026-01-03_06-00-00.jpg"
    
    if len(sys.argv) >= 3:
        ref_path = sys.argv[1]
        mov_path = sys.argv[2]
    
    print(f"=== SIFT Feature Matching ===")
    print(f"Ref: {os.path.basename(ref_path)}")
    print(f"Mov: {os.path.basename(mov_path)}")
    print()
    
    dx, dy, n_inliers, success = sift_match_offset(ref_path, mov_path)
    
    print()
    print(f"=== Result ===")
    print(f"Offset: dx={dx:.3f}, dy={dy:.3f}")
    
    if success:
        os.makedirs("temp", exist_ok=True)
        apply_and_save(mov_path, dx, dy, "temp/sift_aligned.jpg")
        print()
        print(f">>> 검증: python util/manual_align_gui.py --ref {ref_path} --mov temp/sift_aligned.jpg")
