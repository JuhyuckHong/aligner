"""
[ARCHIVED]
Reason: Debug script for LightGlue environment logic.
Date: 2026-01-31
"""

import sys
import os
import cv2
import numpy as np
import time

# Force line buffering
sys.stdout.reconfigure(line_buffering=True)

print("1. Importing PyTorch...")
try:
    import torch
    print(f"   Success. Version: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"   Failed to import torch: {e}")
    sys.exit(1)
except OSError as e:
    print(f"   OS Error (DLL blocked?): {e}")
    sys.exit(1)

print("2. Importing Kornia...")
try:
    import kornia
    from kornia.feature import LightGlue, SuperPoint
    print(f"   Success. Version: {kornia.__version__}")
except ImportError as e:
    print(f"   Failed to import kornia: {e}")
    sys.exit(1)

print("3. Loading Models...")
try:
    device = torch.device("cpu")
    extractor = SuperPoint(max_num_keypoints=1024).to(device).eval()
    matcher = LightGlue(features='superpoint').to(device).eval()
    print("   Success. Models loaded.")
except Exception as e:
    print(f"   Failed to load models: {e}")
    sys.exit(1)

print("4. Running Inference Test...")
try:
    # Create dummy images
    img1 = np.zeros((480, 640), dtype=np.uint8)
    img2 = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(img1, (100, 100), 10, 255, -1)
    cv2.circle(img2, (110, 110), 10, 255, -1) # Shifted 10px

    t_img1 = kornia.image_to_tensor(img1, False).float() / 255.
    t_img2 = kornia.image_to_tensor(img2, False).float() / 255.
    
    start_t = time.time()
    with torch.no_grad():
        feats1 = extractor(t_img1)
        feats2 = extractor(t_img2)
        matches01 = matcher({"image0": feats1, "image1": feats2})
        kpts1 = feats1["keypoints"][0]
        kpts2 = feats2["keypoints"][0]
        matches = matches01["matches"][0]
        
    elapsed = (time.time() - start_t) * 1000
    print(f"   Success. Inference time: {elapsed:.2f}ms")
    print(f"   Matches found: {len(matches)}")
    
except Exception as e:
    print(f"   Failed during inference: {e}")
    sys.exit(1)

print("\n--- LightGlue Test Complete ---")
