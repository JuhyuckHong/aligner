"""
=============================================================================
timelapse_stabilizer.py - High-performance Parallel Timelapse Stabilizer
=============================================================================

Features:
  - 🚀 Parallel Processing: Analysis, Refinement, and Rendering
  - 💾 Modular Logging: JSON-based intermediate logs for easy resuming
  - 🎯 Global Anchor Refinement: Aligns all days to Day 1 to eliminate drift accumulation
  - 🎮 PID Control Smoothing: Reduces fluctuation and drift using PID controller
  - 🌟 Full-Day Correction: Spreads correction over the entire day for zero wobbling
  - 🔄 Two-Pass Architecture: Separates motion estimation from image warping

Usage:
  python timelapse_stabilizer.py --video --workers 8
  python timelapse_stabilizer.py --video --render-only (uses existing logs)
  python timelapse_stabilizer.py --force-analyze (re-run analysis)
  python timelapse_stabilizer.py --video --resize-width 1920 (Create 1080p video)

Author: Antigravity (Google Deepmind)
"""

import cv2
import numpy as np
import os
import argparse
import shutil
import json
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import subprocess
from datetime import datetime
import create_video

# === Configuration ===
ROTATION_THRESHOLD_DEG = 0.02
DAMPING_DEADZONE = 3.0
DAMPING_FACTOR = 1.0

# Brightness/Darkness Thresholds
DARK_THRESHOLD = 60.0   # Images with mean brightness below this are ignored
TARGET_BRIGHTNESS = 110.0 # Target mean brightness for normalization

# Default Options
DEFAULT_OPTIONS = {
    "dark_threshold": DARK_THRESHOLD,
    "remove_dark": True,
    "normalize_brightness": True,
    "target_brightness": TARGET_BRIGHTNESS,
    "verbose_errors": False,
}

# Refinement Precision
DAY_REFINE_SAMPLES = 5 
ECC_ITERATIONS = 100  # Optimized for speed
ECC_EPS = 1e-4

# PID Control Parameters (Aggressive tuning for zero drift)
PID_KP = 0.8  # Strong proportional gain
PID_KI = 0.2  # Significant integral gain to eliminate offset
PID_KD = 0.2  # Moderate derivative gain

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        
    def update(self, current, target):
        error = target - current
        self.integral += error
        derivative = error - self.prev_error
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.prev_error = error
        return current + output # Return new position

def resolve_options(options=None):
    resolved = DEFAULT_OPTIONS.copy()
    if options:
        for k, v in options.items():
            if v is not None:
                resolved[k] = v
    return resolved

def get_images(input_dir, ext='jpg'):
    patterns = [f"*.{ext}", f"*.{ext.upper()}"]
    images = []
    for pattern in patterns:
        images.extend(glob(os.path.join(input_dir, pattern)))
    return sorted(list(set(images)))

def create_gradient(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def phase_correlation(ref, mov):
    f1 = np.float32(ref)
    f2 = np.float32(mov)
    h, w = f1.shape
    hann = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    shift, response = cv2.phaseCorrelate(f1 * hann, f2 * hann)
    return -shift[0], -shift[1], response

def get_euclidean_transform(ref_gray, mov_gray):
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERATIONS, ECC_EPS)
    try:
        (cc, warp_matrix) = cv2.findTransformECC(ref_gray, mov_gray, warp_matrix, warp_mode, criteria)
        angle = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0]) * 180 / np.pi
        dx = warp_matrix[0, 2]
        dy = warp_matrix[1, 2]
        return angle, dx, dy, True
    except:
        return 0, 0, 0, False
        return 0, 0, 0, False

def is_image_dark(img, threshold=DARK_THRESHOLD):
    """
    Checks if an image is too dark (night time).
    """
    if img is None: return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Crop center 50%
    h, w = gray.shape
    h_start, h_end = int(h * 0.25), int(h * 0.75)
    w_start, w_end = int(w * 0.25), int(w * 0.75)
    
    center_crop = gray[h_start:h_end, w_start:w_end]
    mean_brightness = np.mean(center_crop)
    return mean_brightness, (mean_brightness < threshold)

def smooth_array(data, window_size=15):
    """Simple moving average smoothing with edge padding."""
    if not data or len(data) < window_size:
        return data
    
    # Convert to float array
    arr = np.array(data, dtype=np.float32)
    pad_size = window_size // 2
    
    # Pad with edge values
    padded = np.pad(arr, (pad_size, pad_size), mode='edge')
    
    # Convolve
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded, kernel, mode='valid')
    
    # Handle length mismatch due to padding/valid mode quirks if any
    if len(smoothed) > len(data):
        smoothed = smoothed[:len(data)]
    elif len(smoothed) < len(data):
        # Fallback (should not happen with correct padding)
        smoothed = np.pad(smoothed, (0, len(data) - len(smoothed)), mode='edge')
        
    return smoothed

def normalize_brightness(img, target=TARGET_BRIGHTNESS):
    """
    Normalizes the brightness of an image to match the target mean.
    Uses HSV Value channel scaling.
    """
    if img is None: return img
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    current_mean = np.mean(v)
    if current_mean < 10: # Too dark to normalize properly to target
        return img
        
    scale = target / current_mean
    # Clamp scale to prevent quality degradation (noise amplification or excessive darkening)
    scale = np.clip(scale, 0.7, 1.5) 
    
    # Apply scale and clip to valid range
    v = cv2.multiply(v, scale)
    v = np.clip(v, 0, 255).astype(np.uint8)
    
    hsv_norm = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_norm, cv2.COLOR_HSV2BGR)

def apply_brightness_scale(img, scale):
    """
    Multiplies the V channel by a specific scale factor.
    """
    if img is None or scale == 1.0: return img
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    v = cv2.multiply(v, scale)
    v = np.clip(v, 0, 255).astype(np.uint8)
    
    hsv_norm = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_norm, cv2.COLOR_HSV2BGR)



# ---------------------------------------------------------------------------
# UTILS: Brightness Calculation
# ---------------------------------------------------------------------------
def calculate_brightness_scales_for_folder(frames, options):
    """
    Calculates brightness scaling factors for a list of frames (from one folder)
    using temporal smoothing.
    Returns: dict {filename: scale}
    """
    options = resolve_options(options)
    target_b = options.get('target_brightness', TARGET_BRIGHTNESS)
    filtered_frames = [f for f in frames if f.get('status') != "DARK"]
    
    scales_map = {}
    
    if not filtered_frames:
        return scales_map
        
    if not options.get('normalize_brightness', True):
        for f in filtered_frames:
            scales_map[f['filename']] = 1.0
        return scales_map

    # Extract raw brightness
    raw_b = [f.get('brightness', target_b) for f in filtered_frames]
    
    # Smooth
    def smooth_array_local(arr, window_size=15):
        if len(arr) < window_size:
            return arr
        box = np.ones(window_size)/window_size
        return np.convolve(arr, box, mode='same')

    smoothed_b = smooth_array_local(raw_b, window_size=15)
    
    for i, f in enumerate(filtered_frames):
        rb = raw_b[i]
        
        # Calculate Scale: Target / Raw (Clamped)
        # We assume we want to reach Target Brightness.
        
        # To avoid noise amplification in very dark images:
        denominator = max(rb, 10.0) 
        scale = target_b / denominator
        
        # Clamp scale (e.g. 0.5x to 3.0x)
        scale = np.clip(scale, 0.5, 3.0)
        
        scales_map[f['filename']] = float(scale)
        
    return scales_map

def normalize_images_worker(args):
    """
    Worker to apply brightness normalization and save to new folder.
    args: (folder_name, image_list, analysis_results, output_root, options, progress_queue)
    """
    folder_name, image_list, analysis_results, output_root, options, q = args
    options = resolve_options(options)
    
    # Calculate scales
    scales_map = calculate_brightness_scales_for_folder(analysis_results, options)
    
    # Output Dir (step1_normalized)
    step1_5_dir = os.path.join(output_root, "step1_normalized", os.path.basename(folder_name))
    if not os.path.exists(step1_5_dir):
        os.makedirs(step1_5_dir)
        
    cnt = 0
    for img_path in image_list:
        fname = os.path.basename(img_path)
        
        # Skip dark/excluded images if they are not in map
        if fname not in scales_map:
            continue
            
        scale = scales_map[fname]
        
        # Read
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Process (apply_brightness_scale helper or manual?)
        # Use existing helper if available or inline
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        v_new = cv2.multiply(v.astype(float), scale)
        v_new = np.clip(v_new, 0, 255).astype(np.uint8)
        
        hsv_new = cv2.merge([h, s, v_new])
        img_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
        
        # Save
        out_path = os.path.join(step1_5_dir, fname)
        cv2.imwrite(out_path, img_new)
        
        cnt += 1
        if q and cnt % 5 == 0:
             import multiprocessing
             current_proc = multiprocessing.current_process().name
             msg = f"Norm {cnt}/{len(image_list)}"
             q.put(('WORKER_PROGRESS', current_proc, cnt/len(image_list)*100, msg))
             # Legacy
             q.put(('P_INC', 5))
             
    return cnt


# ---------------------------------------------------------------------------
# PHASE 0: Pre-processing (Worker)
# ---------------------------------------------------------------------------
def scan_dark_images_worker(args):
    """
    Scans a list of images and returns analysis for ALL files.
    args: (folder_name, image_list, dark_threshold, progress_queue)
    Returns: list of (img_path, brightness, is_dark)
    """
    folder_name, image_list, threshold, progress_queue = args[:4]
    verbose_errors = args[4] if len(args) > 4 else False
    results = [] # (path, brightness, is_dark)
    error_count = 0
    
    import multiprocessing
    current_proc = multiprocessing.current_process().name
    
    total = len(image_list)
    for i, img_path in enumerate(image_list):
        if progress_queue:
            if i % 10 == 0:
                # Progress Update: (WORKER_PROGRESS, proc_name, current/total, msg)
                msg = f"Scanning {i}/{total}"
                progress_queue.put(('WORKER_PROGRESS', current_proc, i/total*100, msg))
            if i % 20 == 0:
                progress_queue.put(('P_INC', 20)) # Keep global progress roughly

        # Fast check
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            b_val, is_dark = is_image_dark(img, threshold)
            results.append((img_path, float(b_val), is_dark))
        except Exception as e:
            error_count += 1
            if verbose_errors and error_count <= 3:
                print(f"[Scan][{folder_name}] Error reading {os.path.basename(img_path)}: {e}")
            continue
            
    if progress_queue:
        progress_queue.put(('WORKER_PROGRESS', current_proc, 100, "Done"))
    if error_count and verbose_errors:
        print(f"[Scan][{folder_name}] Completed with {error_count} errors.")
        
    return results


# ---------------------------------------------------------------------------
# PHASE 1: Analysis (Worker)
# ---------------------------------------------------------------------------
def analyze_folder_worker(args):
    # args can be:
    # (input_dir, ext)
    # (input_dir, image_list)
    # (input_dir, image_list, progress_queue)
    
    input_dir = args[0]
    second_arg = args[1]
    progress_queue = None
    options = {}
    
    if len(args) >= 3:
        progress_queue = args[2]
    if len(args) >= 4:
        options = args[3]
    options = resolve_options(options)
    
    if isinstance(second_arg, str):
        ext = second_arg
        images = get_images(input_dir, ext)
    else:
        images = second_arg # Explicit list
        
    if not images:
        return (os.path.basename(input_dir), [])

    # Options
    dark_thresh = options.get('dark_threshold', DARK_THRESHOLD)
    remove_dark = options.get('remove_dark', True) # Default to True if not specified
    
    results = []
    scale = 0.5
    
    prev_img = cv2.imread(images[0])
    if prev_img is None:
        return (os.path.basename(input_dir), [])

    folder_name = os.path.basename(input_dir)
            
    # Use Gradient for robustness as per README
    # Resize first to speed up
    prev_small = cv2.resize(prev_img, None, fx=scale, fy=scale)
    prev_grad = create_gradient(prev_small)
    h, w = prev_grad.shape
    
    # Init first result
    results.append({
        "filename": os.path.basename(images[0]),
        "abs_path": images[0],
        "dx": 0.0, "dy": 0.0, "rot": 0.0,
        "status": "OK",
        "brightness": float(np.mean(create_gradient(prev_small))) * 0 + float(is_image_dark(prev_img)[0]) # Hack to get brightness
    })
    
    # ECC Criteria from README (500 iterations)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-4)
    
    import multiprocessing
    current_proc = multiprocessing.current_process().name
    total_imgs = len(images)
    
    for i in range(1, total_imgs):
        curr_path = images[i]
        curr_fname = os.path.basename(curr_path)
        
        # Report Progress
        if progress_queue:
            progress_queue.put(('P_INC', 1))
            if i % 5 == 0:
                # Granular Update
                msg = f"{folder_name[:10]}.. {i}/{total_imgs}"
                progress_queue.put(('WORKER_PROGRESS', current_proc, i/total_imgs*100, msg))
                # Legacy global update
                progress_queue.put(f"[Analyzing] {folder_name}: {i}/{total_imgs}")
            
        curr_img = cv2.imread(curr_path)
        if curr_img is None:
            results.append({
                "filename": curr_fname, "abs_path": curr_path,
                "dx": 0.0, "dy": 0.0, "rot": 0.0, "status": "FAIL_READ"
            })
            continue

        # Check for Dark Image
        b_val, is_dark = is_image_dark(curr_img, threshold=dark_thresh)
        
        if is_dark:
            status = "DARK"
            # Even if we don't remove it here (we might do it in Phase 3),
            # we should mark it. The logic in Phase 3 handles "DARK" status.
            # But the user might want to NOT remove it.
            # If ensure 'remove_dark' is False, we treat it as normal?
            # Actually, current logic:
            # - Analysis just marks it as DARK or brightness.
            # - Integration decides whether to skip.
            # So here we just mark it.
            
            results.append({
                "filename": curr_fname, "abs_path": curr_path,
                "dx": 0.0, "dy": 0.0, "rot": 0.0, "status": "DARK",
                "brightness": float(b_val)
            })
            # Skip updating prev_grad, as this frame is invalid.
            continue
            
        curr_small = cv2.resize(curr_img, None, fx=scale, fy=scale)
        curr_grad = create_gradient(curr_small)
        
        # --- Algorithm per README ---
        # 1. Gradient (Already done: prev_grad, curr_grad)
        # 2. ECC Rotation Detection
        
        d_rot = 0.0
        d_dx = 0.0
        d_dy = 0.0
        status = "OK"
        
        try:
            # Init warp with identity
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            
            # Run ECC (Euclidean: Rot + Trans)
            # README says "ECC 정밀 회전 감지"
            (_, warp_matrix) = cv2.findTransformECC(prev_grad, curr_grad, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
            
            # Extract Rotation
            rot_rad = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])
            d_rot = np.degrees(rot_rad)
            
            # 3. Derotate (Inverse Rotation) for Phase Correlation
            center = (w // 2, h // 2)
            M_derot = cv2.getRotationMatrix2D(center, d_rot, 1.0)
            curr_grad_derot = cv2.warpAffine(curr_grad, M_derot, (w, h))
            
            # 4. Phase Correlation for Translation
            # Use derotated gradient vs previous gradient
            prev_hann = (prev_grad.astype(np.float32) * np.hanning(w) * np.hanning(h)[:, None])
            curr_hann = (curr_grad_derot.astype(np.float32) * np.hanning(w) * np.hanning(h)[:, None])
            
            shift, _ = cv2.phaseCorrelate(prev_hann, curr_hann)
            p_dx, p_dy = shift
            
            # Final dx/dy is a combination of ECC's rough trans (if we used it) or purely PhaseCorr.
            # README diagram implies: [ECC (Rot)] -> [Derotate] -> [PhaseCorr (Trans)]
            # So we use PhaseCorr result as the translation.
            # Note: PhaseCorrelation gives translation of 'curr' relative to 'prev'.
            # Result is (dx, dy).
            d_dx = p_dx
            d_dy = p_dy
            
            if abs(d_rot) > 0.1:
                status = f"ROT({d_rot:.2f})"
                
        except Exception as e:
            # Fallback if ECC fails: just PhaseCorr on raw gradients
            prev_hann = (prev_grad.astype(np.float32) * np.hanning(w) * np.hanning(h)[:, None])
            curr_hann = (curr_grad.astype(np.float32) * np.hanning(w) * np.hanning(h)[:, None])
            shift, _ = cv2.phaseCorrelate(prev_hann, curr_hann)
            d_dx, d_dy = shift
            status = "ECC_FAIL"

        # Scale back to original resolution
        final_dx = d_dx / scale
        final_dy = d_dy / scale
        # Rotation is invariant to scale

        results.append({
            "filename": curr_fname,
            "abs_path": curr_path,
            "dx": float(final_dx),
            "dy": float(final_dy),
            "rot": float(d_rot),
            "status": status,
            "brightness": float(b_val)
        })
        
        prev_grad = curr_grad
        
    return (os.path.basename(input_dir), results)

# ---------------------------------------------------------------------------
# PHASE 2: Refinement (Worker & Helpers)
# ---------------------------------------------------------------------------
def get_noon_samples_with_acc(analysis_list, n_samples=DAY_REFINE_SAMPLES):
    """
    Select noon samples and COMPUTE ACCUMULATED MOTION for them.
    Returns: list of dict {abs_path, acc_dx, acc_dy, acc_rot}
    """
    # Pre-compute accumulation for the whole list
    acc_dx, acc_dy, acc_rot = 0.0, 0.0, 0.0
    accumulated_data = []
    
    for item in analysis_list:
        acc_dx += item['dx']
        acc_dy += item['dy']
        acc_rot += item['rot']
        
        try:
            basename = item['filename']
            time_part = basename.split('_')[1].split('.')[0]
            h, m, s = map(int, time_part.split('-'))
            diff = abs((h*60+m) - 720) # Diff from 12:00
            
            accumulated_data.append({
                'diff': diff,
                'abs_path': item['abs_path'],
                'acc_dx': acc_dx,
                'acc_dy': acc_dy,
                'acc_rot': acc_rot
            })
        except: continue
        
    accumulated_data.sort(key=lambda x: x['diff'])
    return accumulated_data[:n_samples]

def measure_day_gap_worker(args):
    """
    Compare Anchor Day (Day 1) and Target Day (Day N) using Virtual Alignment.
    """
    if len(args) < 4:
        raise ValueError("measure_day_gap_worker expects at least 4 args")
    anchor_day, target_day, s1_list, s2_list = args[:4]
    q = args[4] if len(args) > 4 else None
    print(f"[Gap] Analyzing {os.path.basename(anchor_day)} -> {os.path.basename(target_day)}")
    
    if not s1_list or not s2_list:
        return (target_day, (0.0, 0.0, 0.0))
    
    if anchor_day == target_day:
        return (target_day, (0.0, 0.0, 0.0))
        
    gaps_x = []
    gaps_y = []
    gaps_rot = []
    scale = 0.25
    
    import multiprocessing
    current_proc = multiprocessing.current_process().name
    
    # Calculate total iterations roughly for progress bar
    total_pairs = len(s1_list) * len(s2_list)
    pair_count = 0
    
    for samp1 in s1_list:
        if not os.path.exists(samp1['abs_path']): continue
        img1_full = cv2.imread(samp1['abs_path'])
        if img1_full is None: continue
        
        # Warp Anchor (Day 1) to its base
        img1_small = cv2.resize(img1_full, None, fx=scale, fy=scale)
        center_small = (img1_small.shape[1]//2, img1_small.shape[0]//2)
        
        M1_rot = cv2.getRotationMatrix2D(center_small, -samp1['acc_rot'], 1.0)
        M1_rot[0, 2] -= samp1['acc_dx'] * scale
        M1_rot[1, 2] -= samp1['acc_dy'] * scale
        warped1 = cv2.warpAffine(img1_small, M1_rot, (img1_small.shape[1], img1_small.shape[0]))
        gray1 = cv2.cvtColor(warped1, cv2.COLOR_BGR2GRAY)
        
        for samp2 in s2_list:
            pair_count += 1
            if not os.path.exists(samp2['abs_path']): continue
            img2_full = cv2.imread(samp2['abs_path'])
            if img2_full is None: continue
            
            # Progress Log
            msg = f"Gap {pair_count}/{total_pairs}: {os.path.basename(samp1['abs_path'])} vs {os.path.basename(samp2['abs_path'])}"
            print(f"  [Gap] {msg}") # Keep print for terminal debugging
            
            if q:
                q.put(('WORKER_PROGRESS', current_proc, (pair_count/total_pairs)*100, msg))
            
            img2_small = cv2.resize(img2_full, None, fx=scale, fy=scale)
            
            # Warp Target (Day N) to its base
            M2_rot = cv2.getRotationMatrix2D(center_small, -samp2['acc_rot'], 1.0)
            M2_rot[0, 2] -= samp2['acc_dx'] * scale
            M2_rot[1, 2] -= samp2['acc_dy'] * scale
            warped2 = cv2.warpAffine(img2_small, M2_rot, (img2_small.shape[1], img2_small.shape[0]))
            gray2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY)
            
            # 1. ECC for Rotation
            angle, _, _, ecc_ok = get_euclidean_transform(gray1, gray2)
            
            if ecc_ok:
                gaps_rot.append(angle)
                M_derot = cv2.getRotationMatrix2D(center_small, angle, 1.0)
                gray2_derot = cv2.warpAffine(gray2, M_derot, (gray2.shape[1], gray2.shape[0]))
                
                gx = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
                mag1 = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                gx2 = cv2.Sobel(gray2_derot, cv2.CV_32F, 1, 0, ksize=3)
                gy2 = cv2.Sobel(gray2_derot, cv2.CV_32F, 0, 1, ksize=3)
                mag2 = cv2.normalize(cv2.magnitude(gx2, gy2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                dx, dy, _ = phase_correlation(mag1, mag2)
                gaps_x.append(dx / scale)
                gaps_y.append(dy / scale)
            else:
                gaps_rot.append(0.0)
                grad1 = create_gradient(warped1)
                grad2 = create_gradient(warped2)
                dx, dy, _ = phase_correlation(grad1, grad2)
                gaps_x.append(dx / scale)
                gaps_y.append(dy / scale)

    if gaps_x:
        final_gap_dx = float(np.median(gaps_x))
        final_gap_dy = float(np.median(gaps_y))
        final_gap_rot = float(np.median(gaps_rot))
    else:
        final_gap_dx, final_gap_dy, final_gap_rot = 0.0, 0.0, 0.0
        
    print(f"[Gap] Result {os.path.basename(target_day)}: ({final_gap_dx:.2f}, {final_gap_dy:.2f})")
    return (target_day, (final_gap_dx, final_gap_dy, final_gap_rot))

# ---------------------------------------------------------------------------
# Day Gap Helpers
# ---------------------------------------------------------------------------
def parse_day_gap_entry(entry):
    """Return (dx, dy, rot) from tuple/list/dict day gap entries."""
    dx = dy = rot = 0.0
    if isinstance(entry, dict):
        try:
            dx = float(entry.get("dx", 0.0))
        except Exception:
            dx = 0.0
        try:
            dy = float(entry.get("dy", 0.0))
        except Exception:
            dy = 0.0
        try:
            rot = float(entry.get("rot", 0.0))
        except Exception:
            rot = 0.0
    elif isinstance(entry, (list, tuple)):
        if len(entry) > 0:
            try:
                dx = float(entry[0])
            except Exception:
                dx = 0.0
        if len(entry) > 1:
            try:
                dy = float(entry[1])
            except Exception:
                dy = 0.0
        if len(entry) > 2:
            try:
                rot = float(entry[2])
            except Exception:
                rot = 0.0
    return dx, dy, rot

# ---------------------------------------------------------------------------
# PHASE 3: Integration
# ---------------------------------------------------------------------------
def integrate_trajectory(folder_analyses, day_gaps_dict, options={}):
    options = resolve_options(options)
    sorted_folders = sorted(folder_analyses.keys())
    global_trajectory = {}
    
    print("\n[Integration] Applying PID Control & Full-Day Smoothing...")
    
    # Initialize PID Controllers
    pid_dx = PIDController(PID_KP, PID_KI, PID_KD)
    pid_dy = PIDController(PID_KP, PID_KI, PID_KD)
    pid_rot = PIDController(PID_KP, PID_KI, PID_KD)
    
    current_global_dx = 0.0
    current_global_dy = 0.0
    current_global_rot = 0.0
    
    # PID State
    smoothed_target_dx = 0.0
    smoothed_target_dy = 0.0
    smoothed_target_rot = 0.0
    
    for i, folder in enumerate(sorted_folders):
        frames = folder_analyses[folder]
        if not frames:
            global_trajectory[folder] = []
            continue
        
        # Transition from Previous Day
        if i > 0:
            prev_idx = i - 1
            while prev_idx >= 0 and not folder_analyses[sorted_folders[prev_idx]]:
                prev_idx -= 1
            if prev_idx >= 0:
                prev_folder = sorted_folders[prev_idx]
                prev_frames = folder_analyses[prev_folder]
                last_img = prev_frames[-1]['abs_path']
                first_img = frames[0]['abs_path']
                
                scale = 0.5
                prev_img = cv2.imread(last_img)
                curr_img = cv2.imread(first_img)
                if prev_img is not None and curr_img is not None:
                    prev_grad = cv2.resize(create_gradient(prev_img), None, fx=scale, fy=scale)
                    curr_grad = cv2.resize(create_gradient(curr_img), None, fx=scale, fy=scale)
                    
                    trans_dx, trans_dy, _ = phase_correlation(prev_grad, curr_grad)
                    trans_dx /= scale
                    trans_dy /= scale
                    
                    current_global_dx += trans_dx
                    current_global_dy += trans_dy
                    print(f"  Transition {prev_folder}->{folder}: ({trans_dx:.1f}, {trans_dy:.1f})")
            
        start_dx = current_global_dx
        start_dy = current_global_dy
        start_rot = current_global_rot
        
        # Raw Target from Anchor Refinement
        raw_target_dx, raw_target_dy, raw_target_rot = parse_day_gap_entry(
            day_gaps_dict.get(folder, (0.0, 0.0, 0.0))
        )
        
        # PID Update
        smoothed_target_dx = pid_dx.update(smoothed_target_dx, raw_target_dx)
        smoothed_target_dy = pid_dy.update(smoothed_target_dy, raw_target_dy)
        smoothed_target_rot = pid_rot.update(smoothed_target_rot, raw_target_rot)
        
        refine_gap_dx = smoothed_target_dx - start_dx
        refine_gap_dy = smoothed_target_dy - start_dy
        refine_gap_rot = smoothed_target_rot - start_rot
        
        # Full-Day Correction (Always spread over n_frames)
        n_frames = len(frames)
        trans_len = n_frames 
        
        print(f"  Day {folder}: PID_Trgt({smoothed_target_dx:.1f}, {smoothed_target_dy:.1f}) | Correcting Gap({refine_gap_dx:.1f}, {refine_gap_dy:.1f}) over {trans_len} frames")
        
        local_acc_dx = 0.0
        local_acc_dy = 0.0
        local_acc_rot = 0.0
        
        folder_trajectory = []
        
        folder_trajectory = []
        
        for idx, frame in enumerate(frames):
            # FILTER DARK IMAGES HERE
            if options.get('remove_dark', True) and frame['status'] == "DARK":
                continue
                
            local_acc_dx += frame['dx']
            local_acc_dy += frame['dy']
            local_acc_rot += frame['rot']
            
            # if abs(local_acc_dx) > DAMPING_DEADZONE: local_acc_dx *= DAMPING_FACTOR
            # if abs(local_acc_dy) > DAMPING_DEADZONE: local_acc_dy *= DAMPING_FACTOR
            
            # Linear Interpolation over the whole day
            alpha = idx / trans_len
            
            shift_dx = alpha * refine_gap_dx
            shift_dy = alpha * refine_gap_dy
            shift_rot = alpha * refine_gap_rot
            
            final_dx = start_dx + local_acc_dx + shift_dx
            final_dy = start_dy + local_acc_dy + shift_dy
            final_rot = start_rot + local_acc_rot + shift_rot
            
            folder_trajectory.append({
                "filename": frame['filename'],
                "abs_path": frame['abs_path'],
                "final_dx": final_dx,
                "final_dy": final_dy,
                "rot": final_rot, 
                "status": frame['status']
            })
            
        # --- Temporal Brightness Smoothing ---
        filtered_frames = [f for f in frames if f['status'] != "DARK"]
        
        if filtered_frames and options.get('normalize_brightness', True):
            target_b = options.get('target_brightness', 110.0)
            raw_b = [f.get('brightness', target_b) for f in filtered_frames]
            
            # Smooth
            smoothed_b = smooth_array(raw_b, window_size=15)
            
            # Calculate Scale for non-dark frames
            scale_map = {}
            for i, f in enumerate(filtered_frames):
                rb = raw_b[i]
                sb = smoothed_b[i]
                if rb > 1.0:
                    scale = sb / rb
                    # Clamp for safety
                    scale = np.clip(scale, 0.8, 1.25) # Gentle correction
                else:
                    scale = 1.0
                scale_map[f['filename']] = float(scale)
            
            # Apply scales (dark frames default to 1.0)
            for item in folder_trajectory:
                item['brightness_scale'] = scale_map.get(item['filename'], 1.0)
        else:
            # No normalization
            for item in folder_trajectory:
                item['brightness_scale'] = 1.0
                
        global_trajectory[folder] = folder_trajectory
        
        current_global_dx = start_dx + local_acc_dx + refine_gap_dx
        current_global_dy = start_dy + local_acc_dy + refine_gap_dy
        current_global_rot = start_rot + local_acc_rot + refine_gap_rot
        
    return global_trajectory

# ---------------------------------------------------------------------------
# PHASE 4: Rendering
# ---------------------------------------------------------------------------
def render_folder_worker(args):
    # args can be (input_dir, output_dir, trajectory) 
    # OR (input_dir, output_dir, trajectory, progress_queue)
    input_dir = args[0]
    output_dir = args[1]
    trajectory = args[2]
    progress_queue = None
    if len(args) >= 4:
        progress_queue = args[3]
    
    import multiprocessing
    current_proc = multiprocessing.current_process().name
        
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, item in enumerate(trajectory):
        if progress_queue:
            progress_queue.put(('P_INC', 1))
            if idx % 10 == 0:
                msg = f"Render {idx}/{len(trajectory)}"
                pct = (idx / max(1, len(trajectory))) * 100
                progress_queue.put(('WORKER_PROGRESS', current_proc, pct, msg))

        img = cv2.imread(item['abs_path'])
        if img is None: continue
            
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # analysis dx/dy represents motion of current frame; invert to apply correction
        dx = -item['final_dx']
        dy = -item['final_dy']
        rot = item['rot']
        
        M = cv2.getRotationMatrix2D(center, rot, 1.0) 
        M[0, 2] += dx
        M[1, 2] += dy
        
        aligned = cv2.warpAffine(img, M, (w, h), 
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Normalize Brightness (Temporal Smoothing)
        b_scale = item.get('brightness_scale', 1.0)
        aligned = apply_brightness_scale(aligned, b_scale)
        
        day_name = os.path.basename(output_dir)
        out_name = f"{day_name} {idx+1:04d}.jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), aligned, [cv2.IMWRITE_JPEG_QUALITY, 98])
        
        if idx % 50 == 0:
            if progress_queue:
                progress_queue.put(f"[Rendering] {os.path.basename(input_dir)}: {idx}/{len(trajectory)}")
            else:
                print(f"  [Render] {os.path.basename(input_dir)}: {idx}/{len(trajectory)} (rot={rot:.4f})", flush=True)
    
    return len(trajectory)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_stabilization_task(input_root, output_root, log_prefix, place_name, args):
    """
    Executes the stabilization pipeline for a single place/folder.
    """
    os.makedirs(output_root, exist_ok=True)
    
    analysis_log_path = os.path.join(output_root, f"{log_prefix}analysis_log.json")
    refine_log_path = os.path.join(output_root, f"{log_prefix}refine_log.json")
    full_log_path = os.path.join(output_root, f"{log_prefix}full_log.txt")
    
    # Folder Scanning & Filtering
    valid_folders = []
    if os.path.exists(input_root):
        all_sub = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])
        for d in all_sub:
            try:
                datetime.strptime(d, "%Y-%m-%d")
            except ValueError:
                continue
            if args.start and d < args.start: continue
            if args.end and d > args.end: continue
            valid_folders.append(d)
    
    if not valid_folders and not args.render_only:
        print(f"No valid date folders found in {input_root}")
        return

    if not args.render_only:
        print(f"Targeting {len(valid_folders)} folders: {valid_folders[0]} ~ {valid_folders[-1]}")

    # === Phase 1: Analysis ===
    folder_analyses = {}
    
    if args.render_only:
        pass 
    elif os.path.exists(analysis_log_path) and not args.force_analyze:
        print(f"Loading existing analysis from {analysis_log_path}...")
        with open(analysis_log_path, "r") as f:
            folder_analyses = json.load(f)
            
        # Analyze missing folders if any
        missing = [d for d in valid_folders if d not in folder_analyses]
        if missing:
             print(f"Analyzing {len(missing)} missing folders...")
             tasks = [(os.path.join(input_root, d), args.ext) for d in missing]
             with Pool(args.workers) as pool:
                for folder_name, results in tqdm(pool.imap(analyze_folder_worker, tasks), total=len(tasks)):
                    folder_analyses[folder_name] = results
             with open(analysis_log_path, "w") as f:
                json.dump(folder_analyses, f, indent=2)

    else:
        # Load existing if available to preserve history
        if os.path.exists(analysis_log_path):
             with open(analysis_log_path, "r") as f:
                try: folder_analyses = json.load(f)
                except: folder_analyses = {}
        
        tasks = [(os.path.join(input_root, d), args.ext) for d in valid_folders]
        
        print(f"\n[Phase 1] Analyzing motion with {args.workers} workers...")
        with Pool(args.workers) as pool:
            for folder_name, results in tqdm(pool.imap(analyze_folder_worker, tasks), total=len(tasks)):
                folder_analyses[folder_name] = results
                
        with open(analysis_log_path, "w") as f:
            json.dump(folder_analyses, f, indent=2)
            
    if args.render_only:
        pass
    else:
        # === Phase 2: Refinement (Global Anchor) ===
        day_gaps = {}
        if os.path.exists(refine_log_path) and not args.force_refine and not args.force_analyze:
             with open(refine_log_path, "r") as f:
                day_gaps = json.load(f)
        
        # Ensure Day 1 is available for anchor (Must be Day 1 of ALL data)
        all_dates = sorted(folder_analyses.keys())
        if all_dates:
            day1_name = all_dates[0]
            s1 = get_noon_samples_with_acc(folder_analyses[day1_name])
            
            refine_tasks = []
            # Refine only valid_folders (or missing ones)
            to_refine = valid_folders if (args.force_refine or args.force_analyze) else [d for d in valid_folders if d not in day_gaps]
            
            if to_refine and s1:
                print(f"\n[Phase 2] Measuring Global Anchor Gaps with {args.workers} workers...")
                for dayN_name in to_refine:
                    if dayN_name not in folder_analyses: continue
                    sN = get_noon_samples_with_acc(folder_analyses[dayN_name])
                    refine_tasks.append((day1_name, dayN_name, s1, sN))
                    
                with Pool(args.workers) as pool:
                    results = list(tqdm(pool.imap(measure_day_gap_worker, refine_tasks), total=len(refine_tasks)))
                    for target_day, gap in results:
                        day_gaps[target_day] = gap
                
                with open(refine_log_path, "w") as f:
                    json.dump(day_gaps, f, indent=2)

        # === Phase 3: Integration ===
        print(f"\n[Phase 3] Integrating global trajectory...")
        # Integrate ALL known dates to ensure continuity
        global_traj = integrate_trajectory(folder_analyses, day_gaps)
        
        with open(full_log_path, "w") as f:
            for folder in sorted(global_traj.keys()):
                for item in global_traj[folder]:
                    f.write(f"{folder}\t{item['filename']}\tdx={item['final_dx']:.1f}\tdy={item['final_dy']:.1f}\trot={item['rot']:.3f}\tstatus={item['status']}\tscale={item.get('brightness_scale', 1.0):.3f}\n")

    # === Phase 4: Rendering ===
    print(f"\n[Phase 4] Rendering frames for {len(valid_folders)} folders...")
    
    if args.render_only:
        print(f"Loading log from {full_log_path}...")
        global_traj = {}
        with open(full_log_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                parts = line.strip().split('\t')
                if len(parts) < 6: continue
                folder = parts[0]
                # Filter render list
                if folder not in valid_folders: continue
                
                kv = {}
                for p in parts[2:]:
                    k, v = p.split('=')
                    kv[k] = v
                if folder not in global_traj: global_traj[folder] = []
                global_traj[folder].append({
                    "filename": parts[1],
                    "abs_path": os.path.join(input_root, folder, parts[1]),
                    "final_dx": float(kv['dx']),
                    "final_dy": float(kv['dy']),
                    "rot": float(kv['rot']),
                    "status": kv['status'],
                    "brightness_scale": float(kv.get('scale', 1.0))
                })

    render_tasks = []
    for folder in valid_folders:
        if folder not in global_traj: continue
        render_tasks.append((
            os.path.join(input_root, folder),
            os.path.join(output_root, folder),
            global_traj[folder]
        ))
        
    with Pool(args.workers) as pool:
        list(tqdm(pool.imap(render_folder_worker, render_tasks), total=len(render_tasks)))

    # === Phase 5: Video ===
    if args.video and valid_folders:
        print("\n[Phase 5] Creating video...")
        
        start_d = valid_folders[0]
        end_d = valid_folders[-1]
        res_str = f"{args.resize_width}p" if args.resize_width else "Original"
        now_str = datetime.now().strftime("%H%M%S")
        sub_name = place_name if place_name else "timelapse"
        vid_filename = f"{sub_name}_{start_d}~{end_d}_{res_str}_{now_str}.mp4"
        vid_path = os.path.join(output_root, vid_filename)
        
        video_images = []
        for d in valid_folders:
            imgs = get_images(os.path.join(output_root, d), args.ext)
            video_images.extend(imgs)
            
        if video_images:
            create_video.create_video_chunked(
                input_dir=output_root, 
                output_file=vid_path, 
                fps=args.fps, 
                width=args.resize_width,
                image_list=video_images
            )
            print(f"Saved video to: {vid_path}")
        else:
            print("No images found for video.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="input")
    parser.add_argument("--output", "-o", default="output")
    parser.add_argument("--subfolder", "-f", help="Subfolder name inside input directory")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--ext", default="jpg")
    parser.add_argument("--workers", "-w", type=int, default=max(1, cpu_count()-1))
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--force-analyze", action="store_true")
    parser.add_argument("--force-refine", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--resize-width", type=int, help="Target video width (e.g. 1920)")
    parser.add_argument("--all-places", action="store_true", help="Process all subfolders in input directory sequentially")
    args = parser.parse_args()
    
    if args.all_places:
        if not os.path.exists(args.input):
            print(f"Input directory {args.input} does not exist.")
            return
            
        # Identify all subdirectores in input as 'places'
        places = sorted([d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))])
        if not places:
            print(f"No subfolders found in {args.input}")
            return
            
        print(f"Found {len(places)} places to process: {places}")
        
        for place in places:
            print(f"\n{'='*70}")
            print(f" PROCESSING PLACE: {place}")
            print(f"{'='*70}")
            
            place_input = os.path.join(args.input, place)
            place_output = os.path.join(args.output, place)
            log_prefix = f"{place}_"
            
            try:
                run_stabilization_task(place_input, place_output, log_prefix, place, args)
            except Exception as e:
                print(f"ERROR processing {place}: {e}")
                import traceback
                traceback.print_exc()

    else:
        # Path Resolution for Single Mode
        if args.subfolder:
            input_root = os.path.join(args.input, args.subfolder)
            output_root = os.path.join(args.output, args.subfolder)
            log_prefix = f"{args.subfolder}_"
            place_name = args.subfolder
        else:
            input_root = args.input
            output_root = args.output
            log_prefix = ""
            place_name = None
            
        run_stabilization_task(input_root, output_root, log_prefix, place_name, args)

if __name__ == "__main__":
    main()
