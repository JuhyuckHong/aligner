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

# Refinement Precision
DAY_REFINE_SAMPLES = 7 
ECC_ITERATIONS = 500  # Increased for better convergence
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
    
    if len(args) >= 3:
        progress_queue = args[2]
    
    if isinstance(second_arg, str):
        ext = second_arg
        images = get_images(input_dir, ext)
    else:
        images = second_arg # Explicit list
        
    if not images:
        return (os.path.basename(input_dir), [])

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
        "status": "OK"
    })
    
    # ECC Criteria from README (500 iterations)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-4)
    
    for i in range(1, len(images)):
        curr_path = images[i]
        curr_fname = os.path.basename(curr_path)
        
        # Report Progress
        if progress_queue:
            progress_queue.put(('P_INC', 1))
            if i % 10 == 0:
                progress_queue.put(f"[Analyzing] {folder_name}: {i}/{len(images)}")
            
        curr_img = cv2.imread(curr_path)
        if curr_img is None:
            results.append({
                "filename": curr_fname, "abs_path": curr_path,
                "dx": 0.0, "dy": 0.0, "rot": 0.0, "status": "FAIL_READ"
            })
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
            "status": status
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
    anchor_day, target_day, s1_list, s2_list = args
    
    if not s1_list or not s2_list:
        return (target_day, (0.0, 0.0, 0.0))
    
    if anchor_day == target_day:
        return (target_day, (0.0, 0.0, 0.0))
        
    gaps_x = []
    gaps_y = []
    gaps_rot = []
    scale = 0.5
    
    for samp1 in s1_list:
        if not os.path.exists(samp1['abs_path']): continue
        img1_full = cv2.imread(samp1['abs_path'])
        
        # Warp Anchor (Day 1) to its base
        img1_small = cv2.resize(img1_full, None, fx=scale, fy=scale)
        center_small = (img1_small.shape[1]//2, img1_small.shape[0]//2)
        
        M1_rot = cv2.getRotationMatrix2D(center_small, -samp1['acc_rot'], 1.0)
        M1_rot[0, 2] -= samp1['acc_dx'] * scale
        M1_rot[1, 2] -= samp1['acc_dy'] * scale
        warped1 = cv2.warpAffine(img1_small, M1_rot, (img1_small.shape[1], img1_small.shape[0]))
        gray1 = cv2.cvtColor(warped1, cv2.COLOR_BGR2GRAY)
        
        for samp2 in s2_list:
            if not os.path.exists(samp2['abs_path']): continue
            img2_full = cv2.imread(samp2['abs_path'])
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
        
    return (target_day, (final_gap_dx, final_gap_dy, final_gap_rot))

# ---------------------------------------------------------------------------
# PHASE 3: Integration
# ---------------------------------------------------------------------------
def integrate_trajectory(folder_analyses, day_gaps_dict):
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
        
        # Transition from Previous Day
        if i > 0:
            prev_folder = sorted_folders[i-1]
            last_img = folder_analyses[prev_folder][-1]['abs_path']
            first_img = frames[0]['abs_path']
            
            scale = 0.5
            prev_grad = cv2.resize(create_gradient(cv2.imread(last_img)), None, fx=scale, fy=scale)
            curr_grad = cv2.resize(create_gradient(cv2.imread(first_img)), None, fx=scale, fy=scale)
            
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
        raw_target_dx, raw_target_dy, raw_target_rot = day_gaps_dict.get(folder, (0.0, 0.0, 0.0))
        
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
        
        for idx, frame in enumerate(frames):
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
        
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, item in enumerate(trajectory):
        if progress_queue:
            progress_queue.put(('P_INC', 1))

        img = cv2.imread(item['abs_path'])
        if img is None: continue
            
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        dx = item['final_dx']
        dy = item['final_dy']
        rot = item['rot']
        
        M = cv2.getRotationMatrix2D(center, rot, 1.0) 
        M[0, 2] += dx
        M[1, 2] += dy
        
        aligned = cv2.warpAffine(img, M, (w, h), 
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        cv2.imwrite(os.path.join(output_dir, item['filename']), aligned, [cv2.IMWRITE_JPEG_QUALITY, 98])
        
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
                    f.write(f"{folder}\t{item['filename']}\tdx={item['final_dx']:.1f}\tdy={item['final_dy']:.1f}\trot={item['rot']:.3f}\tstatus={item['status']}\n")

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
                    "status": kv['status']
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
            create_video.create_chunk_video(video_images, vid_path, fps=args.fps, width=args.resize_width)
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
