"""
=============================================================================
timelapse_stabilizer.py - High-performance Parallel Timelapse Stabilizer
=============================================================================

Features:
  - 🚀 Parallel Processing: Analysis, Refinement, and Rendering
  - 💾 Modular Logging: JSON-based intermediate logs for easy resuming
  - 🎯 Virtual Alignment: Accurate Day-to-Day drift measurement using aligned samples
  - 🔄 Two-Pass Architecture: Separates motion estimation from image warping
  - ✨ Advanced Refinement: Early Correction Strategy

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

# === Configuration ===
ROTATION_THRESHOLD_DEG = 0.02
DAMPING_DEADZONE = 3.0
DAMPING_FACTOR = 0.99
DAY_REFINE_SAMPLES = 3

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
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
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
    input_dir, ext = args
    images = get_images(input_dir, ext)
    if not images:
        return (os.path.basename(input_dir), [])

    results = []
    scale = 0.5
    
    prev_img = cv2.imread(images[0])
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    prev_gray_small = cv2.resize(prev_gray, None, fx=scale, fy=scale)
    prev_grad = create_gradient(prev_img)
    prev_grad_small = cv2.resize(prev_grad, None, fx=scale, fy=scale)
    
    # First frame
    results.append({
        "filename": os.path.basename(images[0]),
        "rel_dx": 0.0, "rel_dy": 0.0, "rot": 0.0,
        "status": "FIRST", "abs_path": images[0]
    })
    
    for i in range(1, len(images)):
        curr_path = images[i]
        curr_img = cv2.imread(curr_path)
        filename = os.path.basename(curr_path)
        
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        curr_gray_small = cv2.resize(curr_gray, None, fx=scale, fy=scale)
        curr_grad = create_gradient(curr_img)
        curr_grad_small = cv2.resize(curr_grad, None, fx=scale, fy=scale)
        
        # 1. Translation
        dx, dy, resp = phase_correlation(prev_grad_small, curr_grad_small)
        dx /= scale
        dy /= scale
        
        # 2. Rotation
        angle, ecc_dx, ecc_dy, ecc_ok = get_euclidean_transform(prev_gray_small, curr_gray_small)
        
        final_dx, final_dy = dx, dy
        final_rot = 0.0
        status = "OK"
        
        if ecc_ok and abs(angle) > ROTATION_THRESHOLD_DEG:
            status = f"ROT({angle:.2f})"
            final_rot = angle
            
            # Derotate and Re-measure Translation
            center = (curr_grad_small.shape[1] // 2, curr_grad_small.shape[0] // 2)
            M_derot = cv2.getRotationMatrix2D(center, angle, 1.0)
            curr_grad_derot = cv2.warpAffine(curr_grad_small, M_derot, 
                                             (curr_grad_small.shape[1], curr_grad_small.shape[0]),
                                             flags=cv2.INTER_LINEAR)
            
            pc_dx, pc_dy, _ = phase_correlation(prev_grad_small, curr_grad_derot)
            final_dx = pc_dx / scale
            final_dy = pc_dy / scale
        
        if i % 50 == 0:
            print(f"  [Analyze] {os.path.basename(input_dir)}: {i}/{len(images)}", flush=True)

        results.append({
            "filename": filename,
            "rel_dx": float(final_dx), "rel_dy": float(final_dy), "rot": float(final_rot),
            "status": status, "abs_path": curr_path
        })
        
        prev_gray_small = curr_gray_small
        prev_grad_small = curr_grad_small
        
    return (os.path.basename(input_dir), results)

# ---------------------------------------------------------------------------
# PHASE 2: Refinement (Worker & Helpers)
# ---------------------------------------------------------------------------
def get_noon_samples_with_acc(analysis_list, n_samples=3):
    """
    Select noon samples and COMPUTE ACCUMULATED MOTION for them.
    Returns: list of dict {abs_path, acc_dx, acc_dy, acc_rot}
    """
    timed = []
    
    # Pre-compute accumulation for the whole list
    acc_dx, acc_dy, acc_rot = 0.0, 0.0, 0.0
    accumulated_data = []
    
    for item in analysis_list:
        acc_dx += item['rel_dx']
        acc_dy += item['rel_dy']
        acc_rot += item['rot']
        
        # Check time
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
    Compare Day N and Day N+1 using Virtual Alignment.
    """
    day1, day2, s1_list, s2_list = args
    
    if not s1_list or not s2_list:
        return (day2, (0.0, 0.0))
        
    gaps_x = []
    gaps_y = []
    scale = 0.5
    
    for samp1 in s1_list:
        if not os.path.exists(samp1['abs_path']): continue
        img1_full = cv2.imread(samp1['abs_path'])
        
        h, w = img1_full.shape[:2]
        center = (w//2, h//2)
        
        # Resize first for speed
        img1_small = cv2.resize(img1_full, None, fx=scale, fy=scale)
        center_small = (img1_small.shape[1]//2, img1_small.shape[0]//2)
        
        # Warp using INVERSE accumulated motion (to bring back to Day Base)
        M1_rot = cv2.getRotationMatrix2D(center_small, -samp1['acc_rot'], 1.0)
        M1_rot[0, 2] -= samp1['acc_dx'] * scale
        M1_rot[1, 2] -= samp1['acc_dy'] * scale
        
        warped1 = cv2.warpAffine(img1_small, M1_rot, (img1_small.shape[1], img1_small.shape[0]))
        grad1 = create_gradient(warped1)

        for samp2 in s2_list:
            if not os.path.exists(samp2['abs_path']): continue
            img2_full = cv2.imread(samp2['abs_path'])
            img2_small = cv2.resize(img2_full, None, fx=scale, fy=scale)
            
            M2_rot = cv2.getRotationMatrix2D(center_small, -samp2['acc_rot'], 1.0)
            M2_rot[0, 2] -= samp2['acc_dx'] * scale
            M2_rot[1, 2] -= samp2['acc_dy'] * scale
            
            warped2 = cv2.warpAffine(img2_small, M2_rot, (img2_small.shape[1], img2_small.shape[0]))
            grad2 = create_gradient(warped2)
            
            dx, dy, _ = phase_correlation(grad1, grad2)
            gaps_x.append(dx / scale)
            gaps_y.append(dy / scale)

    if gaps_x:
        final_gap_dx = float(np.median(gaps_x))
        final_gap_dy = float(np.median(gaps_y))
    else:
        final_gap_dx, final_gap_dy = 0.0, 0.0
        
    return (day2, (final_gap_dx, final_gap_dy))

# ---------------------------------------------------------------------------
# PHASE 3: Integration
# ---------------------------------------------------------------------------
def integrate_trajectory(folder_analyses, day_gaps_dict):
    sorted_folders = sorted(folder_analyses.keys())
    global_trajectory = {}
    
    # Calculate Cumulative Refine Targets
    day_refine_targets = {}
    cum_gap_dx, cum_gap_dy = 0.0, 0.0
    day_refine_targets[sorted_folders[0]] = (0.0, 0.0)
    
    for i in range(1, len(sorted_folders)):
        day = sorted_folders[i]
        gap_dx, gap_dy = day_gaps_dict.get(day, (0.0, 0.0))
        cum_gap_dx += gap_dx
        cum_gap_dy += gap_dy
        day_refine_targets[day] = (cum_gap_dx, cum_gap_dy)
        print(f"  Day {day}: Gap=({gap_dx:.1f}, {gap_dy:.1f}) -> Target ({cum_gap_dx:.1f}, {cum_gap_dy:.1f})")

    print("\n[Integration] Stitching days and applying Early Correction...")
    
    current_global_dx = 0.0
    current_global_dy = 0.0
    current_global_rot = 0.0
    
    for i, folder in enumerate(sorted_folders):
        frames = folder_analyses[folder]
        
        # Transition from Previous Day
        if i > 0:
            prev_folder = sorted_folders[i-1]
            last_img = folder_analyses[prev_folder][-1]['abs_path']
            first_img = frames[0]['abs_path']
            
            # Simple stitching (assuming rotation aligned by Virtual Align logic)
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
        
        target_dx, target_dy = day_refine_targets.get(folder, (0.0, 0.0))
        
        refine_gap_dx = target_dx - start_dx
        refine_gap_dy = target_dy - start_dy
        
        n_frames = len(frames)
        trans_len = int(n_frames * 0.2)
        if trans_len < 1: trans_len = 1
        
        local_acc_dx = 0.0
        local_acc_dy = 0.0
        local_acc_rot = 0.0
        
        folder_trajectory = []
        
        for idx, frame in enumerate(frames):
            local_acc_dx += frame['rel_dx']
            local_acc_dy += frame['rel_dy']
            local_acc_rot += frame['rot']
            
            # Damping
            if abs(local_acc_dx) > DAMPING_DEADZONE: local_acc_dx *= DAMPING_FACTOR
            if abs(local_acc_dy) > DAMPING_DEADZONE: local_acc_dy *= DAMPING_FACTOR
            
            # Refine
            if idx < trans_len:
                alpha = idx / trans_len
            else:
                alpha = 1.0
            
            shift_dx = alpha * refine_gap_dx
            shift_dy = alpha * refine_gap_dy
            
            final_dx = start_dx + local_acc_dx + shift_dx
            final_dy = start_dy + local_acc_dy + shift_dy
            final_rot = start_rot + local_acc_rot
            
            folder_trajectory.append({
                "filename": frame['filename'],
                "abs_path": frame['abs_path'],
                "final_dx": final_dx,
                "final_dy": final_dy,
                "rot": final_rot, # Absolute Rotation
                "status": frame['status']
            })
            
        global_trajectory[folder] = folder_trajectory
        
        current_global_dx = start_dx + local_acc_dx + refine_gap_dx
        current_global_dy = start_dy + local_acc_dy + refine_gap_dy
        current_global_rot = start_rot + local_acc_rot
        
    return global_trajectory

# ---------------------------------------------------------------------------
# PHASE 4: Rendering
# ---------------------------------------------------------------------------
def render_folder_worker(args):
    input_dir, output_dir, trajectory = args
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, item in enumerate(trajectory):
        img = cv2.imread(item['abs_path'])
        if img is None: continue
            
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        dx = item['final_dx']
        dy = item['final_dy']
        rot = item['rot']
        
        # Transform (Using Absolute Rotation)
        M = cv2.getRotationMatrix2D(center, rot, 1.0) 
        M[0, 2] += dx
        M[1, 2] += dy
        
        aligned = cv2.warpAffine(img, M, (w, h), 
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        cv2.imwrite(os.path.join(output_dir, item['filename']), aligned, [cv2.IMWRITE_JPEG_QUALITY, 98])
        
        if idx % 50 == 0:
            print(f"  [Render] {os.path.basename(input_dir)}: {idx}/{len(trajectory)} (rot={rot:.4f})", flush=True)
    
    return len(trajectory)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="input")
    parser.add_argument("--output", "-o", default="output")
    parser.add_argument("--ext", default="jpg")
    parser.add_argument("--workers", "-w", type=int, default=max(1, cpu_count()-1))
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--force-analyze", action="store_true")
    parser.add_argument("--force-refine", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--resize-width", type=int, help="Target video width (e.g. 1920)")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    analysis_log_path = os.path.join(args.output, "analysis_log.json")
    refine_log_path = os.path.join(args.output, "refine_log.json")
    full_log_path = os.path.join(args.output, "full_log.txt")
    
    # === Phase 1: Analysis ===
    folder_analyses = {}
    
    if args.render_only:
        pass 
    elif os.path.exists(analysis_log_path) and not args.force_analyze:
        print(f"Loading existing analysis from {analysis_log_path}...")
        with open(analysis_log_path, "r") as f:
            folder_analyses = json.load(f)
    else:
        print("SCANNING FOLDERS...")
        subfolders = sorted([d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))])
        tasks = [(os.path.join(args.input, d), args.ext) for d in subfolders]
        
        print(f"\n[Phase 1] Analyzing motion with {args.workers} workers...")
        with Pool(args.workers) as pool:
            for folder_name, results in tqdm(pool.imap(analyze_folder_worker, tasks), total=len(tasks)):
                folder_analyses[folder_name] = results
                
        # Save Analysis Log
        with open(analysis_log_path, "w") as f:
            json.dump(folder_analyses, f, indent=2)
            
    if args.render_only:
        pass
    else:
        # === Phase 2: Refinement (Parallel) ===
        day_gaps = {}
        if os.path.exists(refine_log_path) and not args.force_refine and not args.force_analyze:
            print(f"Loading existing refinement from {refine_log_path}...")
            with open(refine_log_path, "r") as f:
                day_gaps = json.load(f)
        else:
            print(f"\n[Phase 2] Measuring Day Gaps (Virtual Alignment) with {args.workers} workers...")
            # Prepare tasks
            sorted_folders = sorted(folder_analyses.keys())
            refine_tasks = []
            
            for i in range(len(sorted_folders)-1):
                day1 = sorted_folders[i]
                day2 = sorted_folders[i+1]
                s1 = get_noon_samples_with_acc(folder_analyses[day1])
                s2 = get_noon_samples_with_acc(folder_analyses[day2])
                refine_tasks.append((day1, day2, s1, s2))
                
            with Pool(args.workers) as pool:
                results = list(tqdm(pool.imap(measure_day_gap_worker, refine_tasks), total=len(refine_tasks)))
                for day2, gap in results:
                    day_gaps[day2] = gap
            
            with open(refine_log_path, "w") as f:
                json.dump(day_gaps, f, indent=2)

        # === Phase 3: Integration ===
        print(f"\n[Phase 3] Integrating global trajectory...")
        global_traj = integrate_trajectory(folder_analyses, day_gaps)
        
        # Save Full Log
        with open(full_log_path, "w") as f:
            for folder in sorted(global_traj.keys()):
                for item in global_traj[folder]:
                    f.write(f"{folder}\t{item['filename']}\tdx={item['final_dx']:.1f}\tdy={item['final_dy']:.1f}\trot={item['rot']:.3f}\tstatus={item['status']}\n")

    # === Phase 4: Rendering ===
    print(f"\n[Phase 4] Rendering frames with {args.workers} workers...")
    
    # Load traj if render only
    if args.render_only:
        print(f"Loading log from {full_log_path}...")
        global_traj = {}
        with open(full_log_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                parts = line.strip().split('\t')
                if len(parts) < 6: continue
                folder = parts[0]
                kv = {}
                for p in parts[2:]:
                    k, v = p.split('=')
                    kv[k] = v
                if folder not in global_traj: global_traj[folder] = []
                global_traj[folder].append({
                    "filename": parts[1],
                    "abs_path": os.path.join(args.input, folder, parts[1]),
                    "final_dx": float(kv['dx']),
                    "final_dy": float(kv['dy']),
                    "rot": float(kv['rot']),
                    "status": kv['status']
                })

    render_tasks = []
    subfolders = sorted(global_traj.keys())
    for folder in subfolders:
        render_tasks.append((
            os.path.join(args.input, folder),
            os.path.join(args.output, folder),
            global_traj[folder]
        ))
        
    with Pool(args.workers) as pool:
        list(tqdm(pool.imap(render_folder_worker, render_tasks), total=len(render_tasks)))

    # === Phase 5: Video ===
    if args.video:
        print("\n[Phase 5] Creating video...")
        cmd = ["python", "create_video.py", "-i", args.output, "-o", os.path.join(args.output, "combined.mp4"), "--fps", str(args.fps)]
        if args.resize_width:
            cmd.extend(["--width", str(args.resize_width)])
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
