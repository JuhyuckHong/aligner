"""
=============================================================================
stabilize_parallel.py - High-performance Parallel Timelapse Stabilizer
=============================================================================

Benefits:
  - 🚀 Parallel Processing: Scales with CPU cores (Analysis & Rendering)
  - 🔄 Two-Pass Architecture: Separates motion estimation from image warping
  - 🛠️ Robust Algorithms: Gradient PC, Center ECC, Hybrid Alignment
  - ✨ Advanced Refinement: Day-to-Day Early Correction (Morning Transition)

Usage:
  python stabilize_parallel.py --video --fps 30 --workers 4

Author: Antigravity (Google Deepmind)
"""

import cv2
import numpy as np
import os
import argparse
import json
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import subprocess
import shutil

# === Configuration ===
ROTATION_THRESHOLD_DEG = 0.02
DAMPING_DEADZONE = 3.0
DAMPING_FACTOR = 0.99
DAY_REFINE_SAMPLES = 3  # Median of 3 samples closest to noon

def get_images(input_dir, ext='jpg'):
    patterns = [f"*.{ext}", f"*.{ext.upper()}"]
    images = []
    for pattern in patterns:
        images.extend(glob(os.path.join(input_dir, pattern)))
    return sorted(images)

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
# STEP 1: Analysis (Worker Function)
# ---------------------------------------------------------------------------
def analyze_folder_worker(args):
    """
    Analyze a single folder to compute frame-to-frame relative motion.
    Returns: List of dicts {filename, rel_dx, rel_dy, rot, status}
    """
    input_dir, ext = args
    images = get_images(input_dir, ext)
    if not images:
        return []

    results = []
    scale = 0.5
    h, w = cv2.imread(images[0]).shape[:2]
    
    # Initialize previous frame data
    prev_img = cv2.imread(images[0])
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    prev_gray_small = cv2.resize(prev_gray, None, fx=scale, fy=scale)
    prev_grad = create_gradient(prev_img)
    prev_grad_small = cv2.resize(prev_grad, None, fx=scale, fy=scale)
    
    # First frame is always 0 relative motion
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
        
        # 1. Gradient Phase Correlation (Translation)
        dx, dy, resp = phase_correlation(prev_grad_small, curr_grad_small)
        dx /= scale
        dy /= scale
        
        # 2. ECC Check (Rotation)
        angle, ecc_dx, ecc_dy, ecc_ok = get_euclidean_transform(prev_gray_small, curr_gray_small)
        
        final_dx, final_dy = dx, dy
        final_rot = 0.0
        status = "OK"
        
        # Rotation Logic (Hybrid)
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
        
        if i % 20 == 0:
            print(f"  [Analyze] {os.path.basename(input_dir)}: {i}/{len(images)}", flush=True)
        
        results.append({
            "filename": filename,
            "rel_dx": final_dx,
            "rel_dy": final_dy,
            "rot": final_rot,
            "status": status,
            "abs_path": curr_path
        })
        
        # Update references
        prev_gray_small = curr_gray_small
        prev_grad_small = curr_grad_small
        
    return (os.path.basename(input_dir), results)

# ---------------------------------------------------------------------------
# STEP 2: Global Integration (Main Process)
# ---------------------------------------------------------------------------
def compute_global_trajectory(folder_analyses):
    """
    Connect days and apply refinement to compute absolute coordinates for every frame.
    """
    # Sort by folder name (date)
    sorted_folders = sorted(folder_analyses.keys())
    
    global_trajectory = {} # {folder: [ {filename, final_dx, final_dy, rot} ]}
    
    # Track accumulated offset at the END of previous day
    prev_day_end_dx = 0.0
    prev_day_end_dy = 0.0
    
    # Store day-to-day gaps for calculating Refine Offsets
    day_refine_targets = {sorted_folders[0]: (0,0)} # Day 1 target is 0,0
    
    # 1. Measure Day-to-Day Gaps (Refinement Targets)
    print("\n[Refinement] Measuring Day-to-Day Gaps...")
    cum_gap_dx, cum_gap_dy = 0.0, 0.0
    
    for i in range(len(sorted_folders)-1):
        day1 = sorted_folders[i]
        day2 = sorted_folders[i+1]
        
        # Get 3 samples closest to noon
        def get_noon_samples(analysis_list):
            timed = []
            for item in analysis_list:
                try:
                    basename = item['filename']
                    time_part = basename.split('_')[1].split('.')[0]
                    h, m, s = map(int, time_part.split('-'))
                    diff = abs((h*60+m) - 720)
                    timed.append((diff, item['abs_path']))
                except: continue
            timed.sort(key=lambda x: x[0])
            return [x[1] for x in timed[:3]]
            
        s1 = get_noon_samples(folder_analyses[day1])
        s2 = get_noon_samples(folder_analyses[day2])
        
        if s1 and s2:
            gaps_x, gaps_y = [], []
            scale = 0.5
            for p1 in s1:
                img1 = cv2.resize(create_gradient(cv2.imread(p1)), None, fx=scale, fy=scale)
                for p2 in s2:
                    img2 = cv2.resize(create_gradient(cv2.imread(p2)), None, fx=scale, fy=scale)
                    dx, dy, _ = phase_correlation(img1, img2)
                    gaps_x.append(dx / scale)
                    gaps_y.append(dy / scale)
            
            day_gap_dx = np.median(gaps_x)
            day_gap_dy = np.median(gaps_y)
        else:
            day_gap_dx, day_gap_dy = 0, 0
            
        cum_gap_dx += day_gap_dx
        cum_gap_dy += day_gap_dy
        day_refine_targets[day2] = (cum_gap_dx, cum_gap_dy)
        print(f"  {day1} -> {day2}: Gap=({day_gap_dx:.1f}, {day_gap_dy:.1f}) -> Target ({cum_gap_dx:.1f}, {cum_gap_dy:.1f})")

    # 2. Integrate Trajectories
    print("\n[Integration] Stitching days and applying Early Correction...")
    
    current_global_dx = 0.0
    current_global_dy = 0.0
    
    for i, folder in enumerate(sorted_folders):
        frames = folder_analyses[folder]
        
        # Transition from Previous Day (Night-to-Night)
        if i > 0:
            prev_folder = sorted_folders[i-1]
            last_img = folder_analyses[prev_folder][-1]['abs_path']
            first_img = frames[0]['abs_path']
            
            # Gradient PC
            scale = 0.5
            prev_grad = cv2.resize(create_gradient(cv2.imread(last_img)), None, fx=scale, fy=scale)
            curr_grad = cv2.resize(create_gradient(cv2.imread(first_img)), None, fx=scale, fy=scale)
            
            trans_dx, trans_dy, _ = phase_correlation(prev_grad, curr_grad)
            trans_dx /= scale
            trans_dy /= scale
            
            print(f"  Transition {prev_folder}->{folder}: ({trans_dx:.1f}, {trans_dy:.1f})")
            
            # Apply transition to global accumulator
            current_global_dx += trans_dx
            current_global_dy += trans_dy
        
        # Local Trajectory Integration
        folder_trajectory = []
        
        # Prepare Early Correction Params
        # Start point: current_global (connected from prev night)
        # Target point: day_refine_targets[folder] (target from noon comparison)
        
        start_dx = current_global_dx
        start_dy = current_global_dy
        
        target_dx, target_dy = day_refine_targets[folder]
        
        # Gap to fix
        refine_gap_dx = target_dx - start_dx
        refine_gap_dy = target_dy - start_dy
        
        # Transition settings (Early 20%)
        n_frames = len(frames)
        trans_len = int(n_frames * 0.2)
        max_err = max(abs(refine_gap_dx), abs(refine_gap_dy))
        min_steps = int(max_err / 0.5)
        if min_steps > trans_len: trans_len = min(min_steps, n_frames)
        
        # Intra-day Accumulator
        local_acc_dx = 0.0
        local_acc_dy = 0.0
        
        for idx, frame in enumerate(frames):
            # Accumulate frame-to-frame motion
            local_acc_dx += frame['rel_dx']
            local_acc_dy += frame['rel_dy']
            
            # Damping (Deadzone)
            if abs(local_acc_dx) > DAMPING_DEADZONE: local_acc_dx *= DAMPING_FACTOR
            if abs(local_acc_dy) > DAMPING_DEADZONE: local_acc_dy *= DAMPING_FACTOR
            
            # Calculate Refine Shift (Early Correction)
            if idx < trans_len:
                alpha = idx / trans_len if trans_len > 0 else 1.0
            else:
                alpha = 1.0
            
            shift_dx = alpha * refine_gap_dx
            shift_dy = alpha * refine_gap_dy
            
            # Final Coordinate = Start(NightConn) + LocalMov + RefineShift
            # Wait, local_acc is relative to start of day.
            # Start(NightConn) is the absolute pos of start of day.
            # RefineShift is the correction added on top.
            
            final_dx = start_dx + local_acc_dx + shift_dx
            final_dy = start_dy + local_acc_dy + shift_dy
            
            folder_trajectory.append({
                "filename": frame['filename'],
                "abs_path": frame['abs_path'],
                "final_dx": final_dx,
                "final_dy": final_dy,
                "rot": frame['rot'],
                "status": frame['status']
            })
            
        global_trajectory[folder] = folder_trajectory
        
        # Update global accumulator for next day chain
        # Note: The next day connects to the LAST frame of this day.
        # But we must respect the Refine Target for the long term.
        # So we should use the accumulated value including refinement.
        current_global_dx = start_dx + local_acc_dx + refine_gap_dx
        current_global_dy = start_dy + local_acc_dy + refine_gap_dy
        
    return global_trajectory

# ---------------------------------------------------------------------------
# STEP 3: Rendering (Worker Function)
# ---------------------------------------------------------------------------
def render_folder_worker(args):
    """
    Render optimized frames based on computed trajectory.
    """
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
        
        # Create Transform Matrix
        # 1. Rotation (Center)
        M = cv2.getRotationMatrix2D(center, rot, 1.0)
        
        # 2. Translation (After rotation)
        M[0, 2] += dx
        M[1, 2] += dy
        
        aligned = cv2.warpAffine(img, M, (w, h), 
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        cv2.imwrite(os.path.join(output_dir, item['filename']), aligned, [cv2.IMWRITE_JPEG_QUALITY, 98])
        
        if idx % 50 == 0:
            print(f"  [Render] {os.path.basename(input_dir)}: {idx}/{len(trajectory)}", flush=True)
    
    return len(trajectory)

# ---------------------------------------------------------------------------
# Main Controller
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="input")
    parser.add_argument("--output", "-o", default="output_parallel")
    parser.add_argument("--ext", default="jpg")
    parser.add_argument("--workers", "-w", type=int, default=max(1, cpu_count()-1))
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    
    # 1. Clean Output
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)
    
    # 2. Scan Folders
    print("SCANNING FOLDERS...")
    subfolders = sorted([d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))])
    tasks = [(os.path.join(args.input, d), args.ext) for d in subfolders]
    
    # 3. Analyze (Parallel)
    print(f"\n[Phase 1] Analyzing motion with {args.workers} workers...")
    folder_analyses = {}
    with Pool(args.workers) as pool:
        for folder_name, results in tqdm(pool.imap(analyze_folder_worker, tasks), total=len(tasks)):
            folder_analyses[folder_name] = results
            
    # 4. Integrate (Sequential)
    print(f"\n[Phase 2] Integrating global trajectory...")
    global_traj = compute_global_trajectory(folder_analyses)
    
    # Write Log
    log_path = os.path.join(args.output, "full_log.txt")
    with open(log_path, "w") as f:
        for folder in sorted(global_traj.keys()):
            for item in global_traj[folder]:
                f.write(f"{folder}\t{item['filename']}\tdx={item['final_dx']:.1f}\tdy={item['final_dy']:.1f}\trot={item['rot']:.3f}\tstatus={item['status']}\n")
    print(f"Log saved to {log_path}")
    
    # 5. Render (Parallel)
    print(f"\n[Phase 3] Rendering frames with {args.workers} workers...")
    render_tasks = []
    for folder in subfolders:
        render_tasks.append((
            os.path.join(args.input, folder),
            os.path.join(args.output, folder),
            global_traj[folder]
        ))
        
    with Pool(args.workers) as pool:
        list(tqdm(pool.imap(render_folder_worker, render_tasks), total=len(render_tasks)))
        
    # 6. Video
    if args.video:
        print("\n[Phase 4] Creating video...")
        # (Reuse create_video logic or simple ffmpeg concat)
        # Use stabilize_phase.py's video logic or external call
        subprocess.run(["python", "create_video.py", "-i", args.output, "-o", os.path.join(args.output, "combined.mp4"), "--fps", str(args.fps)])

if __name__ == "__main__":
    main()
