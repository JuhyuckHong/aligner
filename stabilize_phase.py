"""
=============================================================================
stabilize_phase.py - 타임랩스 이미지 정합 메인 스크립트
=============================================================================

알고리즘:
  1. Chained Neighbor Alignment - 이웃 프레임 간 Phase Correlation으로 이동량 계산
  2. Rotation Correction - 회전이 0.1° 이상이면 ECC로 회전 보정
  3. Deadzone + Soft Damping - 드리프트 방지 (±3px 범위 내 자유, 초과 시 0.99 계수)
  4. Day-level Refinement - 날짜 간 랜덤 샘플링으로 중앙값 오프셋 보정

사용법:
  # 기본 실행 (input/ -> output/)
  python stabilize_phase.py
  
  # 영상까지 생성
  python stabilize_phase.py --video --fps 30 --crf 23
  
  # 커스텀 폴더
  python stabilize_phase.py -i my_input -o my_output
  
  # Day Refinement 건너뛰기
  python stabilize_phase.py --no-refine

출력:
  - output/[날짜]/[이미지].jpg  : 보정된 이미지
  - output/logs/*_full.txt     : 전체 로그 (dx, dy, rot, resp, status)
  - output/logs/*_outliers.txt : 아웃라이어 목록
  - output/combined_all.mp4    : 통합 영상 (--video 옵션 시)

로그 형식:
  filename  dx=0.0  dy=0.0  rot=0.000  resp=1.000  status=OK
  - rot: 회전 각도 (degree)
  - status: FIRST / OK / OUTLIER / ROT_CORR
=============================================================================
"""
import cv2
import numpy as np
import os
import subprocess
import tempfile
import shutil
import random
from glob import glob
from tqdm import tqdm



# === Configuration ===
ROTATION_THRESHOLD_DEG = 0.1  # Apply rotation correction if > this value
DAMPING_DEADZONE = 3.0        # px, no damping within this range
DAMPING_FACTOR = 0.99         # Pull back 1% per frame when outside deadzone
DAY_REFINE_SAMPLES = 30       # Samples per day for day-level refinement

def get_images(input_dir, ext='jpg'):
    patterns = [f"*.{ext}", f"*.{ext.upper()}"]
    images = []
    for pattern in patterns:
        images.extend(glob(os.path.join(input_dir, pattern)))
    return sorted(images)

def create_edge(img):
    """Convert image to edge map using Canny (Deprecated: use create_gradient)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def create_gradient(img):
    """Convert image to gradient magnitude (Robust to lighting)"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    # Sobel Gradient (CV_32F to avoid overflow)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    
    # Normalize to 0-255 uint8 (for consistency with edge map)
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return norm

def phase_correlation(ref, mov):
    """Compute translation using phase correlation"""
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

def get_euclidean_transform(ref_gray, mov_gray, scale=0.5):
    """
    Get Euclidean transform (rotation + translation) using ECC.
    Returns: angle_deg, dx, dy, success
    """
    ref_small = cv2.resize(ref_gray, None, fx=scale, fy=scale)
    mov_small = cv2.resize(mov_gray, None, fx=scale, fy=scale)
    
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001)
    
    try:
        cc, warp_matrix = cv2.findTransformECC(mov_small, ref_small, warp_matrix, warp_mode, criteria)
        angle_rad = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])
        angle_deg = np.degrees(angle_rad)
        dx = warp_matrix[0, 2] / scale
        dy = warp_matrix[1, 2] / scale
        
        # Store full warp matrix scaled
        warp_matrix[0, 2] = dx
        warp_matrix[1, 2] = dy
        
        return angle_deg, dx, dy, warp_matrix, True
    except:
        return 0, 0, 0, None, False

def apply_transform(img, dx, dy, angle_deg=0, warp_matrix=None):
    """Apply translation (and optionally rotation) to image"""
    h, w = img.shape[:2]
    
    if warp_matrix is not None and abs(angle_deg) > ROTATION_THRESHOLD_DEG:
        # Use full Euclidean transform
        aligned = cv2.warpAffine(img, warp_matrix, (w, h), 
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    else:
        # Translation only
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(img, M, (w, h), 
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned

def create_chunk_video(image_list, output_file, fps=30, crf=18):
    """Create a video from a list of images using concat demuxer"""
    list_file = output_file + ".txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for img in image_list:
            abs_path = os.path.abspath(img).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")
            f.write(f"duration {1/fps}\n")
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", list_file,
        "-vf", "scale=-2:1080",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", str(crf), "-preset", "medium",
        "-r", str(fps), "-movflags", "+faststart",
        output_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        os.remove(list_file)
    except:
        pass
    return result.returncode == 0

def concat_videos(video_list, output_file):
    """Concatenate multiple videos into one"""
    list_file = output_file + "_concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for vid in video_list:
            abs_path = os.path.abspath(vid).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")
    
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", output_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        os.remove(list_file)
    except:
        pass
    return result.returncode == 0

def stabilize_folder_chained(input_dir, output_dir, ext, img_shape, scale=0.5,
                              prev_acc_dx=0.0, prev_acc_dy=0.0, prev_gray_small=None,
                              prev_last_img_path=None):
    """
    Stabilize a single folder using CHAINED neighbor alignment with rotation correction.
    """
    image_paths = get_images(input_dir, ext)
    if not image_paths:
        return [], [], 0.0, 0.0, None
    
    os.makedirs(output_dir, exist_ok=True)
    n = len(image_paths)
    h, w = img_shape
    
    full_log = []
    outliers = []
    rotation_count = 0
    
    acc_dx = prev_acc_dx
    acc_dy = prev_acc_dy
    
    # Process first frame
    first_path = image_paths[0]
    first_img = cv2.imread(first_path)
    first_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
    first_gray_small = cv2.resize(first_gray, None, fx=scale, fy=scale)
    first_edge = create_gradient(first_img)
    first_edge_small = cv2.resize(first_edge, None, fx=scale, fy=scale)
    
    # Chain to previous folder (Day Transition)
    if prev_last_img_path is not None and os.path.exists(prev_last_img_path):
        # 1. Load previous last image
        prev_img = cv2.imread(prev_last_img_path)
        if prev_img is not None:
            # 2. Compute Gradient (Robust to lighting changes)
            prev_grad = create_gradient(prev_img)
            prev_grad_small = cv2.resize(prev_grad, None, fx=scale, fy=scale)
            
            # 3. Phase Correlation (Benchmark confirmed this is best for Day Transition)
            # first_edge_small is already computed as Gradient in previous steps
            day_dx, day_dy, day_resp = phase_correlation(prev_grad_small, first_edge_small)
            
            dx = day_dx / scale
            dy = day_dy / scale
            
            method = "GradPC"
            print(f"  Day transition [{method}]: dx={dx:.2f}, dy={dy:.2f}, resp={day_resp:.3f}")
        else:
            dx, dy = 0.0, 0.0
            print("  Day transition [FAIL]: Could not load prev image")
    elif prev_gray_small is not None:
        # Fallback if path is missing but gray image exists (rare)
        # Compute gradient on small image (suboptimal but works)
        prev_grad_small = create_gradient(prev_gray_small)
        day_dx, day_dy, day_resp = phase_correlation(prev_grad_small, first_edge_small)
        dx = day_dx / scale
        dy = day_dy / scale
        print(f"  Day transition [GradPC-Small]: dx={dx:.2f}, dy={dy:.2f}")
    else:
        dx, dy = 0.0, 0.0

    acc_dx += dx
    acc_dy += dy
    
    prev_edge_small = first_edge_small
    prev_gray_small_local = first_gray_small
    last_gray_small = first_gray_small
    
    for i in tqdm(range(n), desc=f"  Stabilizing", leave=False):
        curr_path = image_paths[i]
        filename = os.path.basename(curr_path)
        curr_img = cv2.imread(curr_path)
        if curr_img is None:
            continue
        
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        
        if i == 0:
            dx_frame = 0
            dy_frame = 0
            response = 1.0
            status = "FIRST"
            angle = 0
            warp_matrix = None
        else:
            curr_edge = create_gradient(curr_img)
            curr_edge_small = cv2.resize(curr_edge, None, fx=scale, fy=scale)
            curr_gray_small = cv2.resize(curr_gray, None, fx=scale, fy=scale)

            # Phase correlation for translation
            dx_frame, dy_frame, response = phase_correlation(prev_edge_small, curr_edge_small)
            dx_frame /= scale
            dy_frame /= scale
            
            # Check rotation using ECC
            angle, ecc_dx, ecc_dy, warp_matrix, ecc_success = get_euclidean_transform(
                prev_gray_small_local, curr_gray_small, scale=1.0)
            
            status = "OK"
            if abs(dx_frame) > 50 or abs(dy_frame) > 50 or response < 0.03:
                status = "OUTLIER"
                outliers.append(f"{filename}\tdx={dx_frame:.1f}, dy={dy_frame:.1f}, resp={response:.3f}")
                dx_frame = 0
                dy_frame = 0
                angle = 0
                warp_matrix = None
            elif abs(angle) > ROTATION_THRESHOLD_DEG and ecc_success:
                # Use ECC results for rotation correction
                status = "ROT_CORR"
                dx_frame = ecc_dx
                dy_frame = ecc_dy
                rotation_count += 1
            else:
                warp_matrix = None  # No rotation needed
            
            # Accumulate
            acc_dx += dx_frame
            acc_dy += dy_frame
            
            # Deadzone + Soft Damping
            if abs(acc_dx) > DAMPING_DEADZONE:
                acc_dx *= DAMPING_FACTOR
            if abs(acc_dy) > DAMPING_DEADZONE:
                acc_dy *= DAMPING_FACTOR
            
            prev_edge_small = curr_edge_small
            prev_gray_small_local = curr_gray_small
            last_gray_small = curr_gray_small
        
        full_log.append(f"{filename}\tdx={acc_dx:.1f}\tdy={acc_dy:.1f}\trot={angle:.3f}\tresp={response:.3f}\tstatus={status}")
        
        # Apply transformation
        if warp_matrix is not None and abs(angle) > ROTATION_THRESHOLD_DEG:
            # Apply Euclidean (rotation + translation)
            warp_matrix[0, 2] = acc_dx
            warp_matrix[1, 2] = acc_dy
            aligned = cv2.warpAffine(curr_img, warp_matrix, (w, h), 
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        else:
            # Translation only
            M = np.float32([[1, 0, acc_dx], [0, 1, acc_dy]])
            aligned = cv2.warpAffine(curr_img, M, (w, h), 
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        cv2.imwrite(os.path.join(output_dir, filename), aligned, [cv2.IMWRITE_JPEG_QUALITY, 98])
    
    if rotation_count > 0:
        print(f"  Rotation corrections applied: {rotation_count}")
    
    # Return last image path for grid matching in next folder
    last_img_path = image_paths[-1] if image_paths else None
    return full_log, outliers, acc_dx, acc_dy, last_gray_small, last_img_path

def refine_day_alignment(base_dir, folders):
    """
    Post-process to correct day-level drift using random sampling.
    Modifies images in-place.
    """
    print("\n=== Day-Level Refinement ===")
    
    def get_day_samples(folder_path, n_samples=DAY_REFINE_SAMPLES):
        images = sorted(glob(os.path.join(folder_path, "*.jpg")))
        if not images:
            return []
        mid_range = images[len(images)//4 : 3*len(images)//4]
        if len(mid_range) < n_samples:
            return mid_range
        return random.sample(mid_range, n_samples)
    
    def calculate_day_offset(folder1, folder2):
        samples1 = get_day_samples(folder1)
        samples2 = get_day_samples(folder2)
        if not samples1 or not samples2:
            return 0, 0
        
        dx_list, dy_list = [], []
        scale = 0.5
        
        for s1 in samples1:
            img1 = cv2.imread(s1)
            edge1 = create_edge(img1)
            e1 = cv2.resize(edge1, None, fx=scale, fy=scale)
            
            for s2 in samples2:
                img2 = cv2.imread(s2)
                edge2 = create_edge(img2)
                e2 = cv2.resize(edge2, None, fx=scale, fy=scale)
                
                dx, dy, resp = phase_correlation(e1, e2)
                dx /= scale
                dy /= scale
                
                if resp > 0.05:
                    dx_list.append(dx)
                    dy_list.append(dy)
        
        if not dx_list:
            return 0, 0
        return np.median(dx_list), np.median(dy_list)
    
    # Calculate cumulative day offsets
    day_offsets = {folders[0]: (0.0, 0.0)}
    cumulative_dx, cumulative_dy = 0.0, 0.0
    
    for i in range(1, len(folders)):
        prev_folder = os.path.join(base_dir, folders[i-1])
        curr_folder = os.path.join(base_dir, folders[i])
        
        dx, dy = calculate_day_offset(prev_folder, curr_folder)
        cumulative_dx += dx
        cumulative_dy += dy
        day_offsets[folders[i]] = (cumulative_dx, cumulative_dy)
        
        print(f"  {folders[i-1]} -> {folders[i]}: dx={dx:.1f}, dy={dy:.1f} (cumul: {cumulative_dx:.1f}, {cumulative_dy:.1f})")
    
    # Apply offsets
    print("\n  Applying day-level corrections...")
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        offset_dx, offset_dy = day_offsets[folder]
        
        if abs(offset_dx) < 0.5 and abs(offset_dy) < 0.5:
            continue  # Skip if negligible
        
        images = sorted(glob(os.path.join(folder_path, "*.jpg")))
        for img_path in tqdm(images, desc=f"    {folder}", leave=False):
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            M = np.float32([[1, 0, offset_dx], [0, 1, offset_dy]])
            corrected = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            cv2.imwrite(img_path, corrected, [cv2.IMWRITE_JPEG_QUALITY, 98])

def process_all_folders(base_input, base_output, ext='jpg', refine=True):
    """Process all subfolders with chained alignment + rotation correction + day refinement"""
    subfolders = [d for d in os.listdir(base_input) 
                  if os.path.isdir(os.path.join(base_input, d))]
    
    if not subfolders:
        print(f"No subfolders found in '{base_input}'")
        return
    
    subfolders = sorted(subfolders)
    print(f"Found {len(subfolders)} subfolders in '{base_input}'")
    
    # Get image shape
    first_images = get_images(os.path.join(base_input, subfolders[0]), ext)
    if not first_images:
        print("No images found!")
        return
    
    sample_img = cv2.imread(first_images[0])
    img_shape = sample_img.shape[:2]
    scale = 0.5
    
    print(f"Image shape: {img_shape[1]}x{img_shape[0]}")
    print(f"\nUsing CHAINED alignment + Rotation correction (threshold={ROTATION_THRESHOLD_DEG}°)")
    print(f"Damping: Deadzone={DAMPING_DEADZONE}px, Factor={DAMPING_FACTOR}\n")
    
    # Process folders
    all_logs = []
    all_outliers = []
    
    prev_acc_dx = 0.0
    prev_acc_dy = 0.0
    prev_gray_small = None
    prev_last_img_path = None  # For grid-based day transition
    
    for folder in subfolders:
        input_path = os.path.join(base_input, folder)
        output_path = os.path.join(base_output, folder)
        
        images = get_images(input_path, ext)
        if not images:
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing: {folder} ({len(images)} images)")
        print(f"{'='*50}")
        
        folder_log, folder_outliers, prev_acc_dx, prev_acc_dy, prev_gray_small, prev_last_img_path = stabilize_folder_chained(
            input_path, output_path, ext, img_shape, scale,
            prev_acc_dx, prev_acc_dy, prev_gray_small, prev_last_img_path
        )
        
        print(f"  Final offset: dx={prev_acc_dx:.1f}, dy={prev_acc_dy:.1f}")
        
        for entry in folder_log:
            all_logs.append(f"{folder}\t{entry}")
        for entry in folder_outliers:
            all_outliers.append(f"{folder}\t{entry}")
    
    # Day-level refinement
    if refine and len(subfolders) > 1:
        refine_day_alignment(base_output, subfolders)
    
    # Save logs
    from datetime import datetime
    exec_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    first_date = subfolders[0]
    last_date = subfolders[-1]
    
    logs_dir = os.path.join(base_output, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    base_name = f"{exec_time}_{first_date}_to_{last_date}"
    full_log_path = os.path.join(logs_dir, f"{base_name}_full.txt")
    outliers_path = os.path.join(logs_dir, f"{base_name}_outliers.txt")
    
    with open(full_log_path, "w", encoding="utf-8") as f:
        f.write(f"# Stabilization Log\n")
        f.write(f"# Execution: {exec_time}\n")
        f.write(f"# Date Range: {first_date} to {last_date}\n")
        f.write(f"# Method: Chained Neighbor + Rotation Correction + Day Refinement\n")
        f.write(f"# Rotation Threshold: {ROTATION_THRESHOLD_DEG}°\n")
        f.write(f"# Damping: Deadzone={DAMPING_DEADZONE}px, Factor={DAMPING_FACTOR}\n\n")
        f.write("\n".join(all_logs))
    
    with open(outliers_path, "w", encoding="utf-8") as f:
        f.write(f"# Outlier Report\n")
        f.write(f"# Total outliers: {len(all_outliers)}\n\n")
        f.write("\n".join(all_outliers))
    
    print(f"\n{'='*50}")
    print(f"Full log: {full_log_path}")
    print(f"Outliers: {outliers_path} ({len(all_outliers)} items)")
    print(f"{'='*50}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stabilize timelapse with rotation correction")
    parser.add_argument("--input", "-i", default="input", help="Input folder")
    parser.add_argument("--output", "-o", default="output", help="Output folder")
    parser.add_argument("--ext", default="jpg", help="Image extension")
    parser.add_argument("--video", "-v", action="store_true", help="Create video")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--crf", type=int, default=18, help="Video quality (0-51)")
    parser.add_argument("--batch", type=int, default=500, help="Batch size for video")
    parser.add_argument("--no-refine", action="store_true", help="Skip day-level refinement")
    
    args = parser.parse_args()
    
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()
    
    process_all_folders(args.input, args.output, args.ext, refine=not args.no_refine)
    
    if args.video:
        print(f"\n{'='*50}")
        print("Creating combined video...")
        print(f"{'='*50}")
        
        subfolders = [d for d in os.listdir(args.output) 
                      if os.path.isdir(os.path.join(args.output, d)) and d != 'logs']
        
        all_images = []
        for folder in sorted(subfolders):
            folder_path = os.path.join(args.output, folder)
            images = get_images(folder_path, args.ext)
            all_images.extend(images)
        
        if all_images:
            print(f"Total frames: {len(all_images)}")
            combined_path = os.path.join(args.output, "combined_all.mp4")
            
            if len(all_images) <= args.batch:
                if create_chunk_video(all_images, combined_path, args.fps, args.crf):
                    print(f"✓ Video created: {combined_path}")
            else:
                print(f"Processing in batches of {args.batch}...")
                temp_dir = tempfile.mkdtemp(prefix="ffmpeg_")
                chunk_videos = []
                
                try:
                    for i in range(0, len(all_images), args.batch):
                        chunk = all_images[i:i+args.batch]
                        chunk_num = i // args.batch + 1
                        total_chunks = (len(all_images) + args.batch - 1) // args.batch
                        
                        chunk_file = os.path.join(temp_dir, f"chunk_{chunk_num:04d}.mp4")
                        print(f"  Chunk {chunk_num}/{total_chunks}...")
                        
                        if create_chunk_video(chunk, chunk_file, args.fps, args.crf):
                            chunk_videos.append(chunk_file)
                    
                    print("Concatenating chunks...")
                    if concat_videos(chunk_videos, combined_path):
                        print(f"✓ Combined video created: {combined_path}")
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n✓ All done!")
