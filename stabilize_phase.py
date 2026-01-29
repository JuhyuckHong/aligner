import cv2
import numpy as np
import os
import subprocess
from glob import glob
from tqdm import tqdm

def get_images(input_dir, ext='jpg'):
    patterns = [f"*.{ext}", f"*.{ext.upper()}"]
    images = []
    for pattern in patterns:
        images.extend(glob(os.path.join(input_dir, pattern)))
    return sorted(images)

def create_edge(img):
    """Convert image to edge map using Canny"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def phase_correlation(ref, mov):
    """Compute translation using phase correlation"""
    f1 = np.float32(ref)
    f2 = np.float32(mov)
    
    # Apply Hanning window
    h, w = f1.shape
    hann_row = np.hanning(h)
    hann_col = np.hanning(w)
    hann_2d = np.outer(hann_row, hann_col).astype(np.float32)
    
    f1 = f1 * hann_2d
    f2 = f2 * hann_2d
    
    shift, response = cv2.phaseCorrelate(f1, f2)
    
    # Invert signs (phaseCorrelate returns "how much mov shifted from ref")
    # We need "how much to shift mov to align with ref"
    dx = -shift[0]
    dy = -shift[1]
    
    return dx, dy, response

def create_video(image_folder, output_file, fps=30, ext='jpg'):
    images = get_images(image_folder, ext)
    if not images: return
    
    list_file = os.path.join(image_folder, "ffmpeg_list.txt")
    with open(list_file, "w") as f:
        for img in images:
            f.write(f"file '{os.path.abspath(img).replace(chr(92), '/')}'\n")
    
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
           "-r", str(fps), output_file]
    subprocess.run(cmd)
    os.remove(list_file)
    print(f"Video created: {output_file}")

def stabilize_phase_correlation(input_dir, output_dir, ext='jpg'):
    image_paths = get_images(input_dir, ext)
    if not image_paths:
        print("No images found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    n = len(image_paths)
    
    # Reference: Middle frame (best lighting)
    ref_idx = n // 2
    ref_path = image_paths[ref_idx]
    print(f"Reference Frame: {os.path.basename(ref_path)} (index {ref_idx})")
    
    ref_img = cv2.imread(ref_path)
    h, w = ref_img.shape[:2]
    ref_edge = create_edge(ref_img)
    
    # Downscale edges for faster phase correlation
    scale = 0.5
    ref_edge_small = cv2.resize(ref_edge, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    print(f"Stabilizing {n} frames using Phase Correlation on Edge images...")
    
    # Parameters for outlier rejection
    MAX_SHIFT = 100
    last_dx, last_dy = 0.0, 0.0
    skipped = 0
    
    for i in tqdm(range(n)):
        img_path = image_paths[i]
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Create edge image
        edge = create_edge(img)
        edge_small = cv2.resize(edge, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # Phase correlation
        dx, dy, response = phase_correlation(ref_edge_small, edge_small)
        
        # Scale back to original resolution
        dx /= scale
        dy /= scale
        
        # Outlier rejection
        if abs(dx) > MAX_SHIFT or abs(dy) > MAX_SHIFT or response < 0.05:
            dx, dy = last_dx, last_dy
            skipped += 1
        else:
            last_dx, last_dy = dx, dy
        
        # Apply translation to original image
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Save
        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), aligned, [cv2.IMWRITE_JPEG_QUALITY, 98])
    
    print(f"Stabilization Complete. Skipped {skipped} outlier frames.")

def process_all_folders(base_input, base_output, ext='jpg'):
    """Process all subfolders in base_input directory"""
    # Check if base_input has subfolders or direct images
    subfolders = [d for d in os.listdir(base_input) 
                  if os.path.isdir(os.path.join(base_input, d))]
    
    if subfolders:
        print(f"Found {len(subfolders)} subfolders in '{base_input}'")
        print(f"Subfolders: {subfolders[:5]}{'...' if len(subfolders) > 5 else ''}\n")
        
        for folder in sorted(subfolders):
            input_path = os.path.join(base_input, folder)
            output_path = os.path.join(base_output, folder)
            
            # Check if folder has images
            images = get_images(input_path, ext)
            if not images:
                print(f"[SKIP] {folder}: No images found")
                continue
            
            print(f"\n{'='*50}")
            print(f"Processing: {folder} ({len(images)} images)")
            print(f"{'='*50}")
            
            stabilize_phase_correlation(input_path, output_path, ext)
    else:
        # No subfolders, process base_input directly
        print(f"No subfolders found. Processing '{base_input}' directly.")
        stabilize_phase_correlation(base_input, base_output, ext)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stabilize timelapse images using Phase Correlation")
    parser.add_argument("--input", "-i", default="input", help="Input folder (default: input)")
    parser.add_argument("--output", "-o", default="output", help="Output folder (default: output)")
    parser.add_argument("--ext", default="jpg", help="Image extension (default: jpg)")
    parser.add_argument("--video", "-v", action="store_true", help="Also create videos for each folder")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    
    args = parser.parse_args()
    
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()
    
    # Process all folders
    process_all_folders(args.input, args.output, args.ext)
    
    # Optionally create videos
    if args.video:
        print(f"\n{'='*50}")
        print("Creating videos...")
        print(f"{'='*50}")
        
        subfolders = [d for d in os.listdir(args.output) 
                      if os.path.isdir(os.path.join(args.output, d))]
        
        if subfolders:
            for folder in sorted(subfolders):
                output_path = os.path.join(args.output, folder)
                video_path = os.path.join(args.output, f"{folder}.mp4")
                print(f"\nCreating video: {video_path}")
                create_video(output_path, video_path, args.fps, args.ext)
        else:
            video_path = "output.mp4"
            create_video(args.output, video_path, args.fps, args.ext)
    
    print("\n✓ All done!")
