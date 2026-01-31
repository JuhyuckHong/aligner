"""
Outlier Review Tool
- Load outliers.log (consolidated format) and review each frame with manual alignment GUI
- Save manual GT (ground truth) offsets for algorithm improvement

Usage:
    python review_outliers.py --log output/outliers.log
    python review_outliers.py --log output/outliers.log --mode consecutive
"""

import cv2
import numpy as np
import argparse
import os
from glob import glob
from datetime import datetime

class OutlierReviewer:
    def __init__(self, base_input, base_output, log_file=None):
        self.base_input = base_input
        self.base_output = base_output
        self.log_file = log_file or os.path.join(base_output, "outliers.log")
        self.gt_file = os.path.join(base_output, "manual_gt.log")
        
        # GUI settings
        self.zoom_size = 200
        self.zoom_factor = 4
        
        # Load outliers
        self.outliers = []
        self.global_reference = None
        self.load_outliers()
    
    def load_outliers(self):
        """Load outlier list from consolidated log file"""
        if not os.path.exists(self.log_file):
            print(f"Log file not found: {self.log_file}")
            return
        
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    if "Global Reference:" in line:
                        self.global_reference = line.split("Global Reference:")[1].strip()
                    continue
                if not line:
                    continue
                
                parts = line.split("\t")
                if len(parts) >= 3:
                    # New format: folder, filename, reason
                    self.outliers.append({
                        "folder": parts[0],
                        "filename": parts[1],
                        "reason": parts[2]
                    })
                elif len(parts) >= 2:
                    # Old format: filename, reason (single folder mode)
                    self.outliers.append({
                        "folder": "",
                        "filename": parts[0],
                        "reason": parts[1]
                    })
        
        print(f"Loaded {len(self.outliers)} outliers")
        print(f"Global Reference: {self.global_reference}")
    
    def get_image_path(self, folder, filename):
        """Get full path to image"""
        if folder:
            return os.path.join(self.base_input, folder, filename)
        return os.path.join(self.base_input, filename)
    
    def get_all_images_in_folder(self, folder):
        """Get sorted list of images in a folder"""
        if folder:
            folder_path = os.path.join(self.base_input, folder)
        else:
            folder_path = self.base_input
        
        images = sorted(glob(os.path.join(folder_path, "*.jpg")) + 
                       glob(os.path.join(folder_path, "*.JPG")))
        return images
    
    def get_neighbors(self, folder, filename, n=1):
        """Get n neighbors before and after the given file"""
        images = self.get_all_images_in_folder(folder)
        image_index = {os.path.basename(p): i for i, p in enumerate(images)}
        
        if filename not in image_index:
            return [], []
        
        idx = image_index[filename]
        before = images[max(0, idx-n):idx]
        after = images[idx+1:idx+1+n]
        return before, after
    
    def manual_align(self, ref_path, mov_path):
        """Run manual alignment GUI and return offset"""
        ref_img = cv2.imread(ref_path)
        mov_img = cv2.imread(mov_path)
        
        if ref_img is None or mov_img is None:
            print(f"Error loading images")
            return None, None
        
        h, w = ref_img.shape[:2]
        dx, dy = 0, 0
        show_ref = True
        overlay_mode = False
        mouse_x, mouse_y = w // 2, h // 2
        
        main_win = "Review: " + os.path.basename(mov_path)
        zoom_win = "Zoom"
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal mouse_x, mouse_y
            if event == cv2.EVENT_MOUSEMOVE:
                mouse_x, mouse_y = x, y
        
        cv2.namedWindow(main_win)
        cv2.namedWindow(zoom_win)
        cv2.setMouseCallback(main_win, mouse_callback)
        
        print(f"\n--- Reviewing: {os.path.basename(mov_path)} ---")
        print("WASD: 1px | IJKL: 10px | SPACE: toggle | Z: overlay | Q: save & next | ESC: skip")
        
        while True:
            # Apply current offset
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            aligned = cv2.warpAffine(mov_img, M, (w, h), 
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
            
            # Main view
            if overlay_mode:
                display = cv2.addWeighted(ref_img, 0.5, aligned, 0.5, 0)
                mode = "OVERLAY"
            elif show_ref:
                display = ref_img.copy()
                mode = "REF"
            else:
                display = aligned.copy()
                mode = "ALIGNED"
            
            # Info text
            cv2.putText(display, f"[{mode}] dx={dx}, dy={dy}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.drawMarker(display, (mouse_x, mouse_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
            
            cv2.imshow(main_win, display)
            
            # Zoom view
            half = self.zoom_size // (2 * self.zoom_factor)
            x1, y1 = max(0, mouse_x - half), max(0, mouse_y - half)
            x2, y2 = min(w, mouse_x + half), min(h, mouse_y + half)
            
            if overlay_mode:
                zoom_img = cv2.addWeighted(ref_img, 0.5, aligned, 0.5, 0)
            elif show_ref:
                zoom_img = ref_img
            else:
                zoom_img = aligned
            
            crop = zoom_img[y1:y2, x1:x2]
            if crop.size > 0:
                zoomed = cv2.resize(crop, (self.zoom_size, self.zoom_size), interpolation=cv2.INTER_NEAREST)
                cv2.putText(zoomed, mode, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(zoom_win, zoomed)
            
            key = cv2.waitKey(30) & 0xFF
            
            # Movement
            if key == ord('w'): dy -= 1
            elif key == ord('s'): dy += 1
            elif key == ord('a'): dx -= 1
            elif key == ord('d'): dx += 1
            elif key == ord('i'): dy -= 10
            elif key == ord('k'): dy += 10
            elif key == ord('j'): dx -= 10
            elif key == ord('l'): dx += 10
            
            # View mode
            elif key == ord(' '):
                show_ref = not show_ref
                overlay_mode = False
            elif key == ord('z'):
                overlay_mode = not overlay_mode
            
            # Save and next
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return dx, dy
            
            # Skip
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None, None
        
        cv2.destroyAllWindows()
        return None, None
    
    def save_gt(self, folder, filename, ref_filename, dx, dy):
        """Append manual GT to log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.gt_file, "a", encoding="utf-8") as f:
            f.write(f"{folder}\t{filename}\t{ref_filename}\t{dx}\t{dy}\t{timestamp}\n")
        
        print(f"Saved GT: {folder}/{filename} -> dx={dx}, dy={dy}")
    
    def review_all(self, use_neighbor_ref=False):
        """Review all outliers with manual alignment"""
        if not self.outliers:
            print("No outliers to review")
            return
        
        # Global reference path
        if self.global_reference:
            ref_path = self.global_reference
            ref_name = os.path.basename(ref_path)
        else:
            print("No global reference specified in log")
            return
        
        print(f"\n{'='*50}")
        print(f"Reviewing {len(self.outliers)} outliers")
        print(f"Global Reference: {ref_name}")
        print(f"{'='*50}")
        
        # Write GT file header if new
        if not os.path.exists(self.gt_file):
            with open(self.gt_file, "w", encoding="utf-8") as f:
                f.write("# Manual Ground Truth for Algorithm Improvement\n")
                f.write(f"# Global Reference: {self.global_reference}\n")
                f.write("# Format: folder\tfilename\treference\tdx\tdy\ttimestamp\n\n")
        
        reviewed = 0
        for i, outlier in enumerate(self.outliers):
            folder = outlier["folder"]
            filename = outlier["filename"]
            mov_path = self.get_image_path(folder, filename)
            
            if not os.path.exists(mov_path):
                print(f"[{i+1}/{len(self.outliers)}] File not found: {folder}/{filename}")
                continue
            
            # Optionally use neighbor as reference
            current_ref = ref_path
            current_ref_name = ref_name
            
            if use_neighbor_ref:
                before, _ = self.get_neighbors(folder, filename, 1)
                if before:
                    current_ref = before[-1]
                    current_ref_name = os.path.basename(current_ref)
                    print(f"Using neighbor reference: {current_ref_name}")
            
            print(f"\n[{i+1}/{len(self.outliers)}] {folder}/{filename}")
            print(f"Reason: {outlier['reason']}")
            
            dx, dy = self.manual_align(current_ref, mov_path)
            
            if dx is not None:
                self.save_gt(folder, filename, current_ref_name, dx, dy)
                reviewed += 1
            else:
                print(f"Skipped: {filename}")
        
        print(f"\n{'='*50}")
        print(f"Review complete. {reviewed}/{len(self.outliers)} frames processed.")
        print(f"GT saved to: {self.gt_file}")

    def review_consecutive(self):
        """Review consecutive outliers together, grouped by folder"""
        if not self.outliers:
            print("No outliers to review")
            return
        
        # Group by folder first
        folder_outliers = {}
        for outlier in self.outliers:
            folder = outlier["folder"]
            if folder not in folder_outliers:
                folder_outliers[folder] = []
            folder_outliers[folder].append(outlier)
        
        print(f"\nOutliers by folder:")
        for folder, outliers in folder_outliers.items():
            print(f"  {folder}: {len(outliers)} outliers")
        
        # Write GT file header if new
        if not os.path.exists(self.gt_file):
            with open(self.gt_file, "w", encoding="utf-8") as f:
                f.write("# Manual Ground Truth for Algorithm Improvement\n")
                f.write(f"# Global Reference: {self.global_reference}\n")
                f.write("# Format: folder\tfilename\treference\tdx\tdy\ttimestamp\n\n")
        
        # Process each folder
        for folder, outliers in sorted(folder_outliers.items()):
            print(f"\n{'='*50}")
            print(f"Folder: {folder} ({len(outliers)} outliers)")
            print(f"{'='*50}")
            
            # Get all images in this folder for index lookup
            all_images = self.get_all_images_in_folder(folder)
            image_index = {os.path.basename(p): i for i, p in enumerate(all_images)}
            
            # Group consecutive outliers within this folder
            groups = []
            current_group = []
            
            for outlier in outliers:
                filename = outlier["filename"]
                if filename not in image_index:
                    continue
                
                idx = image_index[filename]
                
                if not current_group:
                    current_group = [(outlier, idx)]
                elif idx == current_group[-1][1] + 1:
                    current_group.append((outlier, idx))
                else:
                    if current_group:
                        groups.append(current_group)
                    current_group = [(outlier, idx)]
            
            if current_group:
                groups.append(current_group)
            
            print(f"Found {len(groups)} consecutive groups")
            
            # Review each group
            for gi, group in enumerate(groups):
                print(f"\n--- Group {gi+1}/{len(groups)}: {len(group)} frames ---")
                
                # Use frame before first outlier as reference
                first_idx = group[0][1]
                if first_idx > 0:
                    ref_path = all_images[first_idx - 1]
                else:
                    ref_path = self.global_reference
                
                ref_name = os.path.basename(ref_path)
                print(f"Reference: {ref_name}")
                
                for outlier, idx in group:
                    filename = outlier["filename"]
                    mov_path = self.get_image_path(folder, filename)
                    
                    print(f"\nReviewing: {filename}")
                    print(f"Reason: {outlier['reason']}")
                    
                    dx, dy = self.manual_align(ref_path, mov_path)
                    
                    if dx is not None:
                        self.save_gt(folder, filename, ref_name, dx, dy)
                    else:
                        print(f"Skipped: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Review outliers with manual alignment")
    parser.add_argument("--log", "-l", help="Path to consolidated outliers.log file")
    parser.add_argument("--input", "-i", default="input", help="Base input folder (default: input)")
    parser.add_argument("--output", "-o", default="output", help="Base output folder (default: output)")
    parser.add_argument("--mode", "-m", choices=["all", "consecutive", "neighbor"], default="all",
                       help="Review mode: all (use global ref), consecutive (group outliers), neighbor (use prev frame)")
    
    args = parser.parse_args()
    
    log_file = args.log or os.path.join(args.output, "outliers.log")
    
    reviewer = OutlierReviewer(args.input, args.output, log_file)
    
    if args.mode == "consecutive":
        reviewer.review_consecutive()
    elif args.mode == "neighbor":
        reviewer.review_all(use_neighbor_ref=True)
    else:
        reviewer.review_all(use_neighbor_ref=False)


if __name__ == "__main__":
    main()
