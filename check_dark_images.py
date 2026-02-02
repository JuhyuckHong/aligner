
import cv2
import numpy as np
import os
import glob
from timelapse_stabilizer import is_image_dark, DARK_THRESHOLD

def check_dark_images():
    base_dir = r"c:\Users\juanh\VS\projects\aligner\input\Seoul_Gangnam_Hanog"
    print(f"Scanning {base_dir} for dark images (Threshold: {DARK_THRESHOLD})...")
    
    # Check all subfolders (dates)
    date_folders = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    total_dark = 0
    total_checked = 0
    
    for date in date_folders:
        day_dir = os.path.join(base_dir, date)
        images = sorted(glob.glob(os.path.join(day_dir, "*.jpg")))
        
        if not images: continue
        
        print(f"\nChecking {date} ({len(images)} images)...")
        
        # Check first 3, middle 3, last 3 to sample
        # But if user wants to see 'dark', we should probably check specifically early/late times
        # Let's check all and just print the ones that ARE dark or close to it.
        
        count_dark_day = 0
        for img_path in images:
            total_checked += 1
            img = cv2.imread(img_path)
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_b = np.mean(gray)
            is_dark = mean_b < DARK_THRESHOLD
            
            if is_dark:
                print(f"  [DARK] {os.path.basename(img_path)} (Brightness: {mean_b:.2f})")
                count_dark_day += 1
            elif mean_b < DARK_THRESHOLD + 15: # Show near-misses
                print(f"  [NEAR] {os.path.basename(img_path)} (Brightness: {mean_b:.2f})")
                
        if count_dark_day == 0:
            print("  No dark images found.")
        else:
            total_dark += count_dark_day

    print(f"\nTotal: {total_dark} dark images found out of {total_checked} checked.")

if __name__ == "__main__":
    check_dark_images()
