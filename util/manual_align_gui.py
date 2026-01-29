"""
Manual Alignment GUI Tool
- Compare two images and manually adjust alignment
- Zoom preview for precise comparison

Controls:
  WASD: Move 1px
  IJKL: Move 10px
  SPACE: Toggle Ref/Aligned (both main and zoom view)
  Z: Overlay mode
  Mouse: Hover for zoom preview position
  Q: Quit
"""

import cv2
import numpy as np
import argparse
import os

class ManualAlignGUI:
    def __init__(self, ref_path, mov_path):
        self.ref_img = cv2.imread(ref_path)
        self.mov_img = cv2.imread(mov_path)
        
        if self.ref_img is None:
            raise ValueError(f"Cannot load reference image: {ref_path}")
        if self.mov_img is None:
            raise ValueError(f"Cannot load moving image: {mov_path}")
        
        self.h, self.w = self.ref_img.shape[:2]
        self.dx, self.dy = 0, 0  # Translation offset
        
        # Display state
        self.show_ref = True  # True = show ref, False = show aligned
        self.overlay_mode = False
        self.mouse_x, self.mouse_y = self.w // 2, self.h // 2
        
        # Zoom settings
        self.zoom_size = 200  # Size of zoom window
        self.zoom_factor = 4  # Magnification
        
        # Window names
        self.main_win = "Main View"
        self.zoom_win = "Zoom View"
        
        print(f"\n=== Manual Alignment GUI ===")
        print(f"Reference: {ref_path}")
        print(f"Moving: {mov_path}")
        print(f"\nControls:")
        print(f"  WASD: Move 1px | IJKL: Move 10px")
        print(f"  SPACE: Toggle Ref/Aligned")
        print(f"  Z: Overlay mode")
        print(f"  Mouse: Hover for zoom preview")
        print(f"  Q: Quit and save offset")
    
    def get_aligned_image(self):
        """Apply current translation to moving image"""
        M = np.float32([[1, 0, self.dx], [0, 1, self.dy]])
        aligned = cv2.warpAffine(self.mov_img, M, (self.w, self.h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
        return aligned
    
    def get_overlay(self, img1, img2, alpha=0.5):
        """Create overlay of two images"""
        return cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)
    
    def get_zoom_crop(self, img, cx, cy):
        """Get zoomed crop around center point"""
        half = self.zoom_size // (2 * self.zoom_factor)
        
        # Clamp to image bounds
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(self.w, cx + half)
        y2 = min(self.h, cy + half)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((self.zoom_size, self.zoom_size, 3), dtype=np.uint8)
        
        # Zoom in
        zoomed = cv2.resize(crop, (self.zoom_size, self.zoom_size), 
                           interpolation=cv2.INTER_NEAREST)
        return zoomed
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse movement for zoom position"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
    
    def update_display(self):
        """Update main and zoom windows"""
        aligned = self.get_aligned_image()
        
        # Main view
        if self.overlay_mode:
            main_display = self.get_overlay(self.ref_img, aligned)
            mode_text = "OVERLAY"
        elif self.show_ref:
            main_display = self.ref_img.copy()
            mode_text = "REFERENCE"
        else:
            main_display = aligned.copy()
            mode_text = "ALIGNED"
        
        # Add info text
        info = f"Mode: {mode_text} | Offset: ({self.dx}, {self.dy})"
        cv2.putText(main_display, info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw crosshair at mouse position
        cv2.drawMarker(main_display, (self.mouse_x, self.mouse_y),
                      (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
        
        cv2.imshow(self.main_win, main_display)
        
        # Zoom view - side by side comparison
        if self.overlay_mode:
            zoom_ref = self.get_zoom_crop(self.ref_img, self.mouse_x, self.mouse_y)
            zoom_aligned = self.get_zoom_crop(aligned, self.mouse_x, self.mouse_y)
            zoom_overlay = self.get_overlay(zoom_ref, zoom_aligned)
            zoom_display = zoom_overlay
            # Add label
            cv2.putText(zoom_display, "OVERLAY", (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif self.show_ref:
            # Show reference in zoom
            zoom_display = self.get_zoom_crop(self.ref_img, self.mouse_x, self.mouse_y)
            cv2.putText(zoom_display, "REF", (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # Show aligned in zoom
            zoom_display = self.get_zoom_crop(aligned, self.mouse_x, self.mouse_y)
            cv2.putText(zoom_display, "ALIGNED", (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow(self.zoom_win, zoom_display)
    
    def run(self):
        """Main event loop"""
        cv2.namedWindow(self.main_win)
        cv2.namedWindow(self.zoom_win)
        cv2.setMouseCallback(self.main_win, self.mouse_callback)
        
        while True:
            self.update_display()
            key = cv2.waitKey(30) & 0xFF
            
            # Movement controls
            if key == ord('w'):      # Up 1px
                self.dy -= 1
            elif key == ord('s'):    # Down 1px
                self.dy += 1
            elif key == ord('a'):    # Left 1px
                self.dx -= 1
            elif key == ord('d'):    # Right 1px
                self.dx += 1
            elif key == ord('i'):    # Up 10px
                self.dy -= 10
            elif key == ord('k'):    # Down 10px
                self.dy += 10
            elif key == ord('j'):    # Left 10px
                self.dx -= 10
            elif key == ord('l'):    # Right 10px
                self.dx += 10
            
            # View controls
            elif key == ord(' '):    # Toggle ref/aligned
                self.show_ref = not self.show_ref
                self.overlay_mode = False
            elif key == ord('z'):    # Overlay mode
                self.overlay_mode = not self.overlay_mode
            
            # Quit
            elif key == ord('q') or key == 27:  # Q or ESC
                break
        
        cv2.destroyAllWindows()
        
        print(f"\nFinal offset: dx={self.dx}, dy={self.dy}")
        return self.dx, self.dy


def manual_align_gui(ref_path, mov_path):
    """Convenience function to run the GUI"""
    gui = ManualAlignGUI(ref_path, mov_path)
    return gui.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Alignment GUI")
    parser.add_argument("--ref", "-r", help="Reference image path")
    parser.add_argument("--mov", "-m", help="Moving image path")
    parser.add_argument("--input-dir", "-i", help="Input directory (uses first two images)")
    
    args = parser.parse_args()
    
    if args.ref and args.mov:
        ref_path = args.ref
        mov_path = args.mov
    elif args.input_dir:
        from glob import glob
        images = sorted(glob(os.path.join(args.input_dir, "*.jpg")))
        if len(images) < 2:
            images = sorted(glob(os.path.join(args.input_dir, "*.JPG")))
        if len(images) < 2:
            print("Error: Need at least 2 images in the directory")
            exit(1)
        ref_path = images[0]
        mov_path = images[1]
    else:
        # Default: use test_input folder
        from glob import glob
        images = sorted(glob("test_input/*.jpg"))
        if len(images) < 2:
            images = sorted(glob("test_input/*.JPG"))
        if len(images) >= 2:
            ref_path = images[0]
            mov_path = images[1]
        else:
            print("Usage: python manual_align_gui.py --ref REF_IMG --mov MOV_IMG")
            print("   or: python manual_align_gui.py --input-dir FOLDER")
            exit(1)
    
    dx, dy = manual_align_gui(ref_path, mov_path)
    print(f"Offset to apply: dx={dx}, dy={dy}")
