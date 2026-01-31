"""
gui_visualizer.py - Enhanced Manual Alignment Visualizer with Mouse Support
"""
import cv2
import numpy as np

class ManualAlignVisualizer:
    def __init__(self, ref_img_path, mov_img_path, initial_dx=0.0, initial_dy=0.0):
        self.ref_path = ref_img_path
        self.mov_path = mov_img_path
        
        # Load images
        self.ref_img_full = cv2.imread(ref_img_path)
        self.mov_img_full = cv2.imread(mov_img_path)
        
        if self.ref_img_full is None or self.mov_img_full is None:
            raise FileNotFoundError("Could not load images")
            
        self.h_full, self.w_full = self.ref_img_full.shape[:2]
        
        # Display settings
        self.max_display_h = 900
        self.display_scale = 1.0
        if self.h_full > self.max_display_h:
            self.display_scale = self.max_display_h / self.h_full
            
        self.dx = float(initial_dx)
        self.dy = float(initial_dy)
        self.angle = 0.0 
        self.scale = 1.0
        
        self.show_ref = True
        self.overlay_mode = True # Default to overlay for better UX
        self.mouse_x = self.w_full // 2
        self.mouse_y = self.h_full // 2
        
        # Interaction state
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.start_dx = 0
        self.start_dy = 0
        
        # Window names
        self.win_main = "Manual Alignment (Drag to Move, Q to Save, ESC to Cancel)"
        self.win_zoom = "Fine Tune (x4)"
        
    def get_display_img(self, img_full):
        """Resize full image to display size"""
        if self.display_scale == 1.0:
            return img_full
        return cv2.resize(img_full, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_AREA)

    def run(self):
        cv2.namedWindow(self.win_main)
        cv2.namedWindow(self.win_zoom)
        cv2.setMouseCallback(self.win_main, self.mouse_callback)
        
        print("\n=== Visualizer Started ===")
        print("Controls:")
        print("  [Mouse Drag]: Move Image")
        print("  [WASD]: 1px | [IJKL]: 10px")
        print("  [Space]: Toggle Reference/Aligned")
        print("  [Z]: Toggle Overlay Mode")
        print("  [Q]: Save & Exit")
        print("  [ESC]: Cancel")
        
        while True:
            # 1. Apply Transformation
            center = (self.w_full // 2, self.h_full // 2)
            M = cv2.getRotationMatrix2D(center, self.angle, self.scale)
            M[0, 2] += self.dx
            M[1, 2] += self.dy
            
            aligned_full = cv2.warpAffine(
                self.mov_img_full, M, (self.w_full, self.h_full),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            
            # 2. Prepare Display
            if self.overlay_mode:
                display_full = cv2.addWeighted(self.ref_img_full, 0.5, aligned_full, 0.5, 0)
                mode_text = "OVERLAY"
            elif self.show_ref:
                display_full = self.ref_img_full.copy()
                mode_text = "REFERENCE"
            else:
                display_full = aligned_full.copy()
                mode_text = "ALIGNED"
                
            display_view = self.get_display_img(display_full)
            
            # Draw UI
            info = f"[{mode_text}] dx={self.dx:.2f} dy={self.dy:.2f}"
            cv2.putText(display_view, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw Crosshair at mouse
            view_mouse_x = int(self.mouse_x * self.display_scale)
            view_mouse_y = int(self.mouse_y * self.display_scale)
            cv2.drawMarker(display_view, (view_mouse_x, view_mouse_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)

            cv2.imshow(self.win_main, display_view)
            
            # 3. Zoom Window
            zoom_size = 200
            zoom_factor = 4
            half_roi = zoom_size // (2 * zoom_factor)
            
            x1 = max(0, self.mouse_x - half_roi)
            y1 = max(0, self.mouse_y - half_roi)
            x2 = min(self.w_full, self.mouse_x + half_roi)
            y2 = min(self.h_full, self.mouse_y + half_roi)
            
            crop = display_full[y1:y2, x1:x2]
            if crop.size > 0:
                zoomed = cv2.resize(crop, (zoom_size, zoom_size), interpolation=cv2.INTER_NEAREST)
                cv2.imshow(self.win_zoom, zoomed)
            
            # 4. Input
            key = cv2.waitKey(20) & 0xFF
            
            # Keyboard move
            if   key == ord('i'): self.dy -= 10
            elif key == ord('k'): self.dy += 10
            elif key == ord('j'): self.dx -= 10
            elif key == ord('l'): self.dx += 10
            elif key == ord('w'): self.dy -= 1
            elif key == ord('s'): self.dy += 1
            elif key == ord('a'): self.dx -= 1
            elif key == ord('d'): self.dx += 1
            elif key == ord('8'): self.dy -= 0.1
            elif key == ord('2'): self.dy += 0.1
            elif key == ord('4'): self.dx -= 0.1
            elif key == ord('6'): self.dx += 0.1
            
            elif key == ord(' '):
                self.show_ref = not self.show_ref
                self.overlay_mode = False
            elif key == ord('z'):
                self.overlay_mode = not self.overlay_mode
                
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return self.dx, self.dy
            elif key == 27: # ESC
                cv2.destroyAllWindows()
                return None
            elif key == ord('r'): # Reset
                self.dx = 0.0
                self.dy = 0.0

    def mouse_callback(self, event, x, y, flags, param):
        # Update mouse position (always useful for zoom)
        self.mouse_x = int(x / self.display_scale)
        self.mouse_y = int(y / self.display_scale)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_dragging = True
            self.drag_start_x = x
            self.drag_start_y = y
            self.start_dx = self.dx
            self.start_dy = self.dy
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging:
                # Calculate delta in View coordinates
                delta_view_x = x - self.drag_start_x
                delta_view_y = y - self.drag_start_y
                
                # Convert to Full coordinates
                delta_full_x = delta_view_x / self.display_scale
                delta_full_y = delta_view_y / self.display_scale
                
                # Update dx, dy (Inverted because dragging image means 'moving' it)
                # Actually if I drag RIGHT, I want the image to move RIGHT.
                self.dx = self.start_dx + delta_full_x
                self.dy = self.start_dy + delta_full_y
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False

if __name__ == "__main__":
    # Test stub
    pass
