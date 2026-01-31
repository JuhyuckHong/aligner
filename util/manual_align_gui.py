"""
=============================================================================
util/manual_align_gui.py - 수동 정렬 GUI 도구
=============================================================================

두 이미지를 비교하며 수동으로 오프셋을 조정하는 GUI.
정합 결과 검증이나 GT(Ground Truth) 수집에 사용.

사용법:
  # 두 이미지 직접 지정
  python util/manual_align_gui.py --ref reference.jpg --mov moving.jpg
  
  # 폴더의 처음 두 이미지 사용
  python util/manual_align_gui.py --input-dir output/2026-01-01

조작법:
  WASD        : 1px 이동 (상/좌/하/우)
  IJKL        : 10px 이동
  8426        : 0.1px 이동 (숫자패드 방향)
  7193        : 0.01px 이동 (숫자패드 대각선)
  SPACE       : Reference ↔ Aligned 토글
  Z           : Overlay 모드 (반투명 겹침)
  Q           : 저장 후 종료 (오프셋 출력)
  ESC         : 취소
=============================================================================
"""
import cv2
import numpy as np
import argparse
import os

class ManualAlignGUI:
    def __init__(self, ref_path, mov_path):
        self.ref_path = ref_path
        self.mov_path = mov_path
        
        # Load images
        self.ref_img_full = cv2.imread(ref_path)
        self.mov_img_full = cv2.imread(mov_path)
        
        if self.ref_img_full is None or self.mov_img_full is None:
            raise FileNotFoundError("Could not load images")
            
        self.h_full, self.w_full = self.ref_img_full.shape[:2]
        
        # Display settings
        self.max_display_h = 800
        self.display_scale = 1.0
        if self.h_full > self.max_display_h:
            self.display_scale = self.max_display_h / self.h_full
            
        self.dx = 0.0
        self.dy = 0.0
        self.angle = 0.0 # degree
        self.scale = 1.0 # scale factor
        
        self.show_ref = True
        self.overlay_mode = False
        self.mouse_x = self.w_full // 2
        self.mouse_y = self.h_full // 2
        
        # Window names
        self.win_main = "Manual Align (Q to Save)"
        self.win_zoom = "Zoom (x4)"
        
    def get_display_img(self, img_full):
        """Resize full image to display size"""
        if self.display_scale == 1.0:
            return img_full
        return cv2.resize(img_full, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_AREA)

    def run(self):
        cv2.namedWindow(self.win_main)
        cv2.namedWindow(self.win_zoom)
        cv2.setMouseCallback(self.win_main, self.mouse_callback)
        
        print("\n=== Manual Alignment GUI ===")
        print(f"Ref: {self.ref_path}")
        print(f"Mov: {self.mov_path}")
        print("-" * 30)
        print("WASD: 1px | IJKL: 10px")
        print("Arrow Keys (8426): 0.1px | 7193: 0.01px")
        print("U / O: Rotate 0.1 deg | Y / P: Rotate 0.01 deg")
        print("[ / ]: Scale 0.001    | { / }: Scale 0.01")
        print("SPACE: Toggle View | Z: Overlay | Q: Save | ESC: Quit")
        
        while True:
            # 1. Apply Transformation (Rotation + Scale + Translation)
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
            
            # 2. Prepare Display Images
            if self.overlay_mode:
                display_full = cv2.addWeighted(self.ref_img_full, 0.5, aligned_full, 0.5, 0)
                mode_text = "OVERLAY"
            elif self.show_ref:
                display_full = self.ref_img_full.copy()
                mode_text = "REFERENCE"
            else:
                display_full = aligned_full.copy()
                mode_text = "ALIGNED"
                
            # Resize for Main Window
            display_view = self.get_display_img(display_full)
            
            # Draw UI on Main Window
            view_mouse_x = int(self.mouse_x * self.display_scale)
            view_mouse_y = int(self.mouse_y * self.display_scale)
            
            cv2.drawMarker(display_view, (view_mouse_x, view_mouse_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
            info = f"[{mode_text}] dx={self.dx:.2f} dy={self.dy:.2f} rot={self.angle:.2f} scl={self.scale:.4f}"
            cv2.putText(display_view, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(self.win_main, display_view)
            
            # 3. Prepare Zoom Window
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
            
            # 4. Input Handling
            key = cv2.waitKey(20) & 0xFF
            
            # Translation (WASD, IJKL, Numpad)
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
            elif key == ord('7'): self.dy -= 0.01
            elif key == ord('1'): self.dy += 0.01
            elif key == ord('3'): self.dx -= 0.01
            elif key == ord('9'): self.dx += 0.01
            
            # Rotation (U/O: 0.1, Y/P: 0.01)
            elif key == ord('u'): self.angle += 0.1
            elif key == ord('o'): self.angle -= 0.1
            elif key == ord('y'): self.angle += 0.01
            elif key == ord('p'): self.angle -= 0.01
            
            # Scale ([/]: 0.001, {/}: 0.01)
            elif key == ord('['): self.scale -= 0.001
            elif key == ord(']'): self.scale += 0.001
            elif key == ord('{'): self.scale -= 0.01
            elif key == ord('}'): self.scale += 0.01
            
            # View Control
            elif key == ord(' '):
                self.show_ref = not self.show_ref
                self.overlay_mode = False
            elif key == ord('z'):
                self.overlay_mode = not self.overlay_mode
                
            # Quit
            elif key == ord('q'):
                print(f"Final Offset: dx={self.dx:.4f}, dy={self.dy:.4f}, rot={self.angle:.4f}, scale={self.scale:.5f}")
                cv2.destroyAllWindows()
                return self.dx, self.dy, self.angle, self.scale
            elif key == 27: # ESC
                print("Cancelled.")
                cv2.destroyAllWindows()
                return None, None, None, None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            # Convert display coord to full coord
            self.mouse_x = int(x / self.display_scale)
            self.mouse_y = int(y / self.display_scale)

def main():
    parser = argparse.ArgumentParser(description="Manual Alignment GUI")
    parser.add_argument("--ref", required=False, help="Reference image path")
    parser.add_argument("--mov", required=False, help="Moving image path")
    parser.add_argument("--input-dir", "-i", help="Input directory (uses first 2 images)")
    
    args = parser.parse_args()
    
    ref_path = args.ref
    mov_path = args.mov
    
    if args.input_dir:
        from glob import glob
        images = sorted(glob(os.path.join(args.input_dir, "*.jpg")))
        if len(images) >= 2:
            ref_path = images[0]
            mov_path = images[1]
    
    if not ref_path or not mov_path:
        # Fallback to defaults or error
        print("Please provide --ref/--mov or --input-dir")
        # Try to find default input/output examples for convenience
        if os.path.exists("input") and os.path.exists("output"):
             # Just a hint logic
             pass
        return

    gui = ManualAlignGUI(ref_path, mov_path)
    gui.run()

if __name__ == "__main__":
    main()
