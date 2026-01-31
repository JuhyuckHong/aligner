"""
[ARCHIVED]
Reason: Temporary GUI for visualizing alignment results. Replaced by tools in util/.
Date: 2026-01-31
"""

"""
Grid match 결과 시각화 GUI - 정합 전/후 비교
"""
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Grid match 결과
dx, dy = -1.40, -6.56

# 이미지 로드
img1_path = "output/2026-01-01/2026-01-01_18-00-00.jpg"
img2_path = "output/2026-01-02/2026-01-02_06-00-00.jpg"

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# img2에 offset 적용 (img1에 맞추기)
h, w = img2.shape[:2]
M = np.float32([[1, 0, dx], [0, 1, dy]])
img2_aligned = cv2.warpAffine(img2, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# BGR -> RGB 변환
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2_aligned_rgb = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2RGB)

# 블렌딩 이미지 생성
blend_before = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
blend_after = cv2.addWeighted(img1, 0.5, img2_aligned, 0.5, 0)
blend_before_rgb = cv2.cvtColor(blend_before, cv2.COLOR_BGR2RGB)
blend_after_rgb = cv2.cvtColor(blend_after, cv2.COLOR_BGR2RGB)

# GUI
class AlignCompareGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Grid Match Result: dx={dx:.2f}, dy={dy:.2f}")
        
        # 상태
        self.show_aligned = tk.BooleanVar(value=False)
        self.blend_mode = tk.BooleanVar(value=True)
        self.alpha = tk.DoubleVar(value=0.5)
        
        # 스케일 계산 (화면에 맞게)
        screen_w = root.winfo_screenwidth() - 100
        screen_h = root.winfo_screenheight() - 200
        scale = min(screen_w / w, screen_h / h, 0.5)
        self.disp_w = int(w * scale)
        self.disp_h = int(h * scale)
        
        # 이미지 리사이즈
        self.img1_small = cv2.resize(img1_rgb, (self.disp_w, self.disp_h))
        self.img2_small = cv2.resize(img2_rgb, (self.disp_w, self.disp_h))
        self.img2_aligned_small = cv2.resize(img2_aligned_rgb, (self.disp_w, self.disp_h))
        
        # 프레임
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)
        
        # 토글 버튼들
        tk.Checkbutton(control_frame, text="Aligned (적용 후)", variable=self.show_aligned, 
                       command=self.update_display, font=('Arial', 12)).pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(control_frame, text="Blend Mode", variable=self.blend_mode,
                       command=self.update_display, font=('Arial', 12)).pack(side=tk.LEFT, padx=10)
        
        # 알파 슬라이더
        tk.Label(control_frame, text="Alpha:", font=('Arial', 12)).pack(side=tk.LEFT, padx=5)
        tk.Scale(control_frame, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL,
                 variable=self.alpha, command=lambda x: self.update_display(),
                 length=200).pack(side=tk.LEFT)
        
        # 정보 라벨
        info_frame = tk.Frame(root)
        info_frame.pack(pady=5)
        tk.Label(info_frame, text=f"Image 1 (전날 저녁): {img1_path}", font=('Arial', 10)).pack()
        tk.Label(info_frame, text=f"Image 2 (다음날 아침): {img2_path}", font=('Arial', 10)).pack()
        tk.Label(info_frame, text=f"Grid Match Offset: dx={dx:.2f}px, dy={dy:.2f}px", 
                 font=('Arial', 12, 'bold'), fg='blue').pack()
        
        # 캔버스
        self.canvas = tk.Canvas(root, width=self.disp_w, height=self.disp_h)
        self.canvas.pack(pady=10)
        
        # 키보드 바인딩
        root.bind('<space>', lambda e: self.toggle_aligned())
        root.bind('<b>', lambda e: self.toggle_blend())
        root.bind('<Left>', lambda e: self.adjust_alpha(-0.1))
        root.bind('<Right>', lambda e: self.adjust_alpha(0.1))
        root.bind('<Escape>', lambda e: root.quit())
        
        # 단축키 안내
        shortcut_frame = tk.Frame(root)
        shortcut_frame.pack(pady=5)
        tk.Label(shortcut_frame, text="단축키: Space=정합 토글 | B=블렌드 토글 | ←→=알파 조절 | ESC=종료",
                 font=('Arial', 10), fg='gray').pack()
        
        self.update_display()
    
    def toggle_aligned(self):
        self.show_aligned.set(not self.show_aligned.get())
        self.update_display()
    
    def toggle_blend(self):
        self.blend_mode.set(not self.blend_mode.get())
        self.update_display()
    
    def adjust_alpha(self, delta):
        new_alpha = max(0, min(1, self.alpha.get() + delta))
        self.alpha.set(new_alpha)
        self.update_display()
    
    def update_display(self):
        alpha = self.alpha.get()
        aligned = self.show_aligned.get()
        blend = self.blend_mode.get()
        
        img2_current = self.img2_aligned_small if aligned else self.img2_small
        
        if blend:
            # 블렌딩
            blended = cv2.addWeighted(self.img1_small, alpha, img2_current, 1-alpha, 0)
            display_img = blended
        else:
            # 좌우 반반
            half_w = self.disp_w // 2
            display_img = np.zeros_like(self.img1_small)
            display_img[:, :half_w] = self.img1_small[:, :half_w]
            display_img[:, half_w:] = img2_current[:, half_w:]
            # 중앙선
            cv2.line(display_img, (half_w, 0), (half_w, self.disp_h), (255, 0, 0), 2)
        
        # 상태 표시
        status = "ALIGNED" if aligned else "BEFORE"
        cv2.putText(display_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 0) if aligned else (255, 0, 0), 2)
        
        # Tkinter 이미지로 변환
        img_pil = Image.fromarray(display_img)
        self.photo = ImageTk.PhotoImage(img_pil)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

# 실행
root = tk.Tk()
app = AlignCompareGUI(root)
root.mainloop()
