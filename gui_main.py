"""
gui_main.py - Main GUI for Aligner Pro
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import threading
import shutil
import numpy as np
from PIL import Image, ImageTk

# Import Core Logic
import stabilize_parallel as stabilizer
from gui_visualizer import ManualAlignVisualizer
from multiprocessing import freeze_support, Pool, cpu_count

class AlignerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Antigravity Aligner Pro")
        self.root.geometry("1200x800")
        
        # Data State
        self.folder_analyses = {}       # {folder: [results]}
        self.day_refine_targets = {}    # {folder: (dx, dy)}
        self.sorted_folders = []        # [folder1, folder2, ...]
        self.current_transition_idx = -1
        
        # Styling
        style = ttk.Style()
        style.theme_use('clam')
        
        # Layout
        self.create_top_panel()
        self.create_main_panel()
        self.create_bottom_panel()
        
    def create_top_panel(self):
        self.top_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        self.top_frame.pack(fill="x", padx=10, pady=5)
        
        # Input
        ttk.Label(self.top_frame, text="Input Folder:").grid(row=0, column=0, sticky="w")
        self.entry_input = ttk.Entry(self.top_frame, width=60)
        self.entry_input.grid(row=0, column=1, padx=5)
        self.entry_input.insert(0, os.path.abspath("input"))
        ttk.Button(self.top_frame, text="Browse", command=self.browse_input).grid(row=0, column=2)
        
        # Output
        ttk.Label(self.top_frame, text="Output Folder:").grid(row=1, column=0, sticky="w")
        self.entry_output = ttk.Entry(self.top_frame, width=60)
        self.entry_output.grid(row=1, column=1, padx=5)
        self.entry_output.insert(0, os.path.abspath("output"))
        ttk.Button(self.top_frame, text="Browse", command=self.browse_output).grid(row=1, column=2)
        
        # Workers
        ttk.Label(self.top_frame, text="Workers:").grid(row=0, column=3, padx=20)
        self.var_workers = tk.IntVar(value=max(1, cpu_count()-1))
        self.scale_workers = tk.Scale(self.top_frame, from_=1, to=16, orient="horizontal", variable=self.var_workers)
        self.scale_workers.grid(row=0, column=4)

    def browse_input(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_input.delete(0, tk.END)
            self.entry_input.insert(0, path)

    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, path)

    def create_main_panel(self):
        self.paned = ttk.PanedWindow(self.root, orient="horizontal")
        self.paned.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Left: Transitions List
        self.left_frame = ttk.Frame(self.paned, width=400)
        self.paned.add(self.left_frame, weight=1)
        
        columns = ("transition", "offset", "status")
        self.tree = ttk.Treeview(self.left_frame, columns=columns, show="headings")
        self.tree.heading("transition", text="Transition (Day -> Day)")
        self.tree.heading("offset", text="Offset (dx, dy)")
        self.tree.heading("status", text="Status")
        self.tree.column("transition", width=250)
        self.tree.column("offset", width=100)
        self.tree.column("status", width=100)
        
        self.tree.bind("<<TreeviewSelect>>", self.on_select_transition)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.left_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Right: Preview Pane
        self.right_frame = ttk.LabelFrame(self.paned, text="Alignment Verify", padding=10, width=400)
        self.paned.add(self.right_frame, weight=1)
        
        self.lbl_preview_title = ttk.Label(self.right_frame, text="Select a transition to preview", font=("Arial", 12))
        self.lbl_preview_title.pack(pady=5)
        
        # Image Label
        self.lbl_image = ttk.Label(self.right_frame)
        self.lbl_image.pack(pady=10)
        
        # Controls
        self.btn_edit = ttk.Button(self.right_frame, text="Edit Alignment Manually", command=self.open_visualizer, state="disabled")
        self.btn_edit.pack(pady=10)

    def create_bottom_panel(self):
        self.bottom_frame = ttk.Frame(self.root, padding=10)
        self.bottom_frame.pack(fill="x")
        
        self.btn_analyze = ttk.Button(self.bottom_frame, text="Step 1: Analyze Motion", command=self.run_analysis_thread)
        self.btn_analyze.pack(side="left", padx=5)
        
        self.btn_render = ttk.Button(self.bottom_frame, text="Step 2: Render Frames", command=self.run_render_thread, state="disabled")
        self.btn_render.pack(side="left", padx=5)
        
        self.lbl_status = ttk.Label(self.bottom_frame, text="Ready.")
        self.lbl_status.pack(side="right", padx=10)
        
    # --- Logic ---
    
    def log(self, msg):
        self.lbl_status.config(text=msg)
        self.root.update_idletasks()
        
    def run_analysis_thread(self):
        self.btn_analyze.config(state="disabled")
        self.btn_render.config(state="disabled")
        self.tree.delete(*self.tree.get_children())
        self.folder_analyses = {}
        self.sorted_folders = []
        
        threading.Thread(target=self.run_analysis, daemon=True).start()
        
    def run_analysis(self):
        input_dir = self.entry_input.get()
        ext = "jpg"
        workers = self.var_workers.get()
        
        try:
            self.log("Scanning folders...")
            subfolders = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
            if not subfolders:
                self.log("No subfolders found!")
                return
                
            tasks = [(os.path.join(input_dir, d), ext) for d in subfolders]
            
            self.log(f"Analyzing {len(tasks)} folders...")
            
            # Using pool inside thread requires care, but stabilizer logic uses Pool.
            # We will use multiprocessing pool provided by stabilizer logic logic in main thread usually, 
            # here we call it directly. Note: Tkinter is not thread-safe, update UI via invoke.
            
            results_map = {}
            with Pool(workers) as pool:
                # We can't use tqdm easily here, just map
                for i, (folder_name, results) in enumerate(pool.imap(stabilizer.analyze_folder_worker, tasks)):
                    results_map[folder_name] = results
                    self.root.after(0, self.log, f"Analyzed {i+1}/{len(tasks)}: {os.path.basename(folder_name)}")
                    
            self.folder_analyses = results_map
            
            # Initial Gap Calculation
            self.root.after(0, self.log, "Calculating Gaps...")
            self.day_refine_targets = stabilizer.calculate_day_gaps(self.folder_analyses)
            self.sorted_folders = sorted(self.folder_analyses.keys())
            
            self.root.after(0, self.populate_tree)
            self.root.after(0, lambda: self.btn_render.config(state="normal"))
            self.root.after(0, lambda: self.btn_analyze.config(state="normal"))
            self.root.after(0, self.log, "Analysis Complete. Please Review.")
            
        except Exception as e:
            self.root.after(0, self.log, f"Error: {str(e)}")
            self.root.after(0, lambda: self.btn_analyze.config(state="normal"))

    def populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        
        days = self.sorted_folders
        # targets keyed by day2
        
        for i in range(len(days)-1):
            day1 = days[i]
            day2 = days[i+1]
            
            tgt = self.day_refine_targets.get(day2, (0,0))
            name1 = os.path.basename(day1)
            name2 = os.path.basename(day2)
            
            self.tree.insert("", "end", iid=str(i), values=(
                f"{name1} -> {name2}",
                f"{tgt[0]:.1f}, {tgt[1]:.1f}",
                "Auto"
            ))

    def on_select_transition(self, event):
        sel = self.tree.selection()
        if not sel: return
        
        idx = int(sel[0])
        self.current_transition_idx = idx
        self.btn_edit.config(state="normal")
        
        self.update_preview(idx)
        
    def update_preview(self, idx):
        day1 = self.sorted_folders[idx]
        day2 = self.sorted_folders[idx+1]
        
        # Get noon samples
        s1_list = stabilizer.get_noon_samples(self.folder_analyses[day1], n_samples=1)
        s2_list = stabilizer.get_noon_samples(self.folder_analyses[day2], n_samples=1)
        
        if not s1_list or not s2_list:
            self.lbl_preview_title.config(text="No noon images found")
            return
            
        p1 = s1_list[0]
        p2 = s2_list[0]
        
        img1 = cv2.imread(p1)
        img2 = cv2.imread(p2)
        
        # Current Offset for Day 2
        # Note: day_refine_targets stores 'cumulative' offset for day2.
        # But we want to show the relative alignment between Day 1 and Day 2.
        # Day 1 also has a cumulative offset.
        # RELATIVE GAP = Target(Day2) - Target(Day1)
        
        t1 = self.day_refine_targets.get(day1, (0,0))
        t2 = self.day_refine_targets.get(day2, (0,0))
        
        rel_dx = t2[0] - t1[0]
        rel_dy = t2[1] - t1[1]
        
        self.lbl_preview_title.config(text=f"Relative Shift: {rel_dx:.1f}, {rel_dy:.1f}")
        
        # Generate Overlay
        # Shift img2 by rel_dx, rel_dy
        h, w = img1.shape[:2]
        M = np.float32([[1, 0, rel_dx], [0, 1, rel_dy]])
        img2_shifted = cv2.warpAffine(img2, M, (w, h))
        
        # Smart Crop (Center 300x300)
        cw, ch = 300, 300
        cx, cy = w//2, h//2
        x1 = max(0, cx - cw//2)
        y1 = max(0, cy - ch//2)
        x2 = min(w, x1 + cw)
        y2 = min(h, y1 + ch)
        
        crop1 = img1[y1:y2, x1:x2]
        crop2 = img2_shifted[y1:y2, x1:x2]
        
        # Blend
        blend = cv2.addWeighted(crop1, 0.6, crop2, 0.4, 0)
        
        # Convert to TK
        blend = cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(blend)
        im_tk = ImageTk.PhotoImage(im_pil)
        
        self.lbl_image.configure(image=im_tk)
        self.lbl_image.image = im_tk # Keep ref

    def open_visualizer(self):
        if self.current_transition_idx < 0: return
        
        idx = self.current_transition_idx
        day1 = self.sorted_folders[idx]
        day2 = self.sorted_folders[idx+1]
        
        s1 = stabilizer.get_noon_samples(self.folder_analyses[day1], n_samples=1)[0]
        s2 = stabilizer.get_noon_samples(self.folder_analyses[day2], n_samples=1)[0]
        
        # Calculate current relative offset
        t1 = self.day_refine_targets.get(day1, (0,0))
        t2 = self.day_refine_targets.get(day2, (0,0))
        rel_dx = t2[0] - t1[0]
        rel_dy = t2[1] - t1[1]
        
        vis = ManualAlignVisualizer(s1, s2, initial_dx=rel_dx, initial_dy=rel_dy)
        new_dx, new_dy = vis.run()
        
        if new_dx is not None:
            # Update Target
            # New Target(Day2) = Target(Day1) + NewRelative
            updated_t2_dx = t1[0] + new_dx
            updated_t2_dy = t1[1] + new_dy
            
            self.day_refine_targets[day2] = (updated_t2_dx, updated_t2_dy)
            
            # Propagate change to future days?
            # Yes, if we shift Day 2, Day 3's relative position to Day 2 stays same, 
            # so Day 3's absolute target must shift by the delta.
            delta_dx = updated_t2_dx - t2[0]
            delta_dy = updated_t2_dy - t2[1]
            
            for k in range(idx + 2, len(self.sorted_folders)):
                future_day = self.sorted_folders[k]
                old_tgt = self.day_refine_targets[future_day]
                self.day_refine_targets[future_day] = (old_tgt[0] + delta_dx, old_tgt[1] + delta_dy)
            
            # Refresh UI
            self.populate_tree()
            self.tree.selection_set(str(idx)) # Keep selection
            self.item_set_status(idx, "Manual")
            self.update_preview(idx)

    def item_set_status(self, idx, status):
        # Helper to update tree item status
        current = self.tree.item(str(idx), "values")
        self.tree.item(str(idx), values=(current[0], current[1], status))

    def run_render_thread(self):
        self.btn_render.config(state="disabled")
        self.btn_analyze.config(state="disabled")
        threading.Thread(target=self.run_render, daemon=True).start()
        
    def run_render(self):
        try:
            self.log("Integrating trajectory...")
            global_traj = stabilizer.integrate_trajectory(self.folder_analyses, self.day_refine_targets)
            
            output_dir = self.entry_output.get()
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            
            # Render Tasks
            subfolders = self.sorted_folders
            render_tasks = []
            for folder in subfolders:
                render_tasks.append((
                    folder,  # Input full path
                    os.path.join(output_dir, os.path.basename(folder)), # Output full path
                    global_traj[folder]
                ))
            
            workers = self.var_workers.get()
            self.log("Rendering frames...")
            
            with Pool(workers) as pool:
                for i, _ in enumerate(pool.imap(stabilizer.render_folder_worker, render_tasks)):
                    self.root.after(0, self.log, f"Rendered dataset {i+1}/{len(render_tasks)}")
                    
            self.root.after(0, self.log, "Rendering Complete! Check output folder.")
            self.root.after(0, lambda: self.btn_render.config(state="normal"))
            self.root.after(0, lambda: self.btn_analyze.config(state="normal"))
            messagebox.showinfo("Success", "Rendering Finished!")
            
        except Exception as e:
            self.root.after(0, self.log, f"Error: {str(e)}")
            self.root.after(0, lambda: self.btn_render.config(state="normal"))

if __name__ == "__main__":
    freeze_support()
    root = tk.Tk()
    app = AlignerApp(root)
    root.mainloop()
