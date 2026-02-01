"""
gui_main.py - Main GUI for Timelapse Aligner Pro
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
import timelapse_stabilizer as stabilizer
from gui_visualizer import ManualAlignVisualizer
from multiprocessing import freeze_support, Pool, cpu_count

class AlignerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Timelapse Aligner Pro")
        self.root.geometry("1400x900")
        
        # Data State
        self.input_structure = {}       # {dataset_name: [full_path_images]}
        self.dataset_paths = {}         # {dataset_name: full_path_to_folder}
        self.dataset_is_root = False    # True if dataset_name is just "root" (single folder mode)
        
        self.excluded_files = set()     # Set of absolute paths
        
        self.folder_analyses = {}       # {folder: [results]}
        self.day_gaps = {}              # {folder: (dx, dy)}
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
        self.entry_input.bind("<FocusOut>", self.on_input_change)
        
        ttk.Button(self.top_frame, text="Browse", command=self.browse_input).grid(row=0, column=2)
        ttk.Button(self.top_frame, text="Load / Scan Files", command=self.scan_input_structure).grid(row=0, column=3, padx=5)
        
        # Output
        ttk.Label(self.top_frame, text="Output Folder:").grid(row=1, column=0, sticky="w")
        self.entry_output = ttk.Entry(self.top_frame, width=60)
        self.entry_output.grid(row=1, column=1, padx=5)
        self.entry_output.insert(0, os.path.abspath("output"))
        ttk.Button(self.top_frame, text="Browse", command=self.browse_output).grid(row=1, column=2)
        
        # Workers
        ttk.Label(self.top_frame, text="Workers:").grid(row=1, column=3, padx=20)
        self.var_workers = tk.IntVar(value=max(1, cpu_count()-1))
        self.scale_workers = tk.Scale(self.top_frame, from_=1, to=16, orient="horizontal", variable=self.var_workers)
        self.scale_workers.grid(row=1, column=4)

    def browse_input(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_input.delete(0, tk.END)
            self.entry_input.insert(0, path)
            self.scan_input_structure()
            
    def on_input_change(self, event=None):
        self.scan_input_structure()

    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, path)

    def create_main_panel(self):
        self.paned = ttk.PanedWindow(self.root, orient="horizontal")
        self.paned.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Left Panel (Tabs)
        self.left_tabs = ttk.Notebook(self.paned, width=450)
        self.paned.add(self.left_tabs, weight=1)
        
        # Tab 1: Source Files
        self.tab_files = ttk.Frame(self.left_tabs)
        self.left_tabs.add(self.tab_files, text="1. Source Files")
        
        self.tree_files = ttk.Treeview(self.tab_files, columns=("status", "size"), show="tree headings")
        self.tree_files.heading("#0", text="Folder / File")
        self.tree_files.heading("status", text="Status")
        self.tree_files.heading("size", text="Size")
        self.tree_files.column("#0", width=250)
        self.tree_files.column("status", width=80)
        self.tree_files.column("size", width=80)
        
        # Scroll
        scroll_files = ttk.Scrollbar(self.tab_files, orient="vertical", command=self.tree_files.yview)
        self.tree_files.configure(yscrollcommand=scroll_files.set)
        
        self.tree_files.pack(side="left", fill="both", expand=True)
        scroll_files.pack(side="right", fill="y")
        
        self.tree_files.bind("<<TreeviewSelect>>", self.on_file_select)
        self.tree_files.bind("<Double-1>", self.on_file_toggle)

        # Tab 2: Transitions
        self.tab_transitions = ttk.Frame(self.left_tabs)
        self.left_tabs.add(self.tab_transitions, text="2. Transitions (Analysis)")
        
        columns = ("transition", "offset", "status")
        self.tree_trans = ttk.Treeview(self.tab_transitions, columns=columns, show="headings")
        self.tree_trans.heading("transition", text="Transition")
        self.tree_trans.heading("offset", text="Offset")
        self.tree_trans.heading("status", text="Status")
        self.tree_trans.column("transition", width=200)
        self.tree_trans.column("offset", width=100)
        self.tree_trans.column("status", width=80)
        
        scroll_trans = ttk.Scrollbar(self.tab_transitions, orient="vertical", command=self.tree_trans.yview)
        self.tree_trans.configure(yscrollcommand=scroll_trans.set)
        
        self.tree_trans.pack(side="left", fill="both", expand=True)
        scroll_trans.pack(side="right", fill="y")
        
        self.tree_trans.bind("<<TreeviewSelect>>", self.on_select_transition)
        
        # Right: Preview Pane
        self.right_frame = ttk.LabelFrame(self.paned, text="Preview & Verify", padding=10, width=500)
        self.paned.add(self.right_frame, weight=1)
        
        self.lbl_preview_title = ttk.Label(self.right_frame, text="Select a file or transition", font=("Arial", 12))
        self.lbl_preview_title.pack(pady=5)
        
        self.lbl_image = ttk.Label(self.right_frame)
        self.lbl_image.pack(pady=5, expand=True)
        
        # Controls Frame
        self.ctrl_frame = ttk.Frame(self.right_frame)
        self.ctrl_frame.pack(fill="x", pady=10)
        
        self.btn_exclude = ttk.Button(self.ctrl_frame, text="Toggle Exclude File", command=self.toggle_current_file, state="disabled")
        self.btn_exclude.pack(side="left", padx=5)
        
        self.btn_edit_align = ttk.Button(self.ctrl_frame, text="Edit Alignment Manually", command=self.open_visualizer, state="disabled")
        self.btn_edit_align.pack(side="right", padx=5)

    def create_bottom_panel(self):
        self.bottom_frame = ttk.Frame(self.root, padding=10)
        self.bottom_frame.pack(fill="x")
        
        self.btn_analyze = ttk.Button(self.bottom_frame, text="Step 1: Analyze Motion", command=self.run_analysis_thread)
        self.btn_analyze.pack(side="left", padx=5)
        
        self.btn_render = ttk.Button(self.bottom_frame, text="Step 2: Render Frames", command=self.run_render_thread, state="disabled")
        self.btn_render.pack(side="left", padx=5)
        
        self.lbl_status = ttk.Label(self.bottom_frame, text="Ready.")
        self.lbl_status.pack(side="right", padx=10)
        
    # --- Logic: File Scanning ---
    
    def log(self, msg):
        self.lbl_status.config(text=msg)
        self.root.update_idletasks()
        
    def scan_input_structure(self):
        input_dir = self.entry_input.get()
        if not os.path.exists(input_dir):
            self.log("Input directory does not exist.")
            return

        self.log("Scanning files...")
        self.tree_files.delete(*self.tree_files.get_children())
        self.input_structure = {}
        self.dataset_paths = {}
        self.excluded_files = set() # Reset exclusions on new scan? Or keep matching names? Reset for safety.
        
        ext = "jpg"
        
        # 1. Check subfolders
        subfolders = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
        
        if subfolders:
            self.dataset_is_root = False
            total_imgs = 0
            for d in subfolders:
                full_d = os.path.join(input_dir, d)
                imgs = stabilizer.get_images(full_d, ext)
                if imgs:
                    self.input_structure[d] = imgs
                    self.dataset_paths[d] = full_d
                    total_imgs += len(imgs)
                    
                    # Tree Node
                    folder_node = self.tree_files.insert("", "end", text=d, values=("Dataset", f"{len(imgs)} files"))
                    # Add files
                    for img in imgs:
                        fname = os.path.basename(img)
                        size_mb = os.path.getsize(img) / (1024*1024)
                        self.tree_files.insert(folder_node, "end", text=fname, values=("Active", f"{size_mb:.1f} MB"), tags=("file",))
                        
            self.log(f"Found {len(self.input_structure)} datasets ({total_imgs} images).")
            
        else:
            # 2. Check root
            imgs = stabilizer.get_images(input_dir, ext)
            if imgs:
                self.dataset_is_root = True
                d_name = os.path.basename(input_dir)
                self.input_structure[d_name] = imgs
                self.dataset_paths[d_name] = input_dir
                
                # Tree Node
                # For single dataset, list files directly or under a single root node?
                # Using a single root node is cleaner.
                folder_node = self.tree_files.insert("", "end", text=d_name, values=("Single Set", f"{len(imgs)} files"), open=True)
                for img in imgs:
                    fname = os.path.basename(img)
                    size_mb = os.path.getsize(img) / (1024*1024)
                    self.tree_files.insert(folder_node, "end", text=fname, values=("Active", f"{size_mb:.1f} MB"), tags=("file",))
                    
                self.log(f"Found {len(imgs)} images in root.")
            else:
                self.log("No images found.")
                
    def get_selected_file_path(self):
        # Returns absolute path of selected file item
        sel = self.tree_files.selection()
        if not sel: return None
        item_id = sel[0]
        item = self.tree_files.item(item_id)
        
        if "file" not in self.tree_files.item(item_id, "tags"):
            return None # Selected a folder
            
        fname = item['text']
        parent_id = self.tree_files.parent(item_id)
        parent_name = self.tree_files.item(parent_id)['text']
        
        # Find in structure (scan datasets)
        # Note: parent_name matches d_name in input_structure keys
        # But if dataset_is_root, input_structure key matches parent_name
        
        if parent_name in self.input_structure:
             # Find full path ending with fname
             for path in self.input_structure[parent_name]:
                 if os.path.basename(path) == fname:
                     return path
        return None

    def on_file_select(self, event):
        path = self.get_selected_file_path()
        if path:
            self.show_preview_image(path)
            # Update exclude button
            if path in self.excluded_files:
                self.btn_exclude.config(text="Restore File", state="normal")
            else:
                self.btn_exclude.config(text="Exclude File", state="normal")
        else:
            self.btn_exclude.config(state="disabled")
            
    def on_file_toggle(self, event):
        self.toggle_current_file()
        
    def toggle_current_file(self):
        path = self.get_selected_file_path()
        if not path: return
        
        sel = self.tree_files.selection()[0]
        
        if path in self.excluded_files:
            self.excluded_files.remove(path)
            self.tree_files.item(sel, tags=("file",)) # Default style
            self.tree_files.set(sel, "status", "Active")
            self.btn_exclude.config(text="Exclude File")
        else:
            self.excluded_files.add(path)
            self.tree_files.item(sel, tags=("file", "excluded"))
            self.tree_files.set(sel, "status", "Excluded")
            self.btn_exclude.config(text="Restore File")
            
        # Add visual style for excluded
        self.tree_files.tag_configure("excluded", foreground="gray", font=("Arial", 9, "overstrike"))
        self.tree_files.tag_configure("file", foreground="black", font=("Arial", 9))

    def show_preview_image(self, path):
        if not os.path.exists(path): return
        
        # Load and resize for preview
        img = cv2.imread(path)
        if img is None: return
        
        h, w = img.shape[:2]
        display_w = 480
        scale = display_w / w
        display_h = int(h * scale)
        
        small = cv2.resize(img, (display_w, display_h))
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        im_tk = ImageTk.PhotoImage(Image.fromarray(small))
        
        self.lbl_image.configure(image=im_tk)
        self.lbl_image.image = im_tk
        self.lbl_preview_title.config(text=os.path.basename(path))

    # --- Logic: Analysis ---

    def run_analysis_thread(self):
        if not self.input_structure:
            messagebox.showerror("Error", "No files loaded. Please click 'Load / Scan Files' first.")
            return
            
        self.btn_analyze.config(state="disabled")
        self.btn_render.config(state="disabled")
        self.tree_trans.delete(*self.tree_trans.get_children())
        self.folder_analyses = {}
        
        threading.Thread(target=self.run_analysis, daemon=True).start()
        
    def run_analysis(self):
        workers = self.var_workers.get()
        
        # Prepare Tasks with filtered lists
        tasks = []
        
        for d_name, files in self.input_structure.items():
            dataset_path = self.dataset_paths[d_name]
            # Filter
            valid_files = [f for f in files if f not in self.excluded_files]
            
            if len(valid_files) < 2:
                self.root.after(0, self.log, f"Skipping {d_name} (Not enough files)")
                continue
                
            tasks.append((dataset_path, valid_files))
            
        if not tasks:
            self.root.after(0, self.log, "No valid datasets to analyze.")
            self.root.after(0, lambda: self.btn_analyze.config(state="normal"))
            return

        self.root.after(0, lambda: self.left_tabs.select(self.tab_transitions))
        self.root.after(0, self.log, f"Analyzing {len(tasks)} datasets...")
        
        try:
            results_map = {}
            with Pool(workers) as pool:
                for i, (folder_name, results) in enumerate(pool.imap(stabilizer.analyze_folder_worker, tasks)):
                    results_map[folder_name] = results
                    self.root.after(0, self.log, f"Analyzed {i+1}/{len(tasks)}: {os.path.basename(folder_name)}")
                    
            self.folder_analyses = results_map
            self.sorted_folders = sorted(self.folder_analyses.keys())
            
            # Phase 2: Refinement
            self.root.after(0, self.log, "Measuring Day Gaps...")
            
            refine_tasks = []
            for i in range(len(self.sorted_folders)-1):
                day1 = self.sorted_folders[i]
                day2 = self.sorted_folders[i+1]
                s1 = stabilizer.get_noon_samples_with_acc(self.folder_analyses[day1])
                s2 = stabilizer.get_noon_samples_with_acc(self.folder_analyses[day2])
                refine_tasks.append((day1, day2, s1, s2))
                
            day_gaps = {}
            if refine_tasks:
                with Pool(workers) as pool:
                    for i, (day2, gap) in enumerate(pool.imap(stabilizer.measure_day_gap_worker, refine_tasks)):
                        day_gaps[day2] = gap
                        self.root.after(0, self.log, f"Measured gap for {os.path.basename(day2)}")
            
            self.day_gaps = day_gaps
            
            # UI
            self.root.after(0, self.populate_tree)
            self.root.after(0, lambda: self.btn_render.config(state="normal"))
            self.root.after(0, lambda: self.btn_analyze.config(state="normal"))
            self.root.after(0, self.log, "Analysis Complete.")

        except Exception as e:
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, lambda: self.btn_analyze.config(state="normal"))

    def populate_tree(self):
        self.tree_trans.delete(*self.tree_trans.get_children())
        days = self.sorted_folders
        for i in range(len(days)-1):
            day1 = days[i]
            day2 = days[i+1]
            gap = self.day_gaps.get(day2, (0.0, 0.0))
            name1 = os.path.basename(day1)
            name2 = os.path.basename(day2)
            self.tree_trans.insert("", "end", iid=str(i), values=(
                f"{name1} -> {name2}",
                f"{gap[0]:.1f}, {gap[1]:.1f}",
                "Auto"
            ))

    def on_select_transition(self, event):
        # When selecting transition, switch preview to alignment mode
        sel = self.tree_trans.selection()
        if not sel: return
        idx = int(sel[0])
        self.current_transition_idx = idx
        self.btn_edit_align.config(state="normal")
        self.update_align_preview(idx)

    def update_align_preview(self, idx):
        day1 = self.sorted_folders[idx]
        day2 = self.sorted_folders[idx+1]
        
        s1 = stabilizer.get_noon_samples_with_acc(self.folder_analyses[day1], n_samples=1)
        s2 = stabilizer.get_noon_samples_with_acc(self.folder_analyses[day2], n_samples=1)
        
        if not s1 or not s2: return
        
        samp1 = s1[0]
        samp2 = s2[0]
        img1 = cv2.imread(samp1['abs_path'])
        img2 = cv2.imread(samp2['abs_path'])
        
        gap = self.day_gaps.get(day2, (0.0, 0.0))
        rel_dx, rel_dy = gap
        
        self.lbl_preview_title.config(text=f"Transition: {os.path.basename(day1)} -> {os.path.basename(day2)}")
        
        h, w = img1.shape[:2]
        M = np.float32([[1, 0, rel_dx], [0, 1, rel_dy]])
        img2_shifted = cv2.warpAffine(img2, M, (w, h))
        
        crop_sz = 300
        cx, cy = w//2, h//2
        crop1 = img1[cy-crop_sz//2:cy+crop_sz//2, cx-crop_sz//2:cx+crop_sz//2]
        crop2 = img2_shifted[cy-crop_sz//2:cy+crop_sz//2, cx-crop_sz//2:cx+crop_sz//2]
        
        if crop1.size == 0 or crop2.size == 0:
             # Fallback if image too small
             crop1 = cv2.resize(img1, (crop_sz, crop_sz))
             crop2 = cv2.resize(img2_shifted, (crop_sz, crop_sz))

        blend = cv2.addWeighted(crop1, 0.6, crop2, 0.4, 0)
        blend = cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)
        im_tk = ImageTk.PhotoImage(Image.fromarray(blend))
        
        self.lbl_image.configure(image=im_tk)
        self.lbl_image.image = im_tk

    def open_visualizer(self):
        if self.current_transition_idx < 0: return
        idx = self.current_transition_idx
        day1 = self.sorted_folders[idx]
        day2 = self.sorted_folders[idx+1]
        s1 = stabilizer.get_noon_samples_with_acc(self.folder_analyses[day1], n_samples=1)[0]
        s2 = stabilizer.get_noon_samples_with_acc(self.folder_analyses[day2], n_samples=1)[0]
        gap = self.day_gaps.get(day2, (0.0, 0.0))
        
        vis = ManualAlignVisualizer(s1['abs_path'], s2['abs_path'], initial_dx=gap[0], initial_dy=gap[1])
        new_dx, new_dy = vis.run()
        
        if new_dx is not None:
            self.day_gaps[day2] = (new_dx, new_dy)
            self.populate_tree()
            self.tree_trans.selection_set(str(idx))
            self.update_align_preview(idx)

    def run_render_thread(self):
        self.btn_render.config(state="disabled")
        self.btn_analyze.config(state="disabled")
        threading.Thread(target=self.run_render, daemon=True).start()
        
    def run_render(self):
        try:
            self.log("Integrating trajectory...")
            global_traj = stabilizer.integrate_trajectory(self.folder_analyses, self.day_gaps)
            
            output_dir = self.entry_output.get()
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            
            render_tasks = []
            for folder in self.sorted_folders:
                render_tasks.append((
                    folder,
                    os.path.join(output_dir, os.path.basename(folder)),
                    global_traj[folder]
                ))
            
            workers = self.var_workers.get()
            self.log("Rendering filters...")
            with Pool(workers) as pool:
                for i, _ in enumerate(pool.imap(stabilizer.render_folder_worker, render_tasks)):
                    self.root.after(0, self.log, f"Rendered {i+1}/{len(render_tasks)}")
                    
            self.root.after(0, self.log, "Rendering Complete!")
            self.root.after(0, lambda: self.btn_render.config(state="normal"))
            self.root.after(0, lambda: self.btn_analyze.config(state="normal"))
            messagebox.showinfo("Success", "Finished!")
            
        except Exception as e:
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, lambda: self.btn_render.config(state="normal"))

if __name__ == "__main__":
    freeze_support()
    root = tk.Tk()
    app = AlignerApp(root)
    root.mainloop()
