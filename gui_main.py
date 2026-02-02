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
        self.create_action_panel()
        self.create_main_panel()
        
    def create_top_panel(self):
        self.top_frame = ttk.LabelFrame(self.root, text="기본 설정 (Configuration)", padding=10)
        self.top_frame.pack(fill="x", padx=10, pady=5)
        
        # Input
        ttk.Label(self.top_frame, text="원본 사진 폴더:").grid(row=0, column=0, sticky="w")
        self.entry_input = ttk.Entry(self.top_frame, width=60)
        self.entry_input.grid(row=0, column=1, padx=5)
        self.entry_input.insert(0, r"\\Buildmotion")
        self.entry_input.bind("<FocusOut>", self.on_input_change)
        
        ttk.Button(self.top_frame, text="폴더 찾기", command=self.browse_input).grid(row=0, column=2)
        ttk.Button(self.top_frame, text="파일 불러오기", command=self.scan_input_structure).grid(row=0, column=3, padx=5)
        
        # Output
        ttk.Label(self.top_frame, text="저장될 폴더:").grid(row=1, column=0, sticky="w")
        self.entry_output = ttk.Entry(self.top_frame, width=60)
        self.entry_output.grid(row=1, column=1, padx=5)
        self.entry_output.insert(0, r"\\Buildmotion\NAS_LOG")
        ttk.Button(self.top_frame, text="폴더 찾기", command=self.browse_output).grid(row=1, column=2)
        
        # Workers
        max_cpu = cpu_count()
        ttk.Label(self.top_frame, text="작업 프로세서 수:").grid(row=1, column=3, padx=10, sticky="e")
        
        self.var_workers = tk.IntVar(value=max(1, max_cpu-1))
        
        # Dynamic Label for (Val / Max)
        self.lbl_workers_val = ttk.Label(self.top_frame, text=f"({self.var_workers.get()}/{max_cpu})")
        self.lbl_workers_val.grid(row=1, column=5, padx=5, sticky="w")

        def update_worker_label(val):
            self.lbl_workers_val.config(text=f"({int(float(val))}/{max_cpu})")

        self.scale_workers = tk.Scale(self.top_frame, from_=1, to=max_cpu, orient="horizontal", 
                                      variable=self.var_workers, command=update_worker_label, showvalue=0)
        self.scale_workers.grid(row=1, column=4)

    def browse_input(self):
        current = self.entry_input.get()
        initialdir = current if os.path.exists(current) else None
        path = filedialog.askdirectory(initialdir=initialdir)
        if path:
            self.entry_input.delete(0, tk.END)
            self.entry_input.insert(0, path)
            self.scan_input_structure()

    def on_input_change(self, event=None):
        self.scan_input_structure()

    def browse_output(self):
        current = self.entry_output.get()
        initialdir = current if os.path.exists(current) else None
        path = filedialog.askdirectory(initialdir=initialdir)
        if path:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, path)

    def create_action_panel(self):
        self.action_frame = ttk.Frame(self.root, padding=10, relief="raised", borderwidth=1)
        self.action_frame.pack(fill="x", padx=10, pady=5)
        
        self.btn_run_all = ttk.Button(self.action_frame, text="▶ 전체 자동 실행 (1~3단계)", command=self.run_all_steps)
        self.btn_run_all.pack(side="left", padx=5)
        
        ttk.Separator(self.action_frame, orient="vertical").pack(side="left", fill="y", padx=10)
        
        self.btn_analyze = ttk.Button(self.action_frame, text="1단계: 움직임 분석", command=self.run_analysis_thread)
        self.btn_analyze.pack(side="left", padx=5)
        
        self.btn_render = ttk.Button(self.action_frame, text="2단계: 이미지 보정", command=self.run_render_thread, state="disabled")
        self.btn_render.pack(side="left", padx=5)
        
        self.btn_video = ttk.Button(self.action_frame, text="3단계: 비디오 생성", command=self.run_video_thread, state="disabled")
        self.btn_video.pack(side="left", padx=5)
        
        # Progress Bar
        self.progress = ttk.Progressbar(self.action_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(side="left", padx=20, fill="x", expand=True)
        
        self.lbl_status = ttk.Label(self.action_frame, text="준비됨.", font=("Arial", 10, "bold"))
        self.lbl_status.pack(side="right", padx=10)

    # --- Run All Logic ---
    def run_all_steps(self):
        # Start Step 1, with callback to Step 2
        self.run_analysis_thread(on_success=self.trigger_step_2)
        
    def trigger_step_2(self):
        # Start Step 2, with callback to Step 3
        # We need to wait a bit or just call it? Thread is done.
        # But we need to use root.after to be safe in main thread? 
        # run_analysis calls on_success via root.after? No I will make it so.
        self.run_render_thread(on_success=self.trigger_step_3)
        
    def trigger_step_3(self):
        self.run_video_thread()

    # --- Step 1 ---
    def run_analysis_thread(self, on_success=None):
        if not self.input_structure:
            messagebox.showerror("오류", "파일을 먼저 불러오세요.")
            return

        self.btn_run_all.config(state="disabled")
        self.btn_analyze.config(state="disabled")
        self.btn_render.config(state="disabled")
        self.tree_trans.delete(*self.tree_trans.get_children())
        self.folder_analyses = {}
        
        from multiprocessing import Manager
        m = Manager()
        q = m.Queue()
        self.polling = True
        self.poll_queue(q)
        
        threading.Thread(target=self.run_analysis, args=(q, on_success), daemon=True).start()

    def run_analysis(self, q, on_success=None):
        # ... (Same setup) ...
        workers = self.var_workers.get()
        # ...
        tasks = []
        total_images = 0
        for d_name, files in self.input_structure.items():
            dataset_path = self.dataset_paths[d_name]
            valid_files = [f for f in files if f not in self.excluded_files]
            if len(valid_files) < 1: continue 
            total_images += len(valid_files)
            tasks.append((dataset_path, valid_files, q))
            
        if not tasks:
            self.root.after(0, self.log, "분석할 데이터가 없습니다.")
            self.polling = False
            self.root.after(0, lambda: self.reset_buttons()) 
            return

        self.root.after(0, self.log, f"총 {len(tasks)}개 폴더, {total_images}장 이미지 분석 시작...")
        self.progress_val = 0
        self.root.after(0, lambda: self.progress.configure(value=0, maximum=total_images))
        
        try:
            results_map = {}
            with Pool(workers) as pool:
                for i, (folder_name, results) in enumerate(pool.imap(stabilizer.analyze_folder_worker, tasks)):
                    results_map[folder_name] = results
            
            self.folder_analyses = results_map
            self.sorted_folders = sorted(self.folder_analyses.keys())
            
            # Phase 2
            self.root.after(0, self.log, "날짜간 연결성 분석 중 (Day Gaps)...")
            refine_tasks = []
            if len(self.sorted_folders) > 1:
                for i in range(len(self.sorted_folders)-1):
                    day1 = self.sorted_folders[i]
                    day2 = self.sorted_folders[i+1]
                    s1 = stabilizer.get_noon_samples_with_acc(self.folder_analyses[day1])
                    s2 = stabilizer.get_noon_samples_with_acc(self.folder_analyses[day2])
                    refine_tasks.append((day1, day2, s1, s2))
            
            day_gaps = {}
            if refine_tasks:
                self.progress_val = 0
                self.root.after(0, lambda: self.progress.configure(value=0, maximum=len(refine_tasks)))
                with Pool(workers) as pool:
                    for i, (day2, gap) in enumerate(pool.imap(stabilizer.measure_day_gap_worker, refine_tasks)):
                        day_gaps[day2] = gap
                        q.put(f"[Gap] {os.path.basename(day2)}")
                        self.root.after(0, lambda v=i+1: self.progress.configure(value=v))
            
            self.day_gaps = day_gaps
            self.polling = False
            
            self.root.after(0, self.populate_tree)
            self.root.after(0, lambda: self.left_tabs.select(self.tab_transitions))
            self.root.after(0, lambda: self.progress.configure(value=0))
            self.root.after(0, self.log, "1단계 분석 완료.")
            
            if on_success:
                # Chain Next Step
                self.root.after(0, on_success)
            else:
                self.root.after(0, lambda: messagebox.showinfo("완료", "1단계 분석이 완료되었습니다!"))
                self.root.after(0, lambda: self.reset_buttons(render=True))

        except Exception as e:
            self.polling = False
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, lambda: self.reset_buttons())

    def reset_buttons(self, render=False):
        self.btn_run_all.config(state="normal")
        self.btn_analyze.config(state="normal")
        if render:
            self.btn_render.config(state="normal")
            
    # --- Step 2 ---
    def run_render_thread(self, on_success=None):
        self.btn_run_all.config(state="disabled")
        self.btn_render.config(state="disabled")
        self.btn_analyze.config(state="disabled")
        
        from multiprocessing import Manager
        m = Manager()
        q = m.Queue()
        self.polling = True
        self.poll_queue(q)
        
        threading.Thread(target=self.run_render, args=(q, on_success), daemon=True).start()
        
    def run_render(self, q, on_success=None):
        try:
            self.log("궤적 최적화 중 (Trajectory Integration)...")
            global_traj = stabilizer.integrate_trajectory(self.folder_analyses, self.day_gaps)
            
            output_dir = self.entry_output.get()
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            
            render_tasks = []
            total_frames = 0
            for folder in self.sorted_folders:
                out_path = os.path.join(output_dir, os.path.basename(folder))
                traj = global_traj[folder]
                total_frames += len(traj)
                render_tasks.append((folder, out_path, traj, q))
            
            workers = self.var_workers.get()
            self.log(f"이미지 보정 및 저장 중... (총 {total_frames}장)")
            
            self.progress_val = 0
            self.root.after(0, lambda: self.progress.configure(value=0, maximum=total_frames))
            
            with Pool(workers) as pool:
                for i, _ in enumerate(pool.imap(stabilizer.render_folder_worker, render_tasks)):
                    pass
                    
            self.polling = False
            self.root.after(0, self.log, "보정 완료!")
            self.root.after(0, lambda: self.progress.configure(value=0))
            
            if on_success:
                self.root.after(0, on_success)
            else:
                self.root.after(0, lambda: messagebox.showinfo("완료", "2단계 이미지 보정이 완료되었습니다."))
                self.root.after(0, lambda: self.reset_buttons(render=True))
                # Enable Video button too
                self.root.after(0, lambda: self.btn_video.config(state="normal"))

        except Exception as e:
            self.polling = False
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, lambda: self.reset_buttons(render=True))

    def create_main_panel(self):
        self.paned = ttk.PanedWindow(self.root, orient="horizontal")
        self.paned.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left: Tabs for Files & Transitions
        self.left_tabs = ttk.Notebook(self.paned, width=400)
        self.paned.add(self.left_tabs, weight=1)
        
        # Tab 1: Source Files
        self.tab_files = ttk.Frame(self.left_tabs)
        self.left_tabs.add(self.tab_files, text="1. 원본 파일 목록")
        
        self.tree_files = ttk.Treeview(self.tab_files, columns=("check", "size"), show="tree headings")
        self.tree_files.heading("#0", text="폴더 / 파일명")
        self.tree_files.heading("check", text="선택")
        self.tree_files.heading("size", text="크기")
        self.tree_files.column("#0", width=220)
        self.tree_files.column("check", width=40, anchor="center")
        self.tree_files.column("size", width=80)
        
        # Scroll
        scroll_files = ttk.Scrollbar(self.tab_files, orient="vertical", command=self.tree_files.yview)
        self.tree_files.configure(yscrollcommand=scroll_files.set)
        
        self.tree_files.pack(side="left", fill="both", expand=True)
        scroll_files.pack(side="right", fill="y")
        
        # Bindings
        self.tree_files.bind("<<TreeviewSelect>>", self.on_file_select)
        self.tree_files.bind("<ButtonRelease-1>", self.on_tree_click) # Single click handler
        # Removed Double-1 binding to avoid conflict

        # Tab 2: Transitions
        self.tab_transitions = ttk.Frame(self.left_tabs)
        self.left_tabs.add(self.tab_transitions, text="2. 변환 분석 결과")
        
        columns = ("transition", "offset", "status")
        self.tree_trans = ttk.Treeview(self.tab_transitions, columns=columns, show="headings")
        self.tree_trans.heading("transition", text="변환 구간 (Day A -> Day B)")
        self.tree_trans.heading("offset", text="예측 이동량 (Offset)")
        self.tree_trans.heading("status", text="상태")
        self.tree_trans.column("transition", width=200)
        self.tree_trans.column("offset", width=100)
        self.tree_trans.column("status", width=80)
        
        scroll_trans = ttk.Scrollbar(self.tab_transitions, orient="vertical", command=self.tree_trans.yview)
        self.tree_trans.configure(yscrollcommand=scroll_trans.set)
        
        self.tree_trans.pack(side="left", fill="both", expand=True)
        scroll_trans.pack(side="right", fill="y")
        
        self.tree_trans.bind("<<TreeviewSelect>>", self.on_select_transition)
        
        # Right: Preview Pane
        self.right_frame = ttk.LabelFrame(self.paned, text="미리보기 및 확인", padding=10, width=500)
        self.paned.add(self.right_frame, weight=1)
        
        self.lbl_preview_title = ttk.Label(self.right_frame, text="파일 또는 변환 구간을 선택하세요.", font=("Arial", 12))
        self.lbl_preview_title.pack(pady=5)
        
        self.lbl_image = ttk.Label(self.right_frame)
        self.lbl_image.pack(pady=5, expand=True)
        
        # Controls Frame
        self.ctrl_frame = ttk.Frame(self.right_frame)
        self.ctrl_frame.pack(fill="x", pady=10)
        
        self.btn_exclude = ttk.Button(self.ctrl_frame, text="제외 / 복구 토글", command=self.toggle_current_selection, state="disabled")
        self.btn_exclude.pack(side="left", padx=5)
        
        self.btn_edit_align = ttk.Button(self.ctrl_frame, text="수동 정렬 편집 (활성화: 결과 탭 선택)", command=self.open_visualizer, state="disabled")
        self.btn_edit_align.pack(side="right", padx=5)

        

        
        # Progress Bar

        
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
        self.excluded_files = set() 
        self.folder_map = {} 
        
        ext = "jpg"
        
        self.tree_files.tag_configure("excluded", foreground="red", font=("Arial", 9, "overstrike"))
        self.tree_files.tag_configure("file", foreground="black", font=("Arial", 9))
        self.tree_files.tag_configure("folder_excluded", foreground="red", font=("Arial", 9, "overstrike"))
        
        # 1. Check subfolders (exclude output directory)
        output_dir = os.path.normpath(self.entry_output.get())
        subfolders = sorted([
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
            and os.path.normpath(os.path.join(input_dir, d)) != output_dir
        ])
        
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
                    
                    folder_node = self.tree_files.insert("", "end", text=d, values=("☑", f"{len(imgs)} files"))
                    self.folder_map[folder_node] = d
                    
                    for img in imgs:
                        fname = os.path.basename(img)
                        size_mb = os.path.getsize(img) / (1024*1024)
                        self.tree_files.insert(folder_node, "end", text=fname, values=("☑", f"{size_mb:.1f} MB"), tags=("file",))
                        
            self.log(f"Found {len(self.input_structure)} datasets ({total_imgs} images).")
            
        else:
            # 2. Check root
            imgs = stabilizer.get_images(input_dir, ext)
            if imgs:
                self.dataset_is_root = True
                d_name = os.path.basename(input_dir)
                self.input_structure[d_name] = imgs
                self.dataset_paths[d_name] = input_dir
                
                folder_node = self.tree_files.insert("", "end", text=d_name, values=("☑", f"{len(imgs)} files"), open=True)
                self.folder_map[folder_node] = d_name
                
                for img in imgs:
                    fname = os.path.basename(img)
                    size_mb = os.path.getsize(img) / (1024*1024)
                    self.tree_files.insert(folder_node, "end", text=fname, values=("☑", f"{size_mb:.1f} MB"), tags=("file",))
                    
                self.log(f"Found {len(imgs)} images in root.")
            else:
                self.log("No images found.")
                
    def get_file_path_from_item(self, item_id):
        item = self.tree_files.item(item_id)
        tags = item['tags']
        
        if "file" not in tags:
            return None 
            
        fname = item['text'] # Just text now, no prefix
        
        parent_id = self.tree_files.parent(item_id)
        parent_text = self.tree_files.item(parent_id)['text'] 
        
        if parent_text in self.input_structure:
             for path in self.input_structure[parent_text]:
                 if os.path.basename(path) == fname:
                     return path
        return None

    def on_tree_click(self, event):
        # Identify region
        region = self.tree_files.identify("region", event.x, event.y)
        if region != "cell": return
        
        # Identify column
        col = self.tree_files.identify_column(event.x)
        # Columns: #0 (Display), #1 (check), #2 (size). 
        # identify_column returns string like "#1"
        
        if col == "#1": # The 'check' column
            item_id = self.tree_files.identify_row(event.y)
            if item_id:
                self.toggle_item(item_id)

    def on_file_select(self, event):
        sel = self.tree_files.selection()
        if not sel: return
        item_id = sel[0]
        
        path = self.get_file_path_from_item(item_id)
        
        is_excluded = False
        if path:
            self.show_preview_image(path)
            is_excluded = path in self.excluded_files
        else:
            if item_id in self.folder_map:
                tags = self.tree_files.item(item_id, "tags")
                is_excluded = "folder_excluded" in tags

        if is_excluded:
            self.btn_exclude.config(text="Restore Item", state="normal")
        else:
            self.btn_exclude.config(text="Exclude Item", state="normal")
            
    def toggle_current_selection(self):
        sel = self.tree_files.selection()
        if not sel: return
        item_id = sel[0]
        self.toggle_item(item_id)
        
        self.on_file_select(None)

    def toggle_item(self, item_id):
        item = self.tree_files.item(item_id)
        tags = list(item['tags'])
        values = list(item['values']) # [check, size]
        
        is_checked = (values[0] == "☑")
        new_symbol = "☐" if is_checked else "☑"
        values[0] = new_symbol
        
        # Toggle Logic
        if "file" in tags:
            path = self.get_file_path_from_item(item_id)
            if not path: return
            
            if is_checked: # Now Excluding
                self.excluded_files.add(path)
                if "excluded" not in tags: tags.append("excluded")
            else: # Now Restoring
                if path in self.excluded_files: self.excluded_files.remove(path)
                if "excluded" in tags: tags.remove("excluded")
                
            self.tree_files.item(item_id, values=values, tags=tuple(tags))
            
        elif item_id in self.folder_map or self.tree_files.parent(item_id) == "":
            # Folder
            children = self.tree_files.get_children(item_id)
            
            if is_checked: # Exclude Folder
                if "folder_excluded" not in tags: tags.append("folder_excluded")
                for child in children:
                    # Check child status from values[0]
                    c_vals = self.tree_files.item(child, "values")
                    if c_vals[0] == "☑": 
                        self.toggle_item(child)
            else: # Restore Folder
                if "folder_excluded" in tags: tags.remove("folder_excluded")
                for child in children:
                     c_vals = self.tree_files.item(child, "values")
                     if c_vals[0] == "☐": 
                        self.toggle_item(child)

            self.tree_files.item(item_id, values=values, tags=tuple(tags))

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
        
        # Use Manager for IPC Queue
        from multiprocessing import Manager
        m = Manager()
        q = m.Queue()
        
        # Start Queue Poller in Main Thread
        self.polling = True
        self.poll_queue(q)
        
        threading.Thread(target=self.run_analysis, args=(q,), daemon=True).start()
        
    def poll_queue(self, q):
        if not self.polling: return
        while not q.empty():
            try:
                msg = q.get_nowait()
                if isinstance(msg, tuple) and msg[0] == 'P_INC':
                    self.progress_val += msg[1]
                    self.progress.configure(value=self.progress_val)
                    # Optional: Update label with count e.g. "Analyzing... (120/500)"
                else:
                    self.lbl_status.config(text=str(msg))
            except: break
        self.root.after(50, self.poll_queue, q)
        
    def run_analysis(self, q):
        workers = self.var_workers.get()
        
        # Prepare Tasks
        tasks = []
        total_images = 0
        
        for d_name, files in self.input_structure.items():
            dataset_path = self.dataset_paths[d_name]
            valid_files = [f for f in files if f not in self.excluded_files]
            if len(valid_files) < 1: continue 
            
            total_images += len(valid_files)
            # Pass Queue in args
            tasks.append((dataset_path, valid_files, q))
            
        if not tasks:
            self.root.after(0, self.log, "분석할 데이터가 없습니다.")
            self.root.after(0, lambda: self.btn_analyze.config(state="normal"))
            self.polling = False
            return

        self.root.after(0, self.log, f"총 {len(tasks)}개 폴더, {total_images}장 이미지 분석 시작...")
        
        # Reset Progress
        self.progress_val = 0
        self.root.after(0, lambda: self.progress.configure(value=0, maximum=total_images))
        
        try:
            results_map = {}
            with Pool(workers) as pool:
                for i, (folder_name, results) in enumerate(pool.imap(stabilizer.analyze_folder_worker, tasks)):
                    results_map[folder_name] = results
                    # Note: Progress is handled by P_INC from worker now
                    
            self.folder_analyses = results_map
            self.sorted_folders = sorted(self.folder_analyses.keys())
            
            # Phase 2: Refinement
            self.root.after(0, self.log, "날짜간 연결성 분석 중 (Day Gaps)...")
            
            refine_tasks = []
            if len(self.sorted_folders) > 1:
                # Normal Multi-Day Logic
                for i in range(len(self.sorted_folders)-1):
                    day1 = self.sorted_folders[i]
                    day2 = self.sorted_folders[i+1]
                    s1 = stabilizer.get_noon_samples_with_acc(self.folder_analyses[day1])
                    s2 = stabilizer.get_noon_samples_with_acc(self.folder_analyses[day2])
                    refine_tasks.append((day1, day2, s1, s2))
                
            day_gaps = {}
            if refine_tasks:
                # Update progress max for phase 2 (Refinement is fast, maybe just count tasks)
                # Refinement doesn't emit P_INC, so we use manual update
                self.progress_val = 0
                self.root.after(0, lambda: self.progress.configure(value=0, maximum=len(refine_tasks)))
                
                with Pool(workers) as pool:
                    for i, (day2, gap) in enumerate(pool.imap(stabilizer.measure_day_gap_worker, refine_tasks)):
                        day_gaps[day2] = gap
                        q.put(f"[Gap] {os.path.basename(day2)}")
                        self.root.after(0, lambda v=i+1: self.progress.configure(value=v))
            
            self.day_gaps = day_gaps
            
            # Finish
            self.polling = False
            self.root.after(0, self.populate_tree)
            self.root.after(0, lambda: self.left_tabs.select(self.tab_transitions))
            self.root.after(0, lambda: self.btn_render.config(state="normal"))
            self.root.after(0, lambda: self.btn_analyze.config(state="normal"))
            self.root.after(0, lambda: self.progress.configure(value=0)) # Reset
            self.root.after(0, self.log, "분석 완료.")
            self.root.after(0, lambda: messagebox.showinfo("완료", "1단계 분석이 완료되었습니다!"))

        except Exception as e:
            self.polling = False
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, lambda: self.btn_analyze.config(state="normal"))

    def populate_tree(self):
        self.tree_trans.delete(*self.tree_trans.get_children())
        days = self.sorted_folders
        
        if not days: return
        
        if len(days) == 1:
            # Single Dataset Case
            d_name = os.path.basename(days[0])
            self.tree_trans.insert("", "end", values=(
                f"{d_name} (Single)", 
                "0.0, 0.0", 
                "Ready"
            ))
            return

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
        self.btn_video.config(state="disabled")
        
        # Queue for Render
        from multiprocessing import Manager
        m = Manager()
        q = m.Queue()
        self.polling = True
        self.poll_queue(q)
        
        threading.Thread(target=self.run_render, args=(q,), daemon=True).start()
        
    def run_render(self, q):
        try:
            self.log("궤적 최적화 중 (Trajectory Integration)...")
            global_traj = stabilizer.integrate_trajectory(self.folder_analyses, self.day_gaps)
            
            output_dir = self.entry_output.get()
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            
            render_tasks = []
            total_frames = 0
            
            for folder in self.sorted_folders:
                out_path = os.path.join(output_dir, os.path.basename(folder))
                traj = global_traj[folder]
                total_frames += len(traj)
                
                # Pass Queue
                render_tasks.append((folder, out_path, traj, q))
            
            workers = self.var_workers.get()
            self.log(f"이미지 보정 및 저장 중... (총 {total_frames}장)")
            
            self.progress_val = 0
            self.root.after(0, lambda: self.progress.configure(value=0, maximum=total_frames))
            
            with Pool(workers) as pool:
                for i, _ in enumerate(pool.imap(stabilizer.render_folder_worker, render_tasks)):
                    # Worker sends P_INC
                    pass
                    
            self.polling = False
            self.root.after(0, self.log, "보정 완료!")
            self.root.after(0, lambda: self.progress.configure(value=0))
            self.root.after(0, lambda: self.btn_render.config(state="normal"))
            self.root.after(0, lambda: self.btn_analyze.config(state="normal"))
            self.root.after(0, lambda: self.btn_video.config(state="normal"))
            self.root.after(0, lambda: messagebox.showinfo("완료", "2단계 이미지 보정이 완료되었습니다.\n이제 비디오를 생성할 수 있습니다."))
            
        except Exception as e:
            self.polling = False
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, lambda: self.btn_render.config(state="normal"))

    # --- Video Generation ---

    def run_video_thread(self):
        self.btn_video.config(state="disabled")
        threading.Thread(target=self.run_video, daemon=True).start()

    def run_video(self):
        try:
            self.log("Gathering frames for video...")
            output_dir = self.entry_output.get()
            fps = 30 # Default
            
            # Re-scan output folder for all images
            # or rely on sorted_folders logic if output structure matches
            video_images = []
            
            # We assume output structure: output_dir / folder_name / images
            # Based on sorted_folders which we know are ordered by date
            
            # Check if folders exist
            valid_folders = []
            for d in self.sorted_folders:
                d_name = os.path.basename(d)
                out_sub = os.path.join(output_dir, d_name)
                if os.path.exists(out_sub):
                    valid_folders.append(out_sub)
            
            # Same logic as timelapse_stabilizer.py Phase 5
            for d_path in valid_folders:
                imgs = stabilizer.get_images(d_path, "jpg") # Assume jpg output
                video_images.extend(imgs)
                
            if not video_images:
                self.root.after(0, self.log, "No frames found in output folder.")
                self.root.after(0, lambda: self.btn_video.config(state="normal"))
                return
                
            self.log(f"Creating video from {len(video_images)} frames...")
            
            # Output Video Path
            from datetime import datetime
            now_str = datetime.now().strftime("%H%M%S")
            vid_path = os.path.join(output_dir, f"timelapse_stabilized_{now_str}.mp4")
            
            # Use create_video module
            import create_video
            create_video.create_chunk_video(video_images, vid_path, fps=fps)
            
            self.root.after(0, self.log, f"Video saved to {vid_path}")
            self.root.after(0, lambda: self.btn_video.config(state="normal"))
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Video Created:\n{vid_path}"))
            
        except Exception as e:
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, lambda: self.btn_video.config(state="normal"))

if __name__ == "__main__":
    freeze_support()
    root = tk.Tk()
    app = AlignerApp(root)
    root.mainloop()
