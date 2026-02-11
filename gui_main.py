"""
gui_main.py - Main GUI for Timelapse Aligner Pro
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import sys
import threading
import shutil
import numpy as np
from PIL import Image, ImageTk

# Import Core Logic
import timelapse_stabilizer as stabilizer
from gui_visualizer import ManualAlignVisualizer
from gui_layout import LayoutMixin
from multiprocessing import freeze_support, Pool, cpu_count
from gui_theme import *

class AlignerApp(LayoutMixin):
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
        self.day_gaps = {}              # {folder: (dx, dy, rot[, mode])}
        self.sorted_folders = []        # [folder1, folder2, ...]
        self.current_transition_idx = -1
        
        self.output_structure_step1 = {}
        self.output_structure_step2 = {}
        self.output_item_map_step1 = {}
        self.output_item_map_step2 = {}
        self.step1_input_item_map = {}
        self.current_compare_pair = None
        self.compare_swap = False
        self.cancel_requested = False
        self.current_pool = None
        self.current_step = None
        self.file_size_map = {}
        self.project_name_dirty = False
        self.frame_item_map = {}            # {tree_item_id: (folder_path, frame_index)}
        self.current_frame_selection = None  # (folder_path, frame_index) or None
        
        # Styling
        style = ttk.Style()
        style.theme_use('clam')
        self.root.configure(bg=BG_DARKEST)
        self.apply_theme(style)

        # Layout
        self.create_top_panel()
        self.create_action_panel()
        self.create_main_panel()
        self.disable_compare_controls()
        
    # Layout methods moved to gui_layout.py

    def apply_theme(self, style):
        # ── Frames ──
        style.configure("TFrame", background=BG_DARKEST)
        style.configure("Card.TFrame", background=BG_DARK)

        # ── Labels ──
        style.configure("TLabel", background=BG_DARKEST, foreground=TEXT_PRIMARY,
                         font=(FONT_FAMILY, BODY_SIZE))
        style.configure("Title.TLabel", background=BG_DARKEST, foreground=TEXT_PRIMARY,
                         font=(FONT_FAMILY, TITLE_SIZE, "bold"))
        style.configure("Heading.TLabel", background=BG_DARK, foreground=TEXT_PRIMARY,
                         font=(FONT_FAMILY, HEADING_SIZE, "bold"))
        style.configure("Secondary.TLabel", background=BG_DARK, foreground=TEXT_SECONDARY,
                         font=(FONT_FAMILY, BODY_SIZE))
        style.configure("Status.TLabel", background=BG_DARKEST, foreground=TEXT_PRIMARY,
                         font=(FONT_FAMILY, BODY_SIZE, "bold"))
        style.configure("Muted.TLabel", background=BG_DARKEST, foreground=TEXT_MUTED,
                         font=(FONT_FAMILY, SMALL_SIZE))

        # ── LabelFrame ──
        style.configure("TLabelframe", background=BG_DARK, foreground=TEXT_PRIMARY,
                         bordercolor=BG_BORDER, darkcolor=BG_DARK, lightcolor=BG_DARK)
        style.configure("TLabelframe.Label", background=BG_DARK, foreground=ACCENT_PRIMARY,
                         font=(FONT_FAMILY, BODY_SIZE, "bold"))

        # ── Buttons ──
        style.configure("TButton", background=BG_MID, foreground=TEXT_SECONDARY,
                         bordercolor=BG_BORDER, darkcolor=BG_MID, lightcolor=BG_MID,
                         focuscolor=BG_MID,
                         font=(FONT_FAMILY, BODY_SIZE), padding=(10, 4))
        style.map("TButton",
                  background=[("active", BG_ELEVATED), ("disabled", BG_DARKEST)],
                  foreground=[("active", TEXT_PRIMARY), ("disabled", TEXT_MUTED)])

        style.configure("Accent.TButton", background=ACCENT_DARK, foreground=TEXT_BRIGHT,
                         bordercolor=ACCENT_DARK, darkcolor=ACCENT_DARK, lightcolor=ACCENT_DARK,
                         focuscolor=ACCENT_DARK,
                         font=(FONT_FAMILY, BODY_SIZE, "bold"), padding=(12, 5))
        style.map("Accent.TButton",
                  background=[("active", ACCENT_PRIMARY), ("disabled", BG_MID)],
                  foreground=[("active", TEXT_BRIGHT), ("disabled", TEXT_MUTED)])

        style.configure("Hero.TButton", background=ACCENT_PRIMARY, foreground=BG_DARKEST,
                         bordercolor=ACCENT_PRIMARY, darkcolor=ACCENT_DARK, lightcolor=ACCENT_HOVER,
                         focuscolor=ACCENT_PRIMARY,
                         font=(FONT_FAMILY, HEADING_SIZE, "bold"), padding=(20, 8))
        style.map("Hero.TButton",
                  background=[("active", ACCENT_HERO), ("disabled", BG_MID)],
                  foreground=[("active", BG_DARKEST), ("disabled", TEXT_MUTED)])

        # Step buttons
        style.configure("Step.TButton", background=BG_MID, foreground=TEXT_PRIMARY,
                        bordercolor=BG_BORDER, darkcolor=BG_MID, lightcolor=BG_MID,
                        focuscolor=BG_MID,
                        font=(FONT_FAMILY, BODY_SIZE, "bold"), padding=(12, 6))
        style.map("Step.TButton",
                  background=[("active", BG_ELEVATED), ("disabled", BG_DARKEST)],
                  foreground=[("active", TEXT_PRIMARY), ("disabled", TEXT_MUTED)])

        style.configure("StepHero.TButton", background=ACCENT_PRIMARY, foreground=BG_DARKEST,
                        bordercolor=ACCENT_PRIMARY, darkcolor=ACCENT_PRIMARY, lightcolor=ACCENT_PRIMARY,
                        focuscolor=ACCENT_PRIMARY,
                        font=(FONT_FAMILY, BODY_SIZE, "bold"), padding=(12, 6))
        style.map("StepHero.TButton",
                  background=[("active", ACCENT_HERO), ("disabled", BG_MID)],
                  foreground=[("active", BG_DARKEST), ("disabled", TEXT_MUTED)])

        style.configure("StepActive.TButton", background=ACCENT_PRIMARY, foreground=BG_DARKEST,
                        bordercolor=ACCENT_PRIMARY, darkcolor=ACCENT_PRIMARY, lightcolor=ACCENT_PRIMARY,
                        focuscolor=ACCENT_PRIMARY,
                        font=(FONT_FAMILY, BODY_SIZE, "bold"), padding=(12, 6))
        style.map("StepActive.TButton",
                  background=[("active", ACCENT_HERO), ("disabled", BG_MID)],
                  foreground=[("active", BG_DARKEST), ("disabled", TEXT_MUTED)])

        style.configure("StepDone.TButton", background=STATUS_SUCCESS, foreground=BG_DARKEST,
                        bordercolor=STATUS_SUCCESS, darkcolor=STATUS_SUCCESS, lightcolor=STATUS_SUCCESS,
                        focuscolor=STATUS_SUCCESS,
                        font=(FONT_FAMILY, BODY_SIZE, "bold"), padding=(12, 6))
        style.map("StepDone.TButton",
                  background=[("active", STATUS_SUCCESS), ("disabled", BG_MID)],
                  foreground=[("active", BG_DARKEST), ("disabled", TEXT_MUTED)])

        # ── Entry ──
        style.configure("TEntry", fieldbackground=BG_MID, foreground=TEXT_PRIMARY,
                         bordercolor=BG_BORDER, insertcolor=ACCENT_PRIMARY,
                         selectbackground=ACCENT_DARK, selectforeground=TEXT_BRIGHT,
                         font=(FONT_FAMILY, BODY_SIZE), padding=3)
        style.map("TEntry",
                  fieldbackground=[("focus", BG_ELEVATED), ("disabled", BG_DARKEST)],
                  bordercolor=[("focus", ACCENT_PRIMARY)])

        # ── Checkbutton ──
        style.configure("TCheckbutton", background=BG_DARK, foreground=TEXT_SECONDARY,
                         indicatorbackground=BG_MID, indicatorforeground=ACCENT_PRIMARY,
                         focuscolor=BG_DARK,
                         font=(FONT_FAMILY, SMALL_SIZE))
        style.map("TCheckbutton",
                  background=[("active", BG_DARK)],
                  foreground=[("active", TEXT_PRIMARY)],
                  indicatorbackground=[("selected", ACCENT_PRIMARY)])

        # ── Progressbar ──
        style.configure("Horizontal.TProgressbar",
                         background=ACCENT_PRIMARY, troughcolor=BG_MID,
                         bordercolor=BG_MID, darkcolor=ACCENT_PRIMARY,
                         lightcolor=ACCENT_HOVER, borderwidth=0)

        # ── Notebook ──
        style.configure("TNotebook", background=BG_DARKEST, bordercolor=BG_DARKEST,
                         darkcolor=BG_DARKEST, lightcolor=BG_DARKEST, tabmargins=[0, 0, 0, 0])
        style.configure("TNotebook.Tab", background=BG_MID, foreground=TEXT_MUTED,
                         font=(FONT_FAMILY, SMALL_SIZE), padding=(14, 5),
                         bordercolor=BG_DARKEST)
        style.map("TNotebook.Tab",
                  background=[("selected", BG_SURFACE), ("active", BG_ELEVATED)],
                  foreground=[("selected", ACCENT_PRIMARY), ("active", TEXT_PRIMARY)])

        # ── Treeview ──
        style.configure("Treeview",
                         background=BG_DARK, foreground=TEXT_SECONDARY,
                         fieldbackground=BG_DARK, bordercolor=BG_DARKEST,
                         rowheight=22,
                         font=(FONT_FAMILY, SMALL_SIZE))
        style.configure("Treeview.Heading",
                         background=BG_SURFACE, foreground=TEXT_MUTED,
                         bordercolor=BG_DARKEST, relief="flat",
                         font=(FONT_FAMILY, TINY_SIZE, "bold"))
        style.map("Treeview",
                  background=[("selected", BG_ELEVATED)],
                  foreground=[("selected", ACCENT_PRIMARY)])
        style.map("Treeview.Heading",
                  background=[("active", BG_MID)],
                  relief=[("active", "flat")])

        # ── Scrollbar ──
        style.configure("TScrollbar", background=BG_MID, troughcolor=BG_DARK,
                         bordercolor=BG_DARK, arrowcolor=TEXT_MUTED,
                         gripcount=0, borderwidth=0)
        style.map("TScrollbar",
                  background=[("active", BG_ELEVATED)])

        # ── PanedWindow ──
        style.configure("TPanedwindow", background=BG_DARKEST)
        style.configure("Sash", sashthickness=3, handlesize=0)

        # ── Separator ──
        style.configure("TSeparator", background=BG_BORDER)

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    def _parse_day_gap(self, gap_entry):
        """Return (dx, dy, rot, mode) from tuple/list/dict day gap entries."""
        dx = dy = rot = 0.0
        mode = "Auto"
        if isinstance(gap_entry, dict):
            dx = self._safe_float(gap_entry.get("dx", 0.0))
            dy = self._safe_float(gap_entry.get("dy", 0.0))
            rot = self._safe_float(gap_entry.get("rot", 0.0))
            mode = gap_entry.get("mode") or gap_entry.get("status") or "Auto"
        elif isinstance(gap_entry, (list, tuple)):
            if len(gap_entry) > 0:
                dx = self._safe_float(gap_entry[0])
            if len(gap_entry) > 1:
                dy = self._safe_float(gap_entry[1])
            if len(gap_entry) > 2:
                rot = self._safe_float(gap_entry[2])
            if len(gap_entry) > 3 and isinstance(gap_entry[3], str):
                mode = gap_entry[3]
        return dx, dy, rot, mode

    def update_step_state(self, step_key, state):
        """Update step pipeline indicator. state: 'pending', 'active', 'done'"""
        if hasattr(self, 'step_buttons') and step_key in self.step_buttons:
            btn = self.step_buttons[step_key]
            if state == "pending":
                btn.config(style="Step.TButton")
            elif state == "active":
                btn.config(style="StepActive.TButton")
            elif state == "done":
                btn.config(style="StepDone.TButton")
            return
        if not hasattr(self, 'step_indicators') or step_key not in self.step_indicators:
            return

        canvas, text_id, lbl = self.step_indicators[step_key]
        sz = int(canvas['width'])
        cx, cy = sz // 2, sz // 2
        r = sz // 2 - 3

        canvas.delete("glow")
        canvas.delete("ring")
        canvas.delete("circle")

        if state == "pending":
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                               fill=BG_MID, outline=BG_BORDER, width=1, tags="circle")
            canvas.itemconfig(text_id, fill=TEXT_MUTED)
            lbl.config(fg=TEXT_MUTED)
        elif state == "active":
            # Outer glow ring
            gr = r + 3
            canvas.create_oval(cx - gr, cy - gr, cx + gr, cy + gr,
                               fill="", outline=ACCENT_DIM, width=2, tags="glow")
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                               fill=ACCENT_PRIMARY, outline=ACCENT_HOVER, width=2, tags="circle")
            canvas.itemconfig(text_id, fill=TEXT_BRIGHT)
            lbl.config(fg=ACCENT_PRIMARY)
        elif state == "done":
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                               fill=STATUS_SUCCESS, outline=STATUS_SUCCESS, width=1, tags="circle")
            canvas.itemconfig(text_id, fill=TEXT_BRIGHT, text="✓")
            lbl.config(fg=STATUS_SUCCESS)

        canvas.tag_raise(text_id)

    def update_analysis_option_text(self):
        if not hasattr(self, "lbl_analyze_opts"):
            return
        dark_th = self.entry_dark_thresh.get().strip() if hasattr(self, "entry_dark_thresh") else "-"
        norm_target = self.entry_norm_target.get().strip() if hasattr(self, "entry_norm_target") else "-"
        remove_dark = "ON" if self.var_remove_dark.get() else "OFF"
        norm = "ON" if self.var_norm_bright.get() else "OFF"
        if dark_th == "":
            dark_th = "-"
        if norm_target == "":
            norm_target = "-"
        text = f"옵션: Dark {dark_th} / 제외 {remove_dark} / 밝기 {norm} / Target {norm_target}"
        self.lbl_analyze_opts.config(text=text)

    def begin_task(self, step_key):
        self.cancel_requested = False
        self.current_step = step_key
        if hasattr(self, "btn_cancel"):
            self.btn_cancel.config(state="normal")

    def end_task(self):
        self.current_pool = None
        self.current_step = None
        if hasattr(self, "btn_cancel"):
            self.btn_cancel.config(state="disabled")

    def request_cancel(self):
        self.cancel_requested = True
        self.polling = False
        if self.current_pool:
            try:
                self.current_pool.terminate()
            except Exception:
                pass
        if self.current_step:
            self.update_step_state(self.current_step, "pending")
        self.root.after(0, self.log, "작업 중지 요청됨.")
        self.root.after(0, lambda: self.reset_buttons(render=True))

    def apply_analysis_options(self):
        try:
            stabilizer.ECC_ITERATIONS = int(float(self.entry_ecc_iter.get()))
        except Exception:
            pass
        try:
            stabilizer.ECC_EPS = float(self.entry_ecc_eps.get())
        except Exception:
            pass
        try:
            stabilizer.DAY_REFINE_SAMPLES = int(float(self.entry_refine_samples.get()))
        except Exception:
            pass

    def apply_pid_options(self):
        try:
            stabilizer.PID_KP = float(self.entry_pid_kp.get())
        except Exception:
            pass
        try:
            stabilizer.PID_KI = float(self.entry_pid_ki.get())
        except Exception:
            pass
        try:
            stabilizer.PID_KD = float(self.entry_pid_kd.get())
        except Exception:
            pass



    def on_project_name_change(self, event=None):
        self.project_name_dirty = True

    def set_project_name_default(self, force=False):
        if not hasattr(self, "entry_project"):
            return
        current = self.entry_project.get().strip()
        if not force and self.project_name_dirty and current:
            return
        input_path = self.entry_input.get().strip() if hasattr(self, "entry_input") else ""
        base = os.path.basename(os.path.normpath(input_path)) if input_path else ""
        if not base:
            base = "project"
        self.entry_project.delete(0, tk.END)
        self.entry_project.insert(0, base)
        self.project_name_dirty = False

    def get_project_output_dir(self, create=False):
        base_output = self.entry_output.get().strip() if hasattr(self, "entry_output") else ""
        if not base_output:
            base_output = "output"
        project_raw = ""
        if hasattr(self, "entry_project"):
            project_raw = self.entry_project.get().strip()
        if not project_raw:
            project_raw = os.path.basename(os.path.normpath(self.entry_input.get()))
        project_name = self._sanitize_path_component(project_raw)
        output_dir = os.path.join(base_output, project_name)
        if create:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception:
                pass
        return output_dir

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

    # --- Run All Logic ---
    def run_all_steps(self):
        # Start Step 0, with callback to Step 1
        self.run_preprocess_thread(on_success=self.trigger_step_1)
        
    def trigger_step_1(self):
        self.run_analysis_thread(on_success=self.trigger_step_2)
        
    def trigger_step_2(self):
        # Start Step 2, with callback to Step 3
        # We need to wait a bit or just call it? Thread is done.
        # But we need to use root.after to be safe in main thread? 
        # run_analysis calls on_success via root.after? No I will make it so.
        self.run_render_thread(on_success=self.trigger_step_3)
        
    def trigger_step_3(self):
        self.run_video_thread()

    # --- Step 0 ---
    def run_preprocess_thread(self, on_success=None):
        self.begin_task("preprocess")
        self.update_step_state("preprocess", "active")
        self.reset_worker_monitor_state()
        if not self.input_structure:
            messagebox.showerror("오류", "파일을 먼저 불러오세요.")
            return

        # Optimization: Check if we can reapply threshold locally
        need_scan = False
        def check_needs_scan(item):
            tags = self.tree_files.item(item, "tags")
            if "file" in tags:
                vals = self.tree_files.item(item, "values")
                if len(vals) < 3 or vals[2] == "-" or vals[2] == "": return True
            for child in self.tree_files.get_children(item):
                if check_needs_scan(child): return True
            return False

        for item in self.tree_files.get_children():
             if check_needs_scan(item):
                 need_scan = True
                 break

        if not need_scan:
            self.log("밝기 데이터가 이미 존재합니다. Threshold만 재적용합니다.")
            self.reapply_threshold(on_success)
            return

        self.btn_preprocess.config(state="disabled")
        self.btn_analyze.config(state="disabled")
        self.btn_run_all.config(state="disabled")
        
        from multiprocessing import Manager
        m = Manager()
        q = m.Queue()
        self.polling = True
        self.poll_queue(q)
        
        threading.Thread(target=self.run_preprocess, args=(q, on_success), daemon=True).start()


    def reset_worker_monitor_state(self):
        # Reset worker slots and clear previous progress bars/text
        self.worker_map = {}
        self.worker_slots_taken = 0
        self.setup_worker_monitor()

    def run_preprocess(self, q, on_success=None):
        workers = self.var_workers.get()
        tasks = []
        total_images = 0
        
        self.sync_excluded_files_from_tree()
        
        try: dark_th = float(self.entry_dark_thresh.get())
        except: dark_th = 120.0
        
        for d_name, files in self.input_structure.items():
            dataset_path = self.dataset_paths[d_name]
            # Check all files initially (don't filter by excluded yet, or maybe yes?)
            # If we want to re-check, we should check everything that isn't manually excluded?
            # Let's check 'files' that are currently checked in tree?
            # For simplicity, check all files in list
            
            valid_files = files # Check ALL files to find dark ones, even if previously excluded
            if len(valid_files) < 1: continue
            total_images += len(valid_files)
            tasks.append((d_name, valid_files, dark_th, q))
            
        self.root.after(0, self.log, f"1단계 전처리: {total_images}장 밝기 검사 중... (Threshold={dark_th})")
        self.progress_val = 0
        self.root.after(0, lambda: self.progress.configure(value=0, maximum=total_images))
        
        all_results = []
        
        try:
            with Pool(workers) as pool:
                self.current_pool = pool
                for res_list in pool.imap(stabilizer.scan_dark_images_worker, tasks):
                    if self.cancel_requested:
                        try:
                            pool.terminate()
                        except Exception:
                            pass
                        break
                    all_results.extend(res_list)
            
            self.root.after(0, lambda r=all_results: self.update_preprocess_suggestions(r))
            # Post-process: Update UI with Brightness & Exclude Dark
            if self.cancel_requested:
                self.root.after(0, self.log, "1단계 전처리 중지됨.")
                self.end_task()
                return
            self.polling = False
            self.end_task()
            
            # Count dark files
            dark_count = sum(1 for _, _, is_d in all_results if is_d)
            self.root.after(0, self.log, f"1단계 전처리 완료: {len(all_results)}장 분석됨, {dark_count}장 어두운 사진 제외.")
            
            def update_ui_results(results):
                # results: list of (path, brightness, is_dark)
                
                # Create a map for fast lookup
                res_map = {path: (bri, is_d) for path, bri, is_d in results}
                
                count = 0 
                # Iterate Tree to update rows
                for item_id in self.tree_files.get_children():
                    # Check folders' children
                    children = self.tree_files.get_children(item_id)
                    for child_id in children:
                        path = self.get_file_path_from_item(child_id)
                        if path and path in res_map:
                            bri, is_dark = res_map[path]
                            
                            # Update Brightness Column
                            # Current columns: [check, size, brightness]
                            # Treeview values seem to return tuple.
                            current_values = list(self.tree_files.item(child_id, "values"))
                            
                            # We might have initialized with 3 values or 2 values depending on scan_input_structure
                            # But we updated scan_input_structure to have 3 values ("-")
                            
                            if len(current_values) >= 2:
                                check_mark = current_values[0]
                                size_txt = current_values[1]
                                bri_txt = f"{bri:.1f}"
                                
                                new_values = [check_mark, size_txt, bri_txt]
                                
                                tags = list(self.tree_files.item(child_id, "tags"))
                                
                                if is_dark:
                                    self.excluded_files.add(path)
                                    new_values[0] = "☐" # Uncheck
                                    if "excluded" not in tags: tags.append("excluded")
                                    count += 1
                                    
                                self.tree_files.item(child_id, values=new_values, tags=tags)
                
                self.progress.configure(value=0)
                self.btn_preprocess.config(state="normal")
                self.btn_analyze.config(state="normal")
                self.btn_run_all.config(state="normal")
                self.update_step_state("preprocess", "done")
                self.refresh_step1_result_tree()

                if on_success:
                    self.root.after(0, on_success)
                else:
                    messagebox.showinfo("완료", f"1단계 전처리 완료.\n{count}장의 어두운 사진이 제외되었습니다.")

            self.root.after(0, lambda: update_ui_results(all_results))

        except Exception as e:
            self.polling = False
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, lambda: self.reset_buttons()) 

    def reapply_threshold(self, on_success=None):
        try:
            dark_th = float(self.entry_dark_thresh.get())
        except:
            dark_th = 120.0
            
        count = 0
        
        def process_item(item_id):
            nonlocal count
            values = list(self.tree_files.item(item_id, "values"))
            tags = list(self.tree_files.item(item_id, "tags"))
            
            if "file" in tags:
                if len(values) >= 3:
                     bri_str = values[2]
                     try:
                         bri = float(bri_str)
                         path = self.get_file_path_from_item(item_id)
                         
                         is_dark = bri < dark_th
                         
                         if is_dark:
                             if path: self.excluded_files.add(path)
                             values[0] = "☐"
                             if "excluded" not in tags: tags.append("excluded")
                             count += 1
                         else:
                             # Restore if it was excluded
                             if path and path in self.excluded_files:
                                 self.excluded_files.remove(path)
                             values[0] = "☑"
                             if "excluded" in tags: tags.remove("excluded")
                             
                         self.tree_files.item(item_id, values=values, tags=tuple(tags))
                     except: pass
                        
            for child in self.tree_files.get_children(item_id):
                process_item(child)
                
        for item in self.tree_files.get_children():
            process_item(item)
            
        messagebox.showinfo("완료", f"재설정 완료.\n{count}장의 어두운 사진이 제외되었습니다.")
        self.refresh_step1_result_tree()
        if on_success:
            self.root.after(0, on_success)

    def poll_queue(self, q):
        if not self.polling: return
        while not q.empty():
            try:
                msg = q.get_nowait()
                if isinstance(msg, tuple):
                    if msg[0] == 'P_INC':
                        self.progress_val += msg[1]
                        self.progress.configure(value=self.progress_val)
                    elif msg[0] == 'WORKER_PROGRESS':
                        # ('WORKER_PROGRESS', proc_name, percent, status_text)
                        _, pname, pct, txt = msg
                        
                        # Find index for pname
                        # We use a simple hash or just matching known names?
                        # Since we don't know pnames in advance easily (SpawnPoolWorker-X),
                        # we can just map them dynamically to slots 0..N
                        if not hasattr(self, 'worker_map'):
                            self.worker_map = {}
                            self.worker_slots_taken = 0
                            
                        if pname not in self.worker_map:
                            # Assign new slot
                            slot = self.worker_slots_taken
                            if slot < len(self.worker_bars):
                                self.worker_map[pname] = slot
                                self.worker_slots_taken += 1
                        
                        if pname in self.worker_map:
                            idx = self.worker_map[pname]
                            # Start calling the layout helper
                            self.update_worker_progress(idx, pct, txt)
                        
                else:
                    self.lbl_status.config(text=str(msg))
            except: break
        self.root.after(50, self.poll_queue, q)

    # --- Step 1 ---
    def run_analysis_thread(self, on_success=None):
        self.begin_task("analyze")
        self.update_step_state("analyze", "active")
        self.reset_worker_monitor_state()
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
        self.sync_excluded_files_from_tree()
        # ...
        tasks = []
        total_images = 0
        
        # Gather Options
        try: dark_th = float(self.entry_dark_thresh.get())
        except: dark_th = 120.0
        try: norm_target = float(self.entry_norm_target.get())
        except: norm_target = 160.0
        
        options = {
            'dark_threshold': dark_th,
            'remove_dark': self.var_remove_dark.get(),
            'normalize_brightness': self.var_norm_bright.get(),
            'target_brightness': norm_target
        }
        self.apply_analysis_options()

        output_dir = self.get_project_output_dir(create=True)
        step1_dir = os.path.join(output_dir, "step1_analysis")
        analysis_path = os.path.join(step1_dir, "analysis_results.json")
        day_gaps_path = os.path.join(step1_dir, "day_gaps.json")

        cached_results = {}
        cached_day_gaps = None
        try:
            import json
            if os.path.exists(analysis_path):
                with open(analysis_path, "r") as f:
                    cached_results = json.load(f)
            if os.path.exists(day_gaps_path):
                with open(day_gaps_path, "r") as f:
                    cached_day_gaps = json.load(f)
        except Exception as e:
            self.root.after(0, self.log, f"Cache load failed: {e}")

        cached_ok = set()
        for d_name, files in self.input_structure.items():
            dataset_path = self.dataset_paths[d_name]
            dataset_key = os.path.basename(dataset_path)
            valid_files = [f for f in files if f not in self.excluded_files]
            if len(valid_files) < 1: continue 
            if dataset_key in cached_results and len(cached_results[dataset_key]) == len(valid_files):
                cached_ok.add(dataset_key)
                continue
            total_images += len(valid_files)
            tasks.append((dataset_path, valid_files, q, options))
            
        if not tasks and not cached_ok:
            self.root.after(0, self.log, "분석할 데이터가 없습니다.")
            if self.cancel_requested:
                self.root.after(0, self.log, "분석 중지됨.")
                self.end_task()
                return
            self.polling = False
            self.end_task()
            self.root.after(0, lambda: self.reset_buttons()) 
            return

        self.root.after(0, self.log, f"총 {len(tasks)}개 폴더, {total_images}장 이미지 분석 시작...")
        self.progress_val = 0
        self.root.after(0, lambda: self.progress.configure(value=0, maximum=total_images))
        
        try:
            results_map = {k: cached_results[k] for k in cached_ok if k in cached_results}
            if cached_ok:
                self.root.after(0, self.log, f"캐시 재사용: {len(cached_ok)}개 폴더")
            if tasks:
                with Pool(workers) as pool:
                    self.current_pool = pool
                    for i, (folder_name, results) in enumerate(pool.imap(stabilizer.analyze_folder_worker, tasks)):
                        if self.cancel_requested:
                            try:
                                pool.terminate()
                            except Exception:
                                pass
                            break
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
                    refine_tasks.append((day1, day2, s1, s2, q))
            
            day_gaps = {}
            use_cached_gaps = False
            if cached_day_gaps and not tasks:
                missing = [d for d in self.sorted_folders[1:] if d not in cached_day_gaps]
                if not missing:
                    day_gaps = cached_day_gaps
                    use_cached_gaps = True
                    self.root.after(0, self.log, "Day Gaps 캐시 재사용")
            if refine_tasks and not use_cached_gaps:
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
            self.root.after(0, self.refresh_frame_tree)
            self.root.after(0, lambda: self.left_tabs.select(self.tab_transitions))
            self.root.after(0, lambda: self.progress.configure(value=0))
            self.root.after(0, self.log, "2단계 분석 완료.")
            self.root.after(0, lambda: self.update_step_state("analyze", "done"))

            # Save Analysis Results (Step 1 Output)
            self.save_step1_results()
            if on_success:
                # Chain Next Step
                self.root.after(0, on_success)
            else:
                self.root.after(0, lambda: messagebox.showinfo("완료", f"2단계 분석이 완료되었습니다!\n결과 저장: {step1_dir}"))
                self.root.after(0, lambda: self.reset_buttons(render=True))

        except Exception as e:
            self.polling = False
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, lambda: self.reset_buttons())

    def reset_buttons(self, render=False):
        self.btn_run_all.config(state="normal")
        self.btn_analyze.config(state="normal")
        if render:
            self.btn_normalize.config(state="normal")
            self.btn_render.config(state="normal")
            
    # --- Step 3 ---
    def run_normalization_thread(self, on_success=None):
        self.begin_task("normalize")
        self.update_step_state("normalize", "active")
        self.reset_worker_monitor_state()
        if not self.folder_analyses:
             messagebox.showerror("오류", "2단계 분석이 먼저 완료되어야 합니다.")
             return
             
        self.btn_run_all.config(state="disabled")
        self.btn_analyze.config(state="disabled")
        self.btn_normalize.config(state="disabled")
        self.btn_render.config(state="disabled")
        
        from multiprocessing import Manager
        m = Manager()
        q = m.Queue()
        self.polling = True
        self.poll_queue(q)
        
        threading.Thread(target=self.run_normalization, args=(q, on_success), daemon=True).start()
        
    def run_normalization(self, q, on_success=None):
        try:
            self.log("밝기 평준화 적용 중... (Step 3)")
            
            output_dir = self.get_project_output_dir(create=True)
            
            # Options
            try: norm_target = float(self.entry_norm_target.get())
            except: norm_target = 160.0
            options = {
                'normalize_brightness': self.var_norm_bright.get(),
                'target_brightness': norm_target
            }
            
            tasks = []
            total_files = 0
            
            for folder in self.sorted_folders:
                 if folder not in self.folder_analyses: continue
                 
                 results = self.folder_analyses[folder]
                 found_files = [r['abs_path'] for r in results]
                     
                 tasks.append((folder, found_files, self.folder_analyses[folder], output_dir, options, q))
                 total_files += len(found_files)
                 
            workers = self.var_workers.get()
            self.progress_val = 0
            self.root.after(0, lambda: self.progress.configure(value=0, maximum=total_files))
            
            with Pool(workers) as pool:
                self.current_pool = pool
                for _ in pool.imap(stabilizer.normalize_images_worker, tasks):
                    if self.cancel_requested:
                        try:
                            pool.terminate()
                        except Exception:
                            pass
                        break
                    pass
                    
            if self.cancel_requested:
                self.root.after(0, self.log, "밝기 정규화 중지됨.")
                self.end_task()
                return
            self.polling = False
            self.end_task()
            self.root.after(0, self.log, "밝기 평준화 완료.")
            self.root.after(0, lambda: self.update_step_state("normalize", "done"))
            self.root.after(0, self.scan_output_step1)
            self.root.after(0, lambda: self.progress.configure(value=0))
            
            if on_success:
                self.root.after(0, on_success)
            else:
                 msg = f"3단계 완료!\n저장 경로: {os.path.join(output_dir, 'step1_normalized')}"
                 self.root.after(0, lambda: messagebox.showinfo("완료", msg))
                 self.root.after(0, lambda: self.reset_buttons(render=True))

        except Exception as e:
            self.polling = False
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, lambda: self.reset_buttons(render=True))

    # --- Step 4 ---
    def run_render_thread(self, on_success=None):
        self.begin_task("render")
        self.update_step_state("render", "active")
        self.reset_worker_monitor_state()
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
            
            # Gather Options
            try: dark_th = float(self.entry_dark_thresh.get())
            except: dark_th = 120.0
            
            options = {
                'dark_threshold': dark_th,
                'remove_dark': self.var_remove_dark.get(),
                'normalize_brightness': self.var_norm_bright.get()
            }
            
            self.apply_pid_options()
            global_traj = stabilizer.integrate_trajectory(self.folder_analyses, self.day_gaps, options=options)
            
            global_traj = stabilizer.integrate_trajectory(self.folder_analyses, self.day_gaps, options=options)
            
            output_dir = self.get_project_output_dir(create=True)
            step2_dir = os.path.join(output_dir, "step2_render")
            
            # Clean Step 2 Dir
            if os.path.exists(step2_dir):
                shutil.rmtree(step2_dir)
            # os.makedirs(step2_dir) # Created by worker
            
            render_tasks = []
            total_frames = 0
            for folder in self.sorted_folders:
                out_path = os.path.join(step2_dir, os.path.basename(folder))
                traj = global_traj[folder]
                total_frames += len(traj)
                render_tasks.append((folder, out_path, traj, q))
            
            workers = self.var_workers.get()
            self.log(f"이미지 보정 및 저장 중... (총 {total_frames}장)")
            
            self.progress_val = 0
            self.root.after(0, lambda: self.progress.configure(value=0, maximum=total_frames))
            
            with Pool(workers) as pool:
                self.current_pool = pool
                for i, _ in enumerate(pool.imap(stabilizer.render_folder_worker, render_tasks)):
                    if self.cancel_requested:
                        try:
                            pool.terminate()
                        except Exception:
                            pass
                        break
                    pass
                    
            if self.cancel_requested:
                self.root.after(0, self.log, "렌더 중지됨.")
                self.end_task()
                return
            self.polling = False
            self.end_task()
            self.root.after(0, self.log, "보정 완료!")
            self.root.after(0, lambda: self.update_step_state("render", "done"))
            self.root.after(0, self.scan_output_step2)
            self.root.after(0, lambda: self.progress.configure(value=0))
            
            if on_success:
                self.root.after(0, on_success)
            else:
                self.root.after(0, lambda: messagebox.showinfo("완료", "4단계 이미지 보정이 완료되었습니다."))
                self.root.after(0, lambda: self.reset_buttons(render=True))
                # Enable Video button too
                self.root.after(0, lambda: self.btn_video.config(state="normal"))

        except Exception as e:
            self.polling = False
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, lambda: self.reset_buttons(render=True))

    def DEPRECATED_create_main_panel(self):
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
        self.tree_files.heading("brightness", text="밝기 (V)") # New Column
        
        self.tree_files.column("#0", width=220)
        self.tree_files.column("check", width=40, anchor="center")
        self.tree_files.column("size", width=60)
        self.tree_files.column("brightness", width=60, anchor="center")
        
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
        self.set_project_name_default()
        if not os.path.exists(input_dir):
            self.log("Input directory does not exist.")
            return

        self.log("Scanning files...")
        self.tree_files.delete(*self.tree_files.get_children())
        self.input_structure = {}
        self.dataset_paths = {}
        self.excluded_files = set() 
        self.folder_map = {} 
        self.file_size_map = {}
        
        ext = "jpg"
        
        self.tree_files.tag_configure("excluded", foreground=STATUS_ERROR, font=(FONT_FAMILY, 9, "overstrike"))
        self.tree_files.tag_configure("file", foreground=TEXT_PRIMARY, font=(FONT_FAMILY, 9))
        self.tree_files.tag_configure("folder_excluded", foreground=STATUS_ERROR, font=(FONT_FAMILY, 9, "overstrike"))
        
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
                    self.file_size_map[img] = size_mb
                    # values=("check", "size", "brightness")
                    self.tree_files.insert(folder_node, "end", text=fname, values=("☑", f"{size_mb:.1f} MB", "-"), tags=("file",))
                        
                self.log(f"Found {len(self.input_structure)} datasets ({total_imgs} images).")
            self.refresh_step1_result_tree()
              
        else:
            # 2. Check root
            imgs = stabilizer.get_images(input_dir, ext)
            if imgs:
                self.dataset_is_root = True
                d_name = os.path.basename(input_dir)
                self.input_structure[d_name] = imgs
                self.dataset_paths[d_name] = input_dir
                
                folder_node = self.tree_files.insert("", "end", text=d_name, values=("☑", f"{len(imgs)} files", ""), open=True)
                self.folder_map[folder_node] = d_name
                
                for img in imgs:
                    fname = os.path.basename(img)
                    size_mb = os.path.getsize(img) / (1024*1024)
                    self.file_size_map[img] = size_mb
                    self.tree_files.insert(folder_node, "end", text=fname, values=("☑", f"{size_mb:.1f} MB", "-"), tags=("file",))

                self.log(f"Found {len(imgs)} images in root.")
                self.refresh_step1_result_tree()
            else:
                self.log("No images found.")
                
        if hasattr(self, "lbl_dark_suggest"):
            self.lbl_dark_suggest.config(text="추천: -")
        if hasattr(self, "lbl_norm_suggest"):
            self.lbl_norm_suggest.config(text="추천: -")

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

    def save_step1_results(self):
        output_dir = self.get_project_output_dir(create=True)
        step1_dir = os.path.join(output_dir, "step1_analysis")
        if not os.path.exists(step1_dir):
            os.makedirs(step1_dir)
        try:
            import json
            with open(os.path.join(step1_dir, "analysis_results.json"), "w") as f:
                json.dump(self.folder_analyses, f, indent=2)
            with open(os.path.join(step1_dir, "day_gaps.json"), "w") as f:
                json.dump(self.day_gaps, f, indent=2)
        except Exception as e:
            self.root.after(0, self.log, f"Step1 save failed: {e}")

    def update_preprocess_suggestions(self, results):
        values = [b for _, b, _ in results if isinstance(b, (int, float))]
        if not values:
            if hasattr(self, "lbl_dark_suggest"):
                self.lbl_dark_suggest.config(text="추천: -")
            if hasattr(self, "lbl_norm_suggest"):
                self.lbl_norm_suggest.config(text="추천: -")
            return

        vals = np.array(values, dtype=np.float32)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        if mad > 0:
            suggest = med - 2.5 * mad
        else:
            suggest = float(np.percentile(vals, 10))
        suggest = max(0.0, min(255.0, suggest))

        non_dark = [b for _, b, is_dark in results if not is_dark]
        avg = float(np.mean(non_dark)) if non_dark else None

        if hasattr(self, "lbl_dark_suggest"):
            self.lbl_dark_suggest.config(text=f"추천: {suggest:.1f}")
        if hasattr(self, "lbl_norm_suggest"):
            if avg is None:
                self.lbl_norm_suggest.config(text="추천: -")
            else:
                self.lbl_norm_suggest.config(text=f"추천: {avg:.1f}")

    def resolve_original_path(self, folder_name, filename):
        if folder_name in self.dataset_paths:
            base_dir = self.dataset_paths[folder_name]
        elif self.dataset_is_root and self.dataset_paths:
            base_dir = list(self.dataset_paths.values())[0]
        else:
            base_dir = os.path.join(self.entry_input.get(), folder_name)
        return os.path.join(base_dir, filename)

    def scan_output_step1(self):
        self.build_output_tree("step1")

    def scan_output_step2(self):
        self.build_output_tree("step2")

    def build_output_tree(self, step):
        output_dir = self.get_project_output_dir(create=False)
        if step == "step1":
            root_dir = os.path.join(output_dir, "step1_normalized")
            tree = self.tree_step1
            item_map = self.output_item_map_step1
            structure = self.output_structure_step1
        else:
            root_dir = os.path.join(output_dir, "step2_render")
            tree = self.tree_step2
            item_map = self.output_item_map_step2
            structure = self.output_structure_step2
        
        tree.delete(*tree.get_children())
        item_map.clear()
        structure.clear()
        
        if not os.path.exists(root_dir):
            self.log(f"Output not found: {root_dir}")
            return
        
        tree.tag_configure("file", foreground=TEXT_PRIMARY, font=(FONT_FAMILY, 9))
        
        subfolders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        if subfolders:
            for d in subfolders:
                full_d = os.path.join(root_dir, d)
                imgs = stabilizer.get_images(full_d, "jpg")
                if not imgs:
                    continue
                structure[d] = imgs
                folder_node = tree.insert("", "end", text=d, values=(f"{len(imgs)} files",))
                
                for img in imgs:
                    fname = os.path.basename(img)
                    size_mb = os.path.getsize(img) / (1024 * 1024)
                    item_id = tree.insert(folder_node, "end", text=fname,
                                          values=(f"{size_mb:.1f} MB",), tags=("file",))
                    orig_path = self.resolve_original_path(d, fname)
                    item_map[item_id] = (orig_path, img)
        else:
            imgs = stabilizer.get_images(root_dir, "jpg")
            if not imgs:
                self.log("No output images found.")
                return
            root_name = os.path.basename(self.entry_input.get())
            structure[root_name] = imgs
            folder_node = tree.insert("", "end", text=root_name, values=(f"{len(imgs)} files",), open=True)
            for img in imgs:
                fname = os.path.basename(img)
                size_mb = os.path.getsize(img) / (1024 * 1024)
                item_id = tree.insert(folder_node, "end", text=fname,
                                      values=(f"{size_mb:.1f} MB",), tags=("file",))
                orig_path = self.resolve_original_path(root_name, fname)
                item_map[item_id] = (orig_path, img)

    def on_step1_select(self, event):
        self.handle_output_select(self.tree_step1, self.output_item_map_step1, "Step3 결과")

    def on_step2_select(self, event):
        self.handle_output_select(self.tree_step2, self.output_item_map_step2, "Step4 결과")

    def handle_output_select(self, tree, item_map, title_prefix):
        sel = tree.selection()
        if not sel:
            return
        item_id = sel[0]
        if item_id not in item_map:
            return
        orig_path, result_path = item_map[item_id]
        self.current_compare_pair = (orig_path, result_path)
        self.compare_swap = False
        self.enable_compare_controls()
        self.render_compare_preview()
        self.lbl_preview_title.config(text=f"{title_prefix}: {os.path.basename(result_path)}")

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
        self.disable_compare_controls()
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

    def _set_file_state(self, item_id, exclude):
        item = self.tree_files.item(item_id)
        tags = list(item["tags"])
        values = list(item["values"])
        if not values:
            return
        values[0] = "☐" if exclude else "☑"
        path = self.get_file_path_from_item(item_id)
        if path:
            if exclude:
                self.excluded_files.add(path)
            else:
                self.excluded_files.discard(path)
        if exclude:
            if "excluded" not in tags:
                tags.append("excluded")
        else:
            if "excluded" in tags:
                tags.remove("excluded")
        self.tree_files.item(item_id, values=values, tags=tuple(tags))

    def _set_folder_state(self, item_id, exclude):
        item = self.tree_files.item(item_id)
        tags = list(item["tags"])
        values = list(item["values"])
        if values:
            values[0] = "☐" if exclude else "☑"
        if exclude:
            if "folder_excluded" not in tags:
                tags.append("folder_excluded")
        else:
            if "folder_excluded" in tags:
                tags.remove("folder_excluded")
        self.tree_files.item(item_id, values=values, tags=tuple(tags))

        for child in self.tree_files.get_children(item_id):
            child_tags = self.tree_files.item(child, "tags")
            if "file" in child_tags:
                self._set_file_state(child, exclude)
            else:
                self._set_folder_state(child, exclude)

    def toggle_item(self, item_id, refresh=True):
        item = self.tree_files.item(item_id)
        tags = list(item['tags'])
        values = list(item['values']) # [check, size]
        if not values:
            return
        
        is_checked = (values[0] == "☑")
        exclude = is_checked
        
        # Toggle Logic
        if "file" in tags:
            self._set_file_state(item_id, exclude)
        elif item_id in self.folder_map or self.tree_files.parent(item_id) == "":
            self._set_folder_state(item_id, exclude)

        if refresh:
            self.refresh_step1_result_tree()

    def sync_excluded_files_from_tree(self):
        self.excluded_files = set()
        for item_id in self.tree_files.get_children():
            tags = self.tree_files.item(item_id, "tags")
            if "file" in tags:
                path = self.get_file_path_from_item(item_id)
                if path and "excluded" in tags:
                    self.excluded_files.add(path)
                continue

            folder_excluded = "folder_excluded" in tags
            for child_id in self.tree_files.get_children(item_id):
                path = self.get_file_path_from_item(child_id)
                if not path:
                    continue
                child_tags = self.tree_files.item(child_id, "tags")
                if folder_excluded or "excluded" in child_tags:
                    self.excluded_files.add(path)

    def refresh_step1_result_tree(self):
        if not hasattr(self, "tree_step1_input"):
            return
        self.tree_step1_input.delete(*self.tree_step1_input.get_children())
        self.step1_input_item_map.clear()
        self.sync_excluded_files_from_tree()
        
        total = 0
        for d_name, files in self.input_structure.items():
            remaining = [f for f in files if f not in self.excluded_files]
            if not remaining:
                continue
            folder_node = self.tree_step1_input.insert(
                "", "end", text=d_name, values=(f"{len(remaining)} files",))
            for img in remaining:
                fname = os.path.basename(img)
                size_mb = self.file_size_map.get(img)
                if size_mb is None:
                    try:
                        size_mb = os.path.getsize(img) / (1024 * 1024)
                    except Exception:
                        size_mb = 0.0
                    self.file_size_map[img] = size_mb
                item_id = self.tree_step1_input.insert(
                    folder_node, "end", text=fname,
                    values=(f"{size_mb:.1f} MB",), tags=("file",))
                self.step1_input_item_map[item_id] = img
                total += 1
        self.tree_step1_input.tag_configure(
            "file", foreground=TEXT_PRIMARY, font=(FONT_FAMILY, 9))

    def on_step1_input_select(self, event):
        self.disable_compare_controls()
        sel = self.tree_step1_input.selection()
        if not sel:
            return
        item_id = sel[0]
        if item_id in self.step1_input_item_map:
            path = self.step1_input_item_map[item_id]
            self.show_preview_image(path)

    def on_space_toggle_files(self, event):
        sel = self.tree_files.selection()
        if not sel:
            return "break"
        
        selected_set = set(sel)
        filtered = []
        for item_id in sel:
            parent = self.tree_files.parent(item_id)
            skip = False
            while parent:
                if parent in selected_set:
                    skip = True
                    break
                parent = self.tree_files.parent(parent)
            if not skip:
                filtered.append(item_id)
        
        for item_id in filtered:
            self.toggle_item(item_id, refresh=False)
        
        self.refresh_step1_result_tree()
        self.on_file_select(None)
        return "break"

    def refresh_preview(self):
        if self.current_compare_pair:
            self.render_compare_preview()
            return
        if hasattr(self, 'current_preview_path') and self.current_preview_path and os.path.exists(self.current_preview_path):
            self.show_preview_image(self.current_preview_path)

    def show_preview_image(self, path):
        self.current_preview_path = path 
        if not os.path.exists(path): return
        
        # Load and resize for preview
        img = cv2.imread(path)
        if img is None: return
        
        # --- Normalized Preview Logic ---
        if hasattr(self, 'var_view_norm') and self.var_view_norm.get():
            try:
                folder_path = os.path.dirname(path)
                
                # Check if we have analysis for this folder
                if hasattr(self, 'folder_analyses') and folder_path in self.folder_analyses:
                     res_list = self.folder_analyses[folder_path]
                     
                     # Get Options
                     try: norm_target = float(self.entry_norm_target.get())
                     except: norm_target = 160.0
                     options = {
                         'normalize_brightness': True,
                         'target_brightness': norm_target
                     }
                     
                     # Calculate for THIS folder
                     # Warning: This recalculates scales every time. 
                     # But it's fast (ms for typical list).
                     scales = stabilizer.calculate_brightness_scales_for_folder(res_list, options)
                     
                     fname = os.path.basename(path)
                     if fname in scales:
                         scale = scales[fname]
                         # Apply Scale
                         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                         h, s, v = cv2.split(hsv)
                         v_new = cv2.multiply(v.astype(float), scale)
                         v_new = np.clip(v_new, 0, 255).astype(np.uint8)
                         img = cv2.cvtColor(cv2.merge([h,s,v_new]), cv2.COLOR_HSV2BGR)
                         
                         # Add indicator
                         cv2.putText(img, f"Simulated: {scale:.2f}x", (10, 30), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            except Exception as e:
                print(f"Preview Error: {e}")

        self.render_preview_image(img, os.path.basename(path))

    def render_preview_image(self, img, title=None):
        if img is None: return
        h, w = img.shape[:2]
        display_w = 480
        scale_disp = display_w / w
        display_h = int(h * scale_disp)
        
        small = cv2.resize(img, (display_w, display_h))
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        im_tk = ImageTk.PhotoImage(Image.fromarray(small))
        
        self.lbl_image.configure(image=im_tk)
        self.lbl_image.image = im_tk
        if title:
            self.lbl_preview_title.config(text=title)

    def enable_compare_controls(self):
        try:
            self.compare_slider.state(["!disabled"])
        except Exception:
            pass
        self.btn_compare_toggle.config(state="normal")

    def disable_compare_controls(self):
        try:
            self.compare_slider.state(["disabled"])
        except Exception:
            pass
        self.btn_compare_toggle.config(state="disabled")
        self.current_compare_pair = None
        self.compare_swap = False

    def toggle_compare_view(self):
        self.compare_swap = not self.compare_swap
        self.render_compare_preview()

    def on_compare_slider(self, value):
        self.render_compare_preview()

    def render_compare_preview(self):
        if not self.current_compare_pair:
            return
        orig_path, result_path = self.current_compare_pair
        ref_name = os.path.basename(orig_path) if orig_path else "없음"
        aligned_name = os.path.basename(result_path) if result_path else "없음"
        title = f"Ref(원본): {ref_name} | Aligned(결과): {aligned_name}"
        img_a = cv2.imread(orig_path) if orig_path and os.path.exists(orig_path) else None
        img_b = cv2.imread(result_path) if result_path and os.path.exists(result_path) else None
        
        if img_a is None and img_b is None:
            return
        if img_a is None:
            self.render_preview_image(img_b, title)
            return
        if img_b is None:
            self.render_preview_image(img_a, title)
            return
        
        if self.compare_swap:
            img_a, img_b = img_b, img_a
        
        if img_a.shape[:2] != img_b.shape[:2]:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        
        alpha = float(self.var_compare_blend.get())
        blended = cv2.addWeighted(img_a, 1.0 - alpha, img_b, alpha, 0)
        self.render_preview_image(blended, title)

    # --- Logic: Analysis ---
    
    def OLD_run_analysis_thread(self):
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
        
    def OLD_poll_queue(self, q):
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
        
    def OLD_run_analysis(self, q):
        workers = self.var_workers.get()
        
        # Prepare Tasks
        tasks = []
        total_images = 0
        
        # Gather Options
        try: dark_th = float(self.entry_dark_thresh.get())
        except: dark_th = 120.0
        try: norm_target = float(self.entry_norm_target.get())
        except: norm_target = 160.0
        
        options = {
            'dark_threshold': dark_th,
            'remove_dark': self.var_remove_dark.get(),
            'normalize_brightness': self.var_norm_bright.get(),
            'target_brightness': norm_target
        }
        
        for d_name, files in self.input_structure.items():
            dataset_path = self.dataset_paths[d_name]
            valid_files = [f for f in files if f not in self.excluded_files]
            if len(valid_files) < 1: continue 
            
            total_images += len(valid_files)
            # Pass Queue in args
            tasks.append((dataset_path, valid_files, q, options))
            
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
                    refine_tasks.append((day1, day2, s1, s2, q))
                
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
            self.root.after(0, self.refresh_frame_tree)
            self.root.after(0, lambda: self.left_tabs.select(self.tab_transitions))
            self.root.after(0, lambda: self.btn_render.config(state="normal"))
            self.root.after(0, lambda: self.btn_analyze.config(state="normal"))
            self.root.after(0, lambda: self.progress.configure(value=0)) # Reset
            self.root.after(0, self.log, "분석 완료.")
            self.root.after(0, lambda: messagebox.showinfo("완료", "2단계 분석이 완료되었습니다!"))

        except Exception as e:
            self.polling = False
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, lambda: self.btn_analyze.config(state="normal"))

    # --- Per-Frame Analysis Tree ---

    def refresh_frame_tree(self):
        self.tree_frames.delete(*self.tree_frames.get_children())
        self.frame_item_map.clear()
        self.current_frame_selection = None

        self.tree_frames.tag_configure("frame_normal", foreground=TEXT_PRIMARY)
        self.tree_frames.tag_configure("frame_jump", foreground=STATUS_WARNING)
        self.tree_frames.tag_configure("frame_big_jump", foreground=STATUS_ERROR)
        self.tree_frames.tag_configure("frame_manual", foreground=ACCENT_PRIMARY)
        self.tree_frames.tag_configure("frame_dark", foreground=TEXT_MUTED)

        JUMP_THRESHOLD = 5.0
        BIG_JUMP_THRESHOLD = 15.0

        for folder in self.sorted_folders:
            frames = self.folder_analyses.get(folder, [])
            if not frames:
                continue

            max_jump = 0.0
            for f in frames:
                mag = (f['dx']**2 + f['dy']**2)**0.5
                max_jump = max(max_jump, mag)

            folder_name = os.path.basename(folder)
            folder_node = self.tree_frames.insert(
                "", "end", text=folder_name,
                values=(f"{len(frames)}", "", "", "", f"Max:{max_jump:.1f}", ""))

            acc_dx = 0.0
            acc_dy = 0.0
            for idx, frame in enumerate(frames):
                acc_dx += frame['dx']
                acc_dy += frame['dy']

                mag = (frame['dx']**2 + frame['dy']**2)**0.5
                status = frame.get('status', 'OK')

                if status == "DARK":
                    tag = "frame_dark"
                elif str(status).lower() == "manual":
                    tag = "frame_manual"
                elif mag >= BIG_JUMP_THRESHOLD:
                    tag = "frame_big_jump"
                elif mag >= JUMP_THRESHOLD:
                    tag = "frame_jump"
                else:
                    tag = "frame_normal"

                item_id = self.tree_frames.insert(
                    folder_node, "end",
                    text=frame['filename'],
                    values=(
                        f"{frame['dx']:.2f}",
                        f"{frame['dy']:.2f}",
                        f"{frame['rot']:.3f}",
                        status,
                        f"{acc_dx:.1f}",
                        f"{acc_dy:.1f}"
                    ),
                    tags=(tag,))

                self.frame_item_map[item_id] = (folder, idx)

    def on_frame_select(self, event):
        self.disable_compare_controls()
        self.current_transition_idx = -1
        sel = self.tree_frames.selection()
        if not sel:
            self.btn_edit_align.config(state="disabled")
            self.current_frame_selection = None
            return

        item_id = sel[0]
        if item_id not in self.frame_item_map:
            self.btn_edit_align.config(state="disabled")
            self.current_frame_selection = None
            return

        folder_path, frame_idx = self.frame_item_map[item_id]
        frames = self.folder_analyses[folder_path]
        frame = frames[frame_idx]
        self.current_frame_selection = (folder_path, frame_idx)

        if frame_idx == 0:
            self.show_preview_image(frame['abs_path'])
            self.lbl_preview_title.config(
                text=f"프레임: {frame['filename']} (기준 프레임)")
            self.btn_edit_align.config(state="disabled")
        else:
            prev_frame = frames[frame_idx - 1]
            self._show_frame_align_preview(prev_frame, frame)
            is_adjustable = frame.get('status') not in ("DARK", "FAIL_READ")
            self.btn_edit_align.config(
                state="normal" if is_adjustable else "disabled")

    def _show_frame_align_preview(self, prev_frame, curr_frame):
        img_ref = cv2.imread(prev_frame['abs_path'])
        img_cur = cv2.imread(curr_frame['abs_path'])
        if img_ref is None or img_cur is None:
            return

        h, w = img_ref.shape[:2]
        dx, dy = curr_frame['dx'], curr_frame['dy']
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(img_cur, M, (w, h))

        if aligned.shape[:2] != img_ref.shape[:2]:
            aligned = cv2.resize(aligned, (w, h))

        blended = cv2.addWeighted(img_ref, 0.5, aligned, 0.5, 0)
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

        status = curr_frame.get('status', 'OK')
        title = (f"프레임: {curr_frame['filename']} | "
                 f"dx={dx:.2f} dy={dy:.2f} rot={curr_frame['rot']:.3f} | {status}")
        self.lbl_preview_title.config(text=title)
        self.render_preview_image(blended)

    def on_frame_double_click(self, event):
        sel = self.tree_frames.selection()
        if not sel:
            return
        item_id = sel[0]
        if item_id in self.frame_item_map:
            self.open_manual_align()

    def open_manual_align(self):
        """Unified manual alignment — dispatches to frame or day-to-day."""
        if self.current_frame_selection:
            self.open_frame_visualizer()
        elif self.current_transition_idx >= 0:
            self.open_visualizer()

    def open_frame_visualizer(self):
        if not self.current_frame_selection:
            return

        folder_path, frame_idx = self.current_frame_selection
        frames = self.folder_analyses[folder_path]

        if frame_idx == 0:
            messagebox.showinfo("알림", "첫 번째 프레임은 기준 프레임이므로 조정할 수 없습니다.")
            return

        current_frame = frames[frame_idx]
        prev_frame = frames[frame_idx - 1]

        vis = ManualAlignVisualizer(
            prev_frame['abs_path'],
            current_frame['abs_path'],
            initial_dx=current_frame['dx'],
            initial_dy=current_frame['dy']
        )
        result = vis.run()

        if result is not None:
            new_dx, new_dy = result
            current_frame['dx'] = new_dx
            current_frame['dy'] = new_dy
            current_frame['status'] = "Manual"

            self.save_step1_results()
            self.refresh_frame_tree()

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
            gap_dx, gap_dy, _, gap_mode = self._parse_day_gap(
                self.day_gaps.get(day2, (0.0, 0.0, 0.0))
            )
            name1 = os.path.basename(day1)
            name2 = os.path.basename(day2)
            status_text = "Manual" if str(gap_mode).lower() == "manual" else "Auto"
            self.tree_trans.insert("", "end", iid=str(i), values=(
                f"{name1} -> {name2}",
                f"{gap_dx:.1f}, {gap_dy:.1f}",
                status_text
            ))

    def on_select_transition(self, event):
        self.disable_compare_controls()
        self.current_frame_selection = None
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
        
        rel_dx, rel_dy, _, _ = self._parse_day_gap(self.day_gaps.get(day2, (0.0, 0.0, 0.0)))
        
        self.lbl_preview_title.config(text=f"Transition: {os.path.basename(day1)} -> {os.path.basename(day2)}")
        
        h, w = img1.shape[:2]
        M = np.float32([[1, 0, rel_dx], [0, 1, rel_dy]])
        img2_shifted = cv2.warpAffine(img2, M, (w, h))

        if img2_shifted.shape[:2] != img1.shape[:2]:
            img2_shifted = cv2.resize(img2_shifted, (w, h))

        blend = cv2.addWeighted(img1, 0.6, img2_shifted, 0.4, 0)
        blend = cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)
        self.render_preview_image(blend)

    def open_visualizer(self):
        if self.current_transition_idx < 0: return
        idx = self.current_transition_idx
        day1 = self.sorted_folders[idx]
        day2 = self.sorted_folders[idx+1]
        s1 = stabilizer.get_noon_samples_with_acc(self.folder_analyses[day1], n_samples=1)[0]
        s2 = stabilizer.get_noon_samples_with_acc(self.folder_analyses[day2], n_samples=1)[0]
        gap_dx, gap_dy, gap_rot, _ = self._parse_day_gap(self.day_gaps.get(day2, (0.0, 0.0, 0.0)))
        
        vis = ManualAlignVisualizer(s1['abs_path'], s2['abs_path'], initial_dx=gap_dx, initial_dy=gap_dy)
        new_dx, new_dy = vis.run()
        
        if new_dx is not None:
            self.day_gaps[day2] = (new_dx, new_dy, gap_rot, "Manual")
            self.save_step1_results()
            self.populate_tree()
            self.tree_trans.selection_set(str(idx))
            self.update_align_preview(idx)

    # --- Video Generation ---

    def run_video_thread(self):
        self.begin_task("video")
        self.update_step_state("video", "active")
        self.btn_video.config(state="disabled")
        threading.Thread(target=self.run_video, daemon=True).start()

    def _sanitize_filename_component(self, text):
        cleaned = (text or "").strip()
        if not cleaned:
            return "project"
        for ch in '<>:"/\\|?*':
            cleaned = cleaned.replace(ch, "")
        cleaned = cleaned.replace(" ", "_")
        return cleaned or "project"

    def _sanitize_path_component(self, text):
        cleaned = (text or "").strip()
        if not cleaned:
            return "project"
        for ch in '<>:"/\\|?*':
            cleaned = cleaned.replace(ch, "")
        return cleaned or "project"

    def _extract_date_token(self, text):
        import re
        m = re.search(r"\d{4}-\d{2}-\d{2}", text or "")
        return m.group(0) if m else None

    def _get_date_range_from_names(self, names):
        dates = []
        for name in names:
            token = self._extract_date_token(name)
            if token:
                dates.append(token)
        if not dates:
            return None, None
        dates_sorted = sorted(set(dates))
        return dates_sorted[0], dates_sorted[-1]

    def _get_quality_label(self, width, sample_path=None):
        w = width
        if w is None and sample_path:
            try:
                img = cv2.imread(sample_path)
                if img is not None:
                    w = int(img.shape[1])
            except Exception:
                w = None
        if w is None:
            return "Auto"
        if w >= 4096:
            return "4K"
        if w >= 3840:
            return "UHD"
        if w >= 2560:
            return "QHD"
        if w >= 1920:
            return "FHD"
        if w >= 1280:
            return "HD"
        return f"{w}w"

    def run_video(self):
        try:
            self.log("5단계 비디오 생성: 프레임 수집 중...")
            output_dir = self.get_project_output_dir(create=True)
            step2_dir = os.path.join(output_dir, "step2_render")
            step3_dir = os.path.join(output_dir, "step3_video")
            if not os.path.exists(step3_dir): os.makedirs(step3_dir)
            
            fps = 30
            crf = 18
            width = None

            if hasattr(self, "entry_video_fps"):
                try:
                    fps_val = int(self.entry_video_fps.get().strip())
                    if fps_val > 0:
                        fps = fps_val
                    else:
                        self.root.after(0, self.log, "Invalid FPS; using 30.")
                except Exception:
                    self.root.after(0, self.log, "Invalid FPS; using 30.")

            if hasattr(self, "entry_video_crf"):
                try:
                    crf_val = int(self.entry_video_crf.get().strip())
                    if 0 <= crf_val <= 51:
                        crf = crf_val
                    else:
                        self.root.after(0, self.log, "Invalid CRF (0-51); using 18.")
                except Exception:
                    self.root.after(0, self.log, "Invalid CRF; using 18.")

            if hasattr(self, "entry_video_width"):
                width_raw = self.entry_video_width.get().strip()
                if width_raw:
                    try:
                        width_val = int(width_raw)
                        if width_val > 0:
                            width = width_val
                        else:
                            self.root.after(0, self.log, "Invalid width; ignoring.")
                    except Exception:
                        self.root.after(0, self.log, "Invalid width; ignoring.")
            
            # Re-scan output folder for all images
            # or rely on sorted_folders logic if output structure matches
            video_images = []
            
            # We assume output structure: step2_dir / folder_name / images
            # Based on sorted_folders which we know are ordered by date
            
            # Check if folders exist
            valid_folders = []
            for d in self.sorted_folders:
                d_name = os.path.basename(d)
                out_sub = os.path.join(step2_dir, d_name)
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
                
            width_text = "auto" if width is None else str(width)
            self.log(f"5단계 비디오 생성 중: {len(video_images)} frames... (fps={fps}, crf={crf}, width={width_text})")
            
            # Output Video Path
            project_raw = ""
            if hasattr(self, "entry_project"):
                project_raw = self.entry_project.get().strip()
            if not project_raw:
                project_raw = os.path.basename(os.path.normpath(self.entry_input.get()))
            project_name = self._sanitize_filename_component(project_raw)

            start_date, end_date = self._get_date_range_from_names(self.sorted_folders)
            if not start_date or not end_date:
                img_names = [os.path.basename(p) for p in video_images]
                start_date, end_date = self._get_date_range_from_names(img_names)

            if start_date and end_date:
                date_part = start_date if start_date == end_date else f"{start_date}_{end_date}"
            else:
                date_part = "unknown-date"

            quality_label = self._get_quality_label(width, video_images[0] if video_images else None)
            base_name = f"{project_name}_{date_part}_{quality_label}.mp4"
            vid_path = os.path.join(step3_dir, base_name)

            if os.path.exists(vid_path):
                from datetime import datetime
                now_str = datetime.now().strftime("%H%M%S")
                vid_path = os.path.join(step3_dir, f"{project_name}_{date_part}_{quality_label}_{now_str}.mp4")
            
            # Use create_video module with chunked processing
            import create_video
            create_video.create_video_chunked(
                input_dir=step2_dir, # dummy, likely unused if list provided
                output_file=vid_path, 
                fps=fps,
                crf=crf,
                width=width,
                image_list=video_images
            )
            
            self.root.after(0, self.log, f"5단계 비디오 생성 완료: {vid_path}")
            self.root.after(0, lambda: self.update_step_state("video", "done"))
            self.root.after(0, lambda: self.btn_video.config(state="normal"))
            self.root.after(0, lambda: messagebox.showinfo("완료", f"5단계 비디오 생성 완료:\n{vid_path}"))
            
        except Exception as e:
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, lambda: self.btn_video.config(state="normal"))

if __name__ == "__main__":
    freeze_support()
    root = tk.Tk()
    app = AlignerApp(root)
    root.mainloop()
