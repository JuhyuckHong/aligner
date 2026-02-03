import tkinter as tk
from tkinter import ttk
import os
import sys
from multiprocessing import cpu_count
import math

from gui_theme import *


def _make_card(parent, pad_x=PAD_LG, pad_y=PAD_MD):
    """Create a bordered card surface with subtle edge."""
    outer = tk.Frame(parent, bg=BG_BORDER)
    inner = tk.Frame(outer, bg=BG_SURFACE, padx=pad_x, pady=pad_y)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    return outer, inner


def _make_section_label(parent, text):
    """Tiny uppercase section label — industrial style."""
    lbl = tk.Label(parent, text=text.upper(), bg=BG_SURFACE, fg=TEXT_MUTED,
                   font=(FONT_FAMILY, TINY_SIZE, "bold"))
    return lbl


class LayoutMixin:

    # ════════════════════════════════════════════════════════════
    #  TOP PANEL — Title + Config Card + Worker Monitor
    # ════════════════════════════════════════════════════════════

    def create_top_panel(self):
        # ── Thin branded title strip ──
        title_strip = tk.Canvas(self.root, height=24, bg=BG_DARKEST,
                                highlightthickness=0)
        title_strip.pack(fill="x")
        # Teal accent line at top (2px)
        title_strip.create_rectangle(0, 0, 3000, 2, fill=ACCENT_DIM, outline="")
        title_strip.create_text(
            PAD_LG, 13, text="TIMELAPSE ALIGNER PRO", anchor="w",
            fill=TEXT_MUTED, font=(FONT_FAMILY, TINY_SIZE, "bold"))
        # Version badge
        title_strip.create_text(
            200, 13, text="v2.0", anchor="w",
            fill=BG_BORDER, font=(FONT_FAMILY, TINY_SIZE))

        # ── Main config area: left=fields, right=worker monitor ──
        config_outer, config_inner = _make_card(self.root)
        config_outer.pack(fill="x", padx=PAD_MD, pady=(PAD_SM, PAD_SM))

        # Horizontal split
        fields_frame = tk.Frame(config_inner, bg=BG_SURFACE)
        fields_frame.pack(side="left", fill="both", expand=True)

        # Vertical divider
        tk.Frame(config_inner, bg=BG_BORDER, width=1).pack(
            side="left", fill="y", padx=PAD_MD, pady=PAD_XS)

        # Worker monitor (right)
        worker_area = tk.Frame(config_inner, bg=BG_SURFACE)
        worker_area.pack(side="right", fill="y")

        wm_label = tk.Label(worker_area, text="WORKERS", bg=BG_SURFACE,
                            fg=TEXT_MUTED, font=(FONT_FAMILY, TINY_SIZE, "bold"))
        wm_label.pack(anchor="w", pady=(0, PAD_XS))

        self.worker_frame = tk.Frame(worker_area, bg=BG_SURFACE)
        self.worker_frame.pack(fill="both", expand=True)

        # ── Row 0: Input path ──
        _make_section_label(fields_frame, "입력").grid(
            row=0, column=0, sticky="w", pady=(0, PAD_XS))

        self.entry_input = ttk.Entry(fields_frame, width=40)
        self.entry_input.grid(row=0, column=1, padx=PAD_SM, pady=(0, PAD_XS), sticky="ew")

        # Default Paths
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        default_input = os.path.join(base_dir, "input")
        default_output = os.path.join(base_dir, "output")
        for p in (default_input, default_output):
            if not os.path.exists(p):
                try: os.makedirs(p)
                except: pass

        self.entry_input.insert(0, default_input)
        self.entry_input.bind("<FocusOut>", self.on_input_change)

        btn_frame_in = tk.Frame(fields_frame, bg=BG_SURFACE)
        btn_frame_in.grid(row=0, column=2, padx=PAD_XS, pady=(0, PAD_XS))

        ttk.Button(btn_frame_in, text="찾기", command=self.browse_input).pack(
            side="left", padx=(0, PAD_XS))
        ttk.Button(btn_frame_in, text="▸ 불러오기", style="Accent.TButton",
                   command=self.scan_input_structure).pack(side="left")

        # ── Row 1: Output path ──
        _make_section_label(fields_frame, "출력").grid(
            row=1, column=0, sticky="w", pady=(0, PAD_XS))

        self.entry_output = ttk.Entry(fields_frame, width=40)
        self.entry_output.grid(row=1, column=1, padx=PAD_SM, pady=(0, PAD_XS), sticky="ew")
        self.entry_output.insert(0, default_output)

        ttk.Button(fields_frame, text="찾기", command=self.browse_output).grid(
            row=1, column=2, padx=PAD_XS, pady=(0, PAD_XS), sticky="w")

        # ── Row 2: Project name ──
        _make_section_label(fields_frame, "프로젝트").grid(
            row=2, column=0, sticky="w", pady=(0, PAD_XS))

        self.entry_project = ttk.Entry(fields_frame, width=40)
        self.entry_project.grid(row=2, column=1, padx=PAD_SM, pady=(0, PAD_XS), sticky="ew")
        default_project = os.path.basename(os.path.normpath(default_input)) or "project"
        self.entry_project.insert(0, default_project)
        self.entry_project.bind("<KeyRelease>", self.on_project_name_change)

        # ── CPU controls (right of input/output rows) ──
        cpu_frame = tk.Frame(fields_frame, bg=BG_SURFACE)
        cpu_frame.grid(row=0, column=3, rowspan=3, sticky="n", padx=(PAD_MD, 0))

        max_cpu = cpu_count()
        self.var_workers = tk.IntVar(value=max(1, max_cpu - 1))

        tk.Label(cpu_frame, text="CPU", bg=BG_SURFACE, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, TINY_SIZE, "bold")).pack(anchor="w")

        self.scale_workers = tk.Scale(
            cpu_frame, from_=1, to=max_cpu, orient="horizontal",
            variable=self.var_workers, showvalue=0,
            bg=BG_SURFACE, fg=TEXT_PRIMARY, troughcolor=BG_MID,
            activebackground=ACCENT_PRIMARY, highlightthickness=0,
            sliderrelief="flat", length=120, width=12,
            font=(FONT_FAMILY, TINY_SIZE))
        self.scale_workers.pack(anchor="w", pady=(PAD_XS, 0))

        self.lbl_workers_val = tk.Label(
            cpu_frame, bg=BG_SURFACE, fg=ACCENT_PRIMARY,
            font=(FONT_FAMILY, SMALL_SIZE, "bold"),
            text=f"{self.var_workers.get()}/{max_cpu}")
        self.lbl_workers_val.pack(anchor="w")

        def update_worker_label(val):
            self.lbl_workers_val.config(text=f"{int(float(val))}/{max_cpu}")
            self.root.after(100, self.setup_worker_monitor)
        self.scale_workers.config(command=update_worker_label)

        fields_frame.columnconfigure(1, weight=1)

        # Worker monitor init
        self.setup_worker_monitor()

    # ════════════════════════════════════════════════════════════
    #  WORKER MONITOR
    # ════════════════════════════════════════════════════════════

    def setup_worker_monitor(self):
        for w in self.worker_frame.winfo_children():
            w.destroy()

        n = self.var_workers.get()
        self.worker_bars = {}
        cols = 3
        for c in range(cols):
            self.worker_frame.columnconfigure(c, weight=1)

        for i in range(n):
            r, c = i // cols, i % cols
            h, w = 14, 110
            cv = tk.Canvas(self.worker_frame, height=h, width=w,
                           bg=BG_MID, highlightthickness=0)
            cv.grid(row=r, column=c, padx=1, pady=1, sticky="ew")

            rect = cv.create_rectangle(0, 0, 0, h, fill=ACCENT_DIM, width=0)
            txt = cv.create_text(4, h // 2, text=f"{i+1}",
                                 anchor="w", font=(FONT_FAMILY, TINY_SIZE), fill=TEXT_MUTED)
            self.worker_bars[i] = (cv, rect, txt, h)

    def update_worker_progress(self, idx, percent, text):
        if idx not in self.worker_bars:
            return
        cv, rect, txt_id, h = self.worker_bars[idx]

        total_w = cv.winfo_width()
        if total_w < 10:
            total_w = 110

        display_text = text
        if "Scanning" in text:
            display_text = text.replace("Scanning", "Scn")
        elif "Analyzing" in text:
            display_text = text.replace("Analyzing", "Ana")
        elif "Gap" in text:
            display_text = text.split(':')[0]

        cv.itemconfig(txt_id, text=f"{idx+1} {display_text}", fill=TEXT_PRIMARY)

        new_w = int(total_w * (percent / 100.0))
        cv.coords(rect, 0, 0, new_w, h)

        if "Dark" in text or "Excl" in text:
            cv.itemconfig(rect, fill=STATUS_ERROR)
        elif "Gap" in text:
            cv.itemconfig(rect, fill=STATUS_WARNING)
        elif "Done" in text:
            cv.itemconfig(rect, fill=STATUS_SUCCESS)
        else:
            cv.itemconfig(rect, fill=ACCENT_PRIMARY)

    # ════════════════════════════════════════════════════════════
    #  ACTION PANEL — Hero Button + Step Pipeline + Progress
    # ════════════════════════════════════════════════════════════

    def create_action_panel(self):
        action_outer, action_inner = _make_card(self.root, pad_x=PAD_LG, pad_y=PAD_MD)
        action_outer.pack(fill="x", padx=PAD_MD, pady=(0, PAD_SM))

        steps_grid = tk.Frame(action_inner, bg=BG_SURFACE)
        steps_grid.pack(fill="x")

        self.step_buttons = {}

        def add_hint(parent, text):
            return tk.Label(parent, text=text, bg=BG_SURFACE, fg=TEXT_MUTED,
                            font=(FONT_FAMILY, SMALL_SIZE), justify="left",
                            wraplength=120)

        def make_step_group(parent, key, label, cmd, row, col, state="normal"):
            group = tk.Frame(parent, bg=BG_SURFACE)
            group.grid(row=row, column=col, padx=(0, PAD_SM), pady=(0, PAD_SM), sticky="w")
            btn = ttk.Button(group, text=label, style="Step.TButton",
                             command=cmd, width=10, state=state)
            btn.pack(side="left")
            if key:
                self.step_buttons[key] = btn
            opts = tk.Frame(group, bg=BG_SURFACE)
            opts.pack(side="left", padx=(PAD_XS, 0))
            return opts

        # Row 0: Run all + Step 1-2
        run_group = tk.Frame(steps_grid, bg=BG_SURFACE)
        run_group.grid(row=0, column=0, padx=(0, PAD_SM), pady=(0, PAD_SM), sticky="w")
        self.btn_run_all = ttk.Button(
            run_group, text="전체 실행", style="StepHero.TButton",
            command=self.run_all_steps, width=10)
        self.btn_run_all.pack(side="left")

        # Step 1: Preprocess
        opts = make_step_group(steps_grid, "preprocess", "1 전처리", self.run_preprocess_thread, 0, 1)
        self.var_remove_dark = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="어두운 사진 제외", variable=self.var_remove_dark).pack(side="left")
        tk.Label(opts, text="Threshold", bg=BG_SURFACE, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, SMALL_SIZE, "bold")).pack(side="left", padx=(PAD_SM, 0))
        self.entry_dark_thresh = ttk.Entry(opts, width=4)
        self.entry_dark_thresh.pack(side="left", padx=(PAD_XS, PAD_MD))
        self.entry_dark_thresh.insert(0, "120")
        self.lbl_dark_suggest = tk.Label(opts, text="추천: -", bg=BG_SURFACE, fg=TEXT_MUTED,
                                         font=(FONT_FAMILY, SMALL_SIZE))
        self.lbl_dark_suggest.pack(side="left", padx=(PAD_SM, 0))

        # Step 2: Analyze
        opts = make_step_group(steps_grid, "analyze", "2 분석", self.run_analysis_thread, 0, 2)
        tk.Label(opts, text="ECC Iter", bg=BG_SURFACE, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, SMALL_SIZE, "bold")).pack(side="left")
        self.entry_ecc_iter = ttk.Entry(opts, width=4)
        self.entry_ecc_iter.pack(side="left", padx=(PAD_XS, PAD_XS))
        self.entry_ecc_iter.insert(0, "100")
        add_hint(opts, "↑정확/시간").pack(side="left", padx=(0, PAD_SM))

        tk.Label(opts, text="ECC Eps", bg=BG_SURFACE, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, SMALL_SIZE, "bold")).pack(side="left")
        self.entry_ecc_eps = ttk.Entry(opts, width=6)
        self.entry_ecc_eps.pack(side="left", padx=(PAD_XS, PAD_XS))
        self.entry_ecc_eps.insert(0, "1e-4")
        add_hint(opts, "↑빠름/정확도↓").pack(side="left", padx=(0, PAD_SM))

        tk.Label(opts, text="샘플", bg=BG_SURFACE, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, SMALL_SIZE, "bold")).pack(side="left")
        self.entry_refine_samples = ttk.Entry(opts, width=3)
        self.entry_refine_samples.pack(side="left", padx=(PAD_XS, PAD_XS))
        self.entry_refine_samples.insert(0, "5")
        add_hint(opts, "↑안정/시간").pack(side="left")

        # Row 1: Step 3-5
        # Step 3: Normalize
        opts = make_step_group(steps_grid, "normalize", "3 밝기", self.run_normalization_thread, 1, 0, state="disabled")
        self.var_norm_bright = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="밝기 정규화", variable=self.var_norm_bright).pack(side="left")
        tk.Label(opts, text="Target", bg=BG_SURFACE, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, SMALL_SIZE, "bold")).pack(side="left", padx=(PAD_SM, 0))
        self.entry_norm_target = ttk.Entry(opts, width=4)
        self.entry_norm_target.pack(side="left", padx=(PAD_XS, PAD_MD))
        self.entry_norm_target.insert(0, "160")
        self.lbl_norm_suggest = tk.Label(opts, text="추천: -", bg=BG_SURFACE, fg=TEXT_MUTED,
                                         font=(FONT_FAMILY, SMALL_SIZE))
        self.lbl_norm_suggest.pack(side="left", padx=(PAD_SM, 0))

        # Step 4: Render
        opts = make_step_group(steps_grid, "render", "4 렌더", self.run_render_thread, 1, 1, state="disabled")
        tk.Label(opts, text="Kp", bg=BG_SURFACE, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, SMALL_SIZE, "bold")).pack(side="left")
        self.entry_pid_kp = ttk.Entry(opts, width=4)
        self.entry_pid_kp.pack(side="left", padx=(PAD_XS, PAD_XS))
        self.entry_pid_kp.insert(0, "0.8")
        add_hint(opts, "↑보정강도/출렁임").pack(side="left", padx=(0, PAD_SM))

        tk.Label(opts, text="Ki", bg=BG_SURFACE, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, SMALL_SIZE, "bold")).pack(side="left")
        self.entry_pid_ki = ttk.Entry(opts, width=4)
        self.entry_pid_ki.pack(side="left", padx=(PAD_XS, PAD_XS))
        self.entry_pid_ki.insert(0, "0.2")
        add_hint(opts, "↑장기밀림 보정/느린 흔들림").pack(side="left", padx=(0, PAD_SM))

        tk.Label(opts, text="Kd", bg=BG_SURFACE, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, SMALL_SIZE, "bold")).pack(side="left")
        self.entry_pid_kd = ttk.Entry(opts, width=4)
        self.entry_pid_kd.pack(side="left", padx=(PAD_XS, PAD_XS))
        self.entry_pid_kd.insert(0, "0.2")
        add_hint(opts, "↑잔떨림 억제/반응둔화").pack(side="left")

        # Step 5: Video
        opts = make_step_group(steps_grid, "video", "5 비디오", self.run_video_thread, 1, 2, state="disabled")
        tk.Label(opts, text="FPS", bg=BG_SURFACE, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, SMALL_SIZE, "bold")).pack(side="left")
        self.entry_video_fps = ttk.Entry(opts, width=4)
        self.entry_video_fps.pack(side="left", padx=(PAD_XS, PAD_MD))
        self.entry_video_fps.insert(0, "30")
        tk.Label(opts, text="CRF", bg=BG_SURFACE, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, SMALL_SIZE, "bold")).pack(side="left")
        self.entry_video_crf = ttk.Entry(opts, width=4)
        self.entry_video_crf.pack(side="left", padx=(PAD_XS, PAD_MD))
        self.entry_video_crf.insert(0, "18")
        tk.Label(opts, text="WIDTH", bg=BG_SURFACE, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, SMALL_SIZE, "bold")).pack(side="left")
        self.entry_video_width = ttk.Entry(opts, width=6)
        self.entry_video_width.pack(side="left", padx=(PAD_XS, PAD_XS))
        tk.Label(opts, text="px", bg=BG_SURFACE, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, SMALL_SIZE)).pack(side="left")

        # Compatibility aliases for existing logic
        self.btn_preprocess = self.step_buttons.get("preprocess")
        self.btn_analyze = self.step_buttons.get("analyze")
        self.btn_normalize = self.step_buttons.get("normalize")
        self.btn_render = self.step_buttons.get("render")
        self.btn_video = self.step_buttons.get("video")

        # Progress row
        progress_row = tk.Frame(action_inner, bg=BG_SURFACE)
        progress_row.pack(fill="x", pady=(PAD_SM, 0))

        self.btn_cancel = ttk.Button(
            progress_row, text="중지", style="Step.TButton",
            command=self.request_cancel, state="disabled", width=8)
        self.btn_cancel.pack(side="left", padx=(0, PAD_SM))

        self.progress = ttk.Progressbar(progress_row, orient="horizontal",
                                        mode="determinate")
        self.progress.pack(fill="x", side="left", expand=True)

        self.lbl_status = tk.Label(
            progress_row, text="준비됨.", anchor="w",
            bg=BG_SURFACE, fg=TEXT_SECONDARY,
            font=(FONT_FAMILY, SMALL_SIZE))
        self.lbl_status.pack(fill="x", side="top", pady=(PAD_XS, 0))

    def create_main_panel(self):
        self.paned = ttk.PanedWindow(self.root, orient="horizontal")
        self.paned.pack(fill="both", expand=True, padx=PAD_MD, pady=(0, PAD_MD))

        # ── Left: Tabbed panels ──
        left_outer = tk.Frame(self.paned, bg=BG_DARKEST)
        self.paned.add(left_outer, weight=1)

        self.left_tabs = ttk.Notebook(left_outer)
        self.left_tabs.pack(fill="both", expand=True)

        # Tab 1: Files
        self.tab_files = tk.Frame(self.left_tabs, bg=BG_DARK)
        self.left_tabs.add(self.tab_files, text="  파일 목록  ")

        columns = ("check", "size", "brightness")
        self.tree_files = ttk.Treeview(
            self.tab_files, columns=columns, show="tree headings", selectmode="extended")
        self.tree_files.heading("#0", text="폴더 / 파일명")
        self.tree_files.heading("check", text="✓")
        self.tree_files.heading("size", text="크기")
        self.tree_files.heading("brightness", text="밝기")
        self.tree_files.column("#0", width=200)
        self.tree_files.column("check", width=30, anchor="center")
        self.tree_files.column("size", width=55)
        self.tree_files.column("brightness", width=45, anchor="center")

        sf = ttk.Scrollbar(self.tab_files, orient="vertical",
                           command=self.tree_files.yview)
        self.tree_files.configure(yscrollcommand=sf.set)
        self.tree_files.pack(side="left", fill="both", expand=True)
        sf.pack(side="right", fill="y")

        self.tree_files.bind("<<TreeviewSelect>>", self.on_file_select)
        self.tree_files.bind("<ButtonRelease-1>", self.on_tree_click)
        self.tree_files.bind("<space>", self.on_space_toggle_files)

        # Tab 2: Step 1 Results (after exclusions)
        self.tab_step1_input = tk.Frame(self.left_tabs, bg=BG_DARK)
        self.left_tabs.add(self.tab_step1_input, text="  Step1 결과  ")

        columns = ("size",)
        self.tree_step1_input = ttk.Treeview(
            self.tab_step1_input, columns=columns, show="tree headings")
        self.tree_step1_input.heading("#0", text="폴더 / 파일명")
        self.tree_step1_input.heading("size", text="크기")
        self.tree_step1_input.column("#0", width=200)
        self.tree_step1_input.column("size", width=70)

        s0 = ttk.Scrollbar(self.tab_step1_input, orient="vertical",
                           command=self.tree_step1_input.yview)
        self.tree_step1_input.configure(yscrollcommand=s0.set)
        self.tree_step1_input.pack(side="left", fill="both", expand=True)
        s0.pack(side="right", fill="y")

        self.tree_step1_input.bind("<<TreeviewSelect>>", self.on_step1_input_select)

        # Tab 3: Step2 Results (Transitions)
        self.tab_transitions = tk.Frame(self.left_tabs, bg=BG_DARK)
        self.left_tabs.add(self.tab_transitions, text="  Step2 결과  ")

        columns = ("transition", "offset", "status")
        self.tree_trans = ttk.Treeview(
            self.tab_transitions, columns=columns, show="headings")
        self.tree_trans.heading("transition", text="구간")
        self.tree_trans.heading("offset", text="이동량")
        self.tree_trans.heading("status", text="상태")
        self.tree_trans.column("transition", width=180)
        self.tree_trans.column("offset", width=80)
        self.tree_trans.column("status", width=50)

        st = ttk.Scrollbar(self.tab_transitions, orient="vertical",
                           command=self.tree_trans.yview)
        self.tree_trans.configure(yscrollcommand=st.set)
        self.tree_trans.pack(side="left", fill="both", expand=True)
        st.pack(side="right", fill="y")

        self.tree_trans.bind("<<TreeviewSelect>>", self.on_select_transition)

        # Tab 4: Step3 Results (Normalized)
        self.tab_step1 = tk.Frame(self.left_tabs, bg=BG_DARK)
        self.left_tabs.add(self.tab_step1, text="  Step3 결과  ")

        columns = ("size",)
        self.tree_step1 = ttk.Treeview(
            self.tab_step1, columns=columns, show="tree headings")
        self.tree_step1.heading("#0", text="폴더 / 파일명")
        self.tree_step1.heading("size", text="크기")
        self.tree_step1.column("#0", width=200)
        self.tree_step1.column("size", width=70)

        s1 = ttk.Scrollbar(self.tab_step1, orient="vertical",
                           command=self.tree_step1.yview)
        self.tree_step1.configure(yscrollcommand=s1.set)
        self.tree_step1.pack(side="left", fill="both", expand=True)
        s1.pack(side="right", fill="y")

        self.tree_step1.bind("<<TreeviewSelect>>", self.on_step1_select)

        # Tab 5: Step4 Results (Render)
        self.tab_step2 = tk.Frame(self.left_tabs, bg=BG_DARK)
        self.left_tabs.add(self.tab_step2, text="  Step4 결과  ")

        columns = ("size",)
        self.tree_step2 = ttk.Treeview(
            self.tab_step2, columns=columns, show="tree headings")
        self.tree_step2.heading("#0", text="폴더 / 파일명")
        self.tree_step2.heading("size", text="크기")
        self.tree_step2.column("#0", width=200)
        self.tree_step2.column("size", width=70)

        s2 = ttk.Scrollbar(self.tab_step2, orient="vertical",
                           command=self.tree_step2.yview)
        self.tree_step2.configure(yscrollcommand=s2.set)
        self.tree_step2.pack(side="left", fill="both", expand=True)
        s2.pack(side="right", fill="y")

        self.tree_step2.bind("<<TreeviewSelect>>", self.on_step2_select)

        # ── Right: Preview pane ──
        preview_outer = tk.Frame(self.paned, bg=BG_BORDER)
        self.paned.add(preview_outer, weight=1)

        self.right_frame = tk.Frame(preview_outer, bg=BG_DARK)
        self.right_frame.pack(fill="both", expand=True, padx=1, pady=1)

        # Preview header
        header = tk.Frame(self.right_frame, bg=BG_SURFACE)
        header.pack(fill="x")

        self.lbl_preview_title = tk.Label(
            header, text="파일 또는 변환 구간을 선택하세요.",
            bg=BG_SURFACE, fg=TEXT_MUTED,
            font=(FONT_FAMILY, SMALL_SIZE), anchor="w", padx=PAD_MD, pady=PAD_SM)
        self.lbl_preview_title.pack(fill="x")

        # Image viewport — deep dark
        self.lbl_image = tk.Label(self.right_frame, bg=BG_DARKEST)
        self.lbl_image.pack(fill="both", expand=True, padx=PAD_SM, pady=PAD_SM)

        # Bottom toolbar
        toolbar = tk.Frame(self.right_frame, bg=BG_SURFACE)
        toolbar.pack(fill="x")

        inner_tb = tk.Frame(toolbar, bg=BG_SURFACE, padx=PAD_MD, pady=PAD_SM)
        inner_tb.pack(fill="x")

        self.var_view_norm = tk.BooleanVar(value=False)
        self.chk_view_norm = ttk.Checkbutton(
            inner_tb, text="밝기 보정 미리보기",
            variable=self.var_view_norm, command=self.refresh_preview)
        self.chk_view_norm.pack(side="left")

        self.var_compare_blend = tk.DoubleVar(value=0.5)
        self.lbl_compare = tk.Label(
            inner_tb, text="비교", bg=BG_SURFACE, fg=TEXT_MUTED,
            font=(FONT_FAMILY, SMALL_SIZE))
        self.lbl_compare.pack(side="left", padx=(PAD_MD, 0))

        self.compare_slider = ttk.Scale(
            inner_tb, from_=0.0, to=1.0, orient="horizontal",
            variable=self.var_compare_blend, command=self.on_compare_slider)
        self.compare_slider.pack(side="left", padx=(PAD_SM, 0), fill="x", expand=True)

        self.btn_compare_toggle = ttk.Button(
            inner_tb, text="A/B", command=self.toggle_compare_view, state="disabled")
        self.btn_compare_toggle.pack(side="left", padx=(PAD_SM, 0))

        self.btn_edit_align = ttk.Button(
            inner_tb, text="수동 정렬",
            style="Accent.TButton", command=self.open_visualizer, state="disabled")
        self.btn_edit_align.pack(side="right", padx=(PAD_SM, 0))

        self.btn_exclude = ttk.Button(
            inner_tb, text="제외/복구",
            command=self.toggle_current_selection, state="disabled")
        self.btn_exclude.pack(side="right")
