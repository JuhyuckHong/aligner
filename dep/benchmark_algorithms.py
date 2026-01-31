"""
[ARCHIVED]
Reason: Benchmark V1: Basic comparison of PhaseCorr, ECC, SIFT. Replaced by V2.
Date: 2026-01-31
"""

"""
=============================================================================
util/benchmark_algorithms.py - 정합 알고리즘 종합 벤치마크
=============================================================================

사용법:
  ./venv/Scripts/python util/benchmark_algorithms.py              # 전체 평가
  ./venv/Scripts/python util/benchmark_algorithms.py --add-gt     # GT 추가 모드
=============================================================================
"""
import cv2
import numpy as np
import json
import os
import sys
import time

# PyTorch & Kornia (LightGlue)
try:
    # import torch
    # import kornia
    # from kornia.feature import LoFTR, LightGlue, SuperPoint
    HAS_TORCH = False
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch/Kornia not found. Deep Learning methods will be skipped.")

GT_FILE = "util/gt_benchmark.json"

class AlgorithmBenchmark:
    def __init__(self):
        self.results = {}
        
    def run_all(self):
        data = self.load_gt()
        cases = [c for c in data["cases"] if c.get("gt_dx") is not None]
        
        if not cases:
            print("No GT cases found.")
            return

        print(f"=== Benchmark Results ({len(cases)} cases) ===\n")
        
        # 알고리즘 목록
        methods = [
            ("PhaseCorr(Full)", self.alg_phase_corr_full),
            ("PhaseCorr(Center)", self.alg_phase_corr_center),
            ("ECC(Affine)", self.alg_ecc_affine),
            ("SIFT+RANSAC", self.alg_sift),
            ("AKAZE+RANSAC", self.alg_akaze),
        ]
        
        if HAS_TORCH:
            methods.append(("LightGlue", self.alg_lightglue))
            
        # 초기화
        for name, _ in methods:
            self.results[name] = {"err_t": [], "err_r": [], "err_s": [], "time": []}

        for case in cases:
            print(f"Case: {case['id']}")
            print(f"  GT: dx={case['gt_dx']:.2f}, dy={case['gt_dy']:.2f}, rot={case.get('gt_angle',0):.2f}, scl={case.get('gt_scale',1):.4f}")
            
            ref = cv2.imread(case["ref"])
            mov = cv2.imread(case["mov"])
            
            if ref is None or mov is None:
                print("  Failed to load images")
                continue
                
            for name, func in methods:
                start_t = time.time()
                try:
                    dx, dy, rot, scale = func(ref, mov)
                    elapsed = (time.time() - start_t) * 1000
                    
                    # 에러 계산
                    gt_dx, gt_dy = case["gt_dx"], case["gt_dy"]
                    gt_rot = case.get("gt_angle", 0.0)
                    gt_scl = case.get("gt_scale", 1.0)
                    
                    err_dx = abs(dx - gt_dx)
                    err_dy = abs(dy - gt_dy)
                    err_t = np.sqrt(err_dx**2 + err_dy**2)
                    err_r = abs(rot - gt_rot)
                    err_s = abs(scale - gt_scl)
                    
                    self.results[name]["err_t"].append(err_t)
                    self.results[name]["err_r"].append(err_r)
                    self.results[name]["err_s"].append(err_s)
                    self.results[name]["time"].append(elapsed)
                    
                    status = "✓" if err_t < 1.0 else "✗"
                    print(f"  {name:18s}: T={err_t:5.2f}px (dx={dx:5.1f},dy={dy:5.1f}) R={rot:5.2f}° S={scale:5.3f} ({elapsed:4.0f}ms) {status}")
                    
                except Exception as e:
                    print(f"  {name:18s}: ERROR - {e}")
            print()
            
        self.print_summary()

    def print_summary(self):
        print("\n=== Summary ===")
        print(f"{'Algorithm':<18} {'Avg Trans':>10} {'Avg Rot':>10} {'Avg Scale':>10} {'Avg Time':>10}")
        print("-" * 65)
        for name, res in self.results.items():
            if not res["err_t"]: continue
            avg_t = np.mean(res["err_t"])
            avg_r = np.mean(res["err_r"])
            avg_s = np.mean(res["err_s"])
            avg_time = np.mean(res["time"])
            print(f"{name:<18} {avg_t:10.2f}px {avg_r:10.3f}° {avg_s:10.4f} {avg_time:10.0f}ms")

    # --- Algorithms ---

    def alg_phase_corr_full(self, ref, mov):
        ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        mov_g = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY)
        ref_e = self._get_edge(ref_g)
        mov_e = self._get_edge(mov_g)
        
        dx, dy = self._phase_corr(ref_e, mov_e)
        return dx, dy, 0.0, 1.0

    def alg_phase_corr_center(self, ref, mov):
        h, w = ref.shape[:2]
        ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        mov_g = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY)
        ref_e = self._get_edge(ref_g)
        mov_e = self._get_edge(mov_g)
        
        # Center 50% crop (25% area)
        mh, mw = int(h * 0.25), int(w * 0.25)
        ref_c = ref_e[mh:h-mh, mw:w-mw]
        mov_c = mov_e[mh:h-mh, mw:w-mw]
        
        dx, dy = self._phase_corr(ref_c, mov_c)
        return dx, dy, 0.0, 1.0

    def alg_ecc_affine(self, ref, mov):
        ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        mov_g = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY)
        
        # ECC needs float32
        # Downscale for speed/stability
        scale = 0.5
        ref_s = cv2.resize(ref_g, None, fx=scale, fy=scale)
        mov_s = cv2.resize(mov_g, None, fx=scale, fy=scale)
        
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-4)
        
        try:
            _, warp = cv2.findTransformECC(mov_s, ref_s, warp, cv2.MOTION_AFFINE, criteria)
            
            # Decompose Affine
            dx = warp[0, 2] / scale
            dy = warp[1, 2] / scale
            
            a = warp[0, 0]
            b = warp[0, 1]
            scale_val = np.sqrt(a*a + b*b)
            rot_rad = np.arctan2(b, a)
            rot_deg = np.degrees(rot_rad)
            
            return dx, dy, rot_deg, scale_val
        except:
            return 0, 0, 0, 1.0

    def alg_sift(self, ref, mov):
        return self._feature_match(ref, mov, cv2.SIFT_create(2000))

    def alg_akaze(self, ref, mov):
        return self._feature_match(ref, mov, cv2.AKAZE_create())

    def alg_lightglue(self, ref, mov):
        if not HAS_TORCH: return 0,0,0,1
        
        device = torch.device("cpu") # or cuda
        
        # Initialize
        extractor = SuperPoint(max_num_keypoints=2048).to(device).eval()
        matcher = LightGlue(features='superpoint').to(device).eval()
        
        # Prepare Data
        ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        mov_g = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY)
        
        # Kornia expects (B, 1, H, W) float normalized [0,1]
        t_ref = kornia.image_to_tensor(ref_g, False).float() / 255.
        t_mov = kornia.image_to_tensor(mov_g, False).float() / 255.
        
        with torch.no_grad():
            feats0 = extractor(t_ref)
            feats1 = extractor(t_mov)
            matches01 = matcher({"image0": feats0, "image1": feats1})
            
            kpts0 = feats0["keypoints"][0]
            kpts1 = feats1["keypoints"][0]
            matches = matches01["matches"][0]
            
            mkpts0 = kpts0[matches[..., 0]].cpu().numpy()
            mkpts1 = kpts1[matches[..., 1]].cpu().numpy()
            
        if len(mkpts0) < 10:
            return 0,0,0,1
            
        return self._estimate_affine(mkpts1, mkpts0) # Mov -> Ref

    # --- Helpers ---
    
    def _phase_corr(self, img1, img2):
        img1 = np.float32(img1)
        img2 = np.float32(img2)
        h, w = img1.shape
        hann = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        shift, _ = cv2.phaseCorrelate(img1 * hann, img2 * hann)
        return -shift[0], -shift[1]
    
    def _get_edge(self, gray):
        blur = cv2.GaussianBlur(gray, (5,5), 1.4)
        return cv2.Canny(blur, 50, 150)
        
    def _feature_match(self, ref, mov, detector):
        ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        mov_g = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY)
        
        kp1, des1 = detector.detectAndCompute(ref_g, None)
        kp2, des2 = detector.detectAndCompute(mov_g, None)
        
        if des1 is None or des2 is None: return 0,0,0,1
        
        # FLANN
        index_params = dict(algorithm=1, trees=5) # 1=KDTREE
        if isinstance(detector, cv2.AKAZE): # AKAZE needs binary matcher
             index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
             
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
        if len(good) < 10: return 0,0,0,1
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        
        return self._estimate_affine(pts2, pts1) # Mov -> Ref
        
    def _estimate_affine(self, pts_mov, pts_ref):
        # Estimate Affine (Scale/Rot/Trans)
        M, inliers = cv2.estimateAffinePartial2D(pts_mov, pts_ref, method=cv2.RANSAC)
        if M is None: return 0,0,0,1
        
        dx = M[0, 2]
        dy = M[1, 2]
        scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
        rot_deg = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
        
        return dx, dy, rot_deg, scale

    def load_gt(self):
        if not os.path.exists(GT_FILE): return {"cases": []}
        with open(GT_FILE, "r", encoding="utf-8") as f: return json.load(f)
        
    def add_gt_interactive(self):
        print("Use manual_align_gui.py directly for better experience.")
        print("Then update json manually.")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bench = AlgorithmBenchmark()
    if "--add-gt" in sys.argv:
        bench.add_gt_interactive()
    else:
        bench.run_all()
