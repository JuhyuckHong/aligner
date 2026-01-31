"""
[ARCHIVED]
Reason: Benchmark V2: Advanced benchmark including LightGlue (failed due to env) and various preprocessors.
Date: 2026-01-31
"""

"""
=============================================================================
util/benchmark_v2.py - 고급 정합 알고리즘 종합 벤치마크 (Fail-Fast 포함)
=============================================================================

사용법:
  ./venv/Scripts/python util/benchmark_v2.py
=============================================================================
"""
import cv2
import numpy as np
import json
import os
import sys
import time
import math

# Force line buffering
sys.stdout.reconfigure(line_buffering=True)

# PyTorch & Kornia
try:
    import torch
    import kornia
    from kornia.feature import LoFTR, LightGlue, SuperPoint
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch/Kornia not found or blocked. Deep Learning methods will be skipped.")
except OSError:
    HAS_TORCH = False
    print("Warning: DLL load failed (Windows Defender). Run unblock script.")

GT_FILE = "util/gt_benchmark.json"

class AlgorithmBenchmark:
    def __init__(self):
        self.results = {}
        self.fail_counts = {}
        self.MAX_FAIL = 2  # 2번 연속 실패/타임아웃 시 해당 알고리즘 영구 Skip
        self.TIMEOUT = 5000 # 5초 제한
        
    def run_all(self):
        data = self.load_gt()
        cases = [c for c in data["cases"] if c.get("gt_dx") is not None]
        
        if not cases:
            print("No GT cases found.")
            return

        print(f"=== Benchmark V2 Results ({len(cases)} cases) ===\n")
        
        # 알고리즘 등록
        self.methods = [
            # ("1.Phase(Full)", self.alg_phase_full),
            # ("2.Gradient+PC", self.alg_grad_phase),
            # ("3.Log+Grad+PC", self.alg_log_grad_phase),
            # ("4.DoG+PC",      self.alg_dog_phase),
            # ("5.Center+Grad", self.alg_center_grad),
            # ("6.Phase+ECC",   self.alg_phase_ecc),
            # ("7.AKAZE",       self.alg_akaze),
            # ("8.SIFT",        self.alg_sift),
        ]
        
        # if HAS_TORCH:
        self.methods.append(("9.LightGlue", self.alg_lightglue))
            
        # 초기화
        for name, _ in self.methods:
            self.results[name] = {"err_t": [], "err_r": [], "err_s": [], "time": [], "fails": 0}
            self.fail_counts[name] = 0

        for case in cases:
            print(f"Case: {case['id']}")
            print(f"  GT: dx={case['gt_dx']:.2f}, dy={case['gt_dy']:.2f}, rot={case.get('gt_angle',0):.2f}, scl={case.get('gt_scale',1):.4f}")
            
            ref = cv2.imread(case["ref"])
            mov = cv2.imread(case["mov"])
            
            if ref is None or mov is None:
                print("  Failed to load images")
                continue
                
            for name, func in self.methods:
                # Skip logic
                # if self.fail_counts[name] >= self.MAX_FAIL:
                #     continue
                    
                start_t = time.time()
                # try:
                if True: # Force Run
                    # Timeout check logic inside func is hard, so we just measure time
                    dx, dy, rot, scale = func(ref, mov)
                    elapsed = (time.time() - start_t) * 1000
                    
                    if elapsed > self.TIMEOUT:
                        raise TimeoutError(f"Timeout {elapsed:.0f}ms")
                        
                    # Error calc
                    gt_dx, gt_dy = case["gt_dx"], case["gt_dy"]
                    gt_rot = case.get("gt_angle", 0.0)
                    gt_scl = case.get("gt_scale", 1.0)
                    
                    err_dx = abs(dx - gt_dx)
                    err_dy = abs(dy - gt_dy)
                    err_t = np.sqrt(err_dx**2 + err_dy**2)
                    err_r = abs(rot - gt_rot)
                    err_s = abs(scale - gt_scl)
                    
                    # Success check (Threshold: 2px)
                    if err_t > 5.0 and elapsed > 1000: # 너무 틀리고 느리면 실패 간주
                         self.fail_counts[name] += 1
                         status = "✗ (High Err)"
                    elif err_t > 2.0:
                         status = "△"
                         self.fail_counts[name] = max(0, self.fail_counts[name]-1) # 성공하면 카운트 감소
                    else:
                         status = "✓"
                         self.fail_counts[name] = 0
                    
                    # self.results[name]["err_t"].append(err_t)
                    # self.results[name]["err_r"].append(err_r)
                    # self.results[name]["err_s"].append(err_s)
                    # self.results[name]["time"].append(elapsed)
                    
                    print(f"  {name:15s}: T={err_t:5.2f}px R={rot:5.2f}° S={scale:5.3f} ({elapsed:4.0f}ms) {status}")
                    
                # except Exception as e:
                #     self.fail_counts[name] += 1
                #     print(f"  {name:15s}: ERROR - {e}")
            print()
            
        self.print_summary()

    def print_summary(self):
        print("\n=== Summary (Lower is Better) ===")
        print(f"{'Algorithm':<15} {'Avg Trans':>10} {'Avg Rot':>10} {'Avg Scale':>10} {'Avg Time':>10} {'Fails':>5}")
        print("-" * 75)
        for name, _ in self.methods:
            res = self.results[name]
            if not res["err_t"]:
                print(f"{name:<15} {'SKIPPED':>10}")
                continue
                
            avg_t = np.mean(res["err_t"])
            avg_r = np.mean(res["err_r"])
            avg_s = np.mean(res["err_s"])
            avg_time = np.mean(res["time"])
            fails = self.fail_counts[name]
            
            print(f"{name:<15} {avg_t:10.2f}px {avg_r:10.3f}° {avg_s:10.4f} {avg_time:10.0f}ms {fails:>5}")

    # --- Preprocessors ---

    def _to_gray(self, img):
        if len(img.shape) == 3: return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _get_grad(self, img):
        g = self._to_gray(img)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _get_log_grad(self, img):
        g = self._to_gray(img).astype(np.float32)
        g_log = np.log1p(g)
        gx = cv2.Sobel(g_log, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g_log, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    def _get_dog(self, img):
        g = self._to_gray(img).astype(np.float32)
        g1 = cv2.GaussianBlur(g, (0,0), 1.0)
        g2 = cv2.GaussianBlur(g, (0,0), 3.0)
        dog = g1 - g2
        return cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _phase_corr(self, img1, img2):
        img1 = np.float32(img1)
        img2 = np.float32(img2)
        h, w = img1.shape
        hann = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        shift, _ = cv2.phaseCorrelate(img1 * hann, img2 * hann)
        return -shift[0], -shift[1]
        
    def _feature_match(self, ref, mov, detector):
        ref_g = self._to_gray(ref)
        mov_g = self._to_gray(mov)
        
        kp1, des1 = detector.detectAndCompute(ref_g, None)
        kp2, des2 = detector.detectAndCompute(mov_g, None)
        
        if des1 is None or des2 is None: return 0,0,0,1
        
        # FLANN
        index_params = dict(algorithm=1, trees=5)
        if isinstance(detector, cv2.AKAZE): 
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
        
        return self._estimate_affine(pts2, pts1)

    def _estimate_affine(self, pts_mov, pts_ref):
        M, inliers = cv2.estimateAffinePartial2D(pts_mov, pts_ref, method=cv2.RANSAC)
        if M is None: return 0,0,0,1
        dx = M[0, 2]
        dy = M[1, 2]
        scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
        rot_deg = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
        return dx, dy, rot_deg, scale

    # --- Algorithms 1~10 ---

    def alg_phase_full(self, ref, mov):
        # 1) Basic Phase Corr
        ref_g = self._to_gray(ref)
        mov_g = self._to_gray(mov)
        # Apply Edge (Canny) as original baseline
        ref_e = cv2.Canny(cv2.GaussianBlur(ref_g, (5,5), 1.4), 50, 150)
        mov_e = cv2.Canny(cv2.GaussianBlur(mov_g, (5,5), 1.4), 50, 150)
        dx, dy = self._phase_corr(ref_e, mov_e)
        return dx, dy, 0.0, 1.0

    def alg_grad_phase(self, ref, mov):
        # 2) Gradient Magnitude
        r = self._get_grad(ref)
        m = self._get_grad(mov)
        dx, dy = self._phase_corr(r, m)
        return dx, dy, 0.0, 1.0

    def alg_log_grad_phase(self, ref, mov):
        # 3) Log + Gradient
        r = self._get_log_grad(ref)
        m = self._get_log_grad(mov)
        dx, dy = self._phase_corr(r, m)
        return dx, dy, 0.0, 1.0
        
    def alg_dog_phase(self, ref, mov):
        # 4) DoG
        r = self._get_dog(ref)
        m = self._get_dog(mov)
        dx, dy = self._phase_corr(r, m)
        return dx, dy, 0.0, 1.0

    def alg_center_grad(self, ref, mov):
        # 5) Center 25% + Gradient
        h, w = ref.shape[:2]
        mh, mw = int(h * 0.25), int(w * 0.25)
        
        r = self._get_grad(ref)[mh:h-mh, mw:w-mw]
        m = self._get_grad(mov)[mh:h-mh, mw:w-mw]
        
        dx, dy = self._phase_corr(r, m)
        return dx, dy, 0.0, 1.0

    def alg_phase_ecc(self, ref, mov):
        # 6) Phase Init + ECC Refine
        # Init with Gradient Phase
        r = self._get_grad(ref).astype(np.float32)
        m = self._get_grad(mov).astype(np.float32)
        dx, dy = self._phase_corr(r, m)
        
        # ECC Refine (using gray images)
        ref_g = self._to_gray(ref).astype(np.float32)
        mov_g = self._to_gray(mov).astype(np.float32)
        
        # Downscale for speed
        scale = 0.5
        ref_s = cv2.resize(ref_g, None, fx=scale, fy=scale)
        mov_s = cv2.resize(mov_g, None, fx=scale, fy=scale)
        
        warp = np.eye(2, 3, dtype=np.float32)
        warp[0, 2] = dx * scale
        warp[1, 2] = dy * scale
        
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-3) # Fast
        try:
             # Use TRANSLATION for speed/stability in refinement
            _, warp = cv2.findTransformECC(mov_s, ref_s, warp, cv2.MOTION_TRANSLATION, criteria)
            return warp[0, 2]/scale, warp[1, 2]/scale, 0.0, 1.0
        except:
            return dx, dy, 0.0, 1.0

    def alg_akaze(self, ref, mov):
        # 7) AKAZE
        return self._feature_match(ref, mov, cv2.AKAZE_create())

    def alg_sift(self, ref, mov):
        # 8) SIFT
        return self._feature_match(ref, mov, cv2.SIFT_create(1000)) # Limit features for speed

    def alg_lightglue(self, ref, mov):
        # 9) LightGlue
        if not HAS_TORCH: return 0,0,0,1
        
        device = torch.device("cpu")
        extractor = SuperPoint(max_num_keypoints=1024).to(device).eval() # Limit 1024
        matcher = LightGlue(features='superpoint').to(device).eval()
        
        ref_g = self._to_gray(ref)
        mov_g = self._to_gray(mov)
        t_ref = kornia.image_to_tensor(ref_g, False).float() / 255.
        t_mov = kornia.image_to_tensor(mov_g, False).float() / 255.
        
        with torch.no_grad():
            feats0 = extractor(t_ref)
            feats1 = extractor(t_mov)
            matches01 = matcher({"image0": feats0, "image1": feats1})
            matches = matches01["matches"][0]
            kpts0 = feats0["keypoints"][0]
            kpts1 = feats1["keypoints"][0]
            
            mkpts0 = kpts0[matches[..., 0]].cpu().numpy()
            mkpts1 = kpts1[matches[..., 1]].cpu().numpy()
            
        print(f"      [LightGlue] Matches: {len(mkpts0)}")
        if len(mkpts0) < 10:
             print("      [LightGlue] Not enough matches -> Returning Identity")
             return 0,0,0,1
             
        res = self._estimate_affine(mkpts1, mkpts0)
        print(f"      [LightGlue] Affine: dx={res[0]:.2f}, dy={res[1]:.2f}, rot={res[2]:.2f}, scl={res[3]:.4f}")
        return res

    def load_gt(self):
        if not os.path.exists(GT_FILE): return {"cases": []}
        with open(GT_FILE, "r", encoding="utf-8") as f: return json.load(f)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bench = AlgorithmBenchmark()
    bench.run_all()
