import os
import json
from glob import glob

def get_img(folder, hour):
    pattern = os.path.join(folder, f"*_{hour:02d}-00-*.jpg")
    files = sorted(glob(pattern))
    if files: return files[0].replace('\\', '/')
    pattern = sorted(glob(os.path.join(folder, f"*_{hour:02d}-*.jpg")))
    if files: return files[0].replace('\\', '/')
    return None

def generate_gt_cases():
    base_dir = "input"
    folders = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    cases = []
    
    # Existing 01-01 to 01-02 transition GT
    case_0102 = {
        "id": "day_trans_2026-01-01_to_2026-01-02",
        "type": "day_transition",
        "ref": "input/2026-01-01/2026-01-01_18-00-00.jpg",
        "mov": "input/2026-01-02/2026-01-02_06-00-00.jpg",
        "gt_dx": 1.0, "gt_dy": -1.0, "notes": "Measured"
    }
    cases.append(case_0102)
    
    for i, folder in enumerate(folders):
        path = os.path.join(base_dir, folder)
        f = get_img(path, 6)
        n = get_img(path, 12)
        l = get_img(path, 18)
        
        # Intra-day
        if f and n:
            cases.append({
                "id": f"intra_{folder}_first_noon",
                "type": "intra_day", "ref": f, "mov": n,
                "gt_dx": None, "gt_dy": None, "notes": ""
            })
        if n and l:
            cases.append({
                "id": f"intra_{folder}_noon_last",
                "type": "intra_day", "ref": n, "mov": l,
                "gt_dx": None, "gt_dy": None, "notes": ""
            })
            
        # Transitions & Same Time
        if i < len(folders) - 1:
            next_folder = folders[i+1]
            next_path = os.path.join(base_dir, next_folder)
            
            nf = get_img(next_path, 6)
            nn = get_img(next_path, 12)
            nl = get_img(next_path, 18)
            
            # Skip first transition if already manually added above
            if not (folder == "2026-01-01" and next_folder == "2026-01-02"):
                if l and nf:
                    cases.append({
                        "id": f"day_trans_{folder}_to_{next_folder}",
                        "type": "day_transition", "ref": l, "mov": nf,
                        "gt_dx": None, "gt_dy": None, "notes": ""
                    })
            
            if n and nn:
                cases.append({
                    "id": f"same_time_{folder}_{next_folder}_noon",
                    "type": "same_time", "ref": n, "mov": nn,
                    "gt_dx": None, "gt_dy": None, "notes": ""
                })

    result = {
        "description": "GT (ground truth) for alignment algorithm evaluation",
        "case_types": {
            "day_transition": "전날 밤(18:00) \u2192 다음날 아침(06:00) - 조도 차이 큼",
            "same_time": "전날 vs 다음날 같은 시간 비교 - 조도 동일",
            "intra_day": "같은 날 내 비교 (first/noon/last)"
        },
        "cases": cases
    }
    
    with open("util/gt_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    generate_gt_cases()
