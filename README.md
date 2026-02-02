# Timelapse Auto-Stabilizer

타임랩스 이미지 시퀀스의 흔들림(Translation)과 회전(Rotation)을 자동으로 보정하여 부드러운 영상을 생성하는 도구입니다. 대량의 이미지를 병렬 처리하여 고속으로 안정화합니다.

## ✨ 주요 기능 (Key Features)
- **Zero Drift PID Control**: PID 제어기(비례-적분-미분)와 Global Anchor 전략을 통해 장기간 타임랩스에서도 드리프트(위치/회전 누적 오차)를 완벽하게 제거합니다.
- **Robust Rotation Correction**: Gradient 기반 ECC 알고리즘(반복 횟수 500회)을 사용하여 조명 변화가 심한 환경에서도 미세한 회전을 정밀하게 감지합니다.
- **Smart Dark Removal**: 이미지 중심부 밝기를 분석하여 너무 어두운(밤/새벽) 이미지는 분석 및 결과에서 자동으로 제외합니다 (Threshold 60.0).
- **Uniform Brightness**: 모든 결과 프레임의 밝기를 균일하게 정규화(Normalization)하여 플리커(깜빡임) 없는 영상을 생성합니다. (화질 저하 방지를 위한 Safe Limit 적용)
- **Subfolder & Range Support**: 특정 프로젝트 폴더 및 날짜 범위를 지정하여 부분 처리가 가능합니다.
- **Parallel Processing**: 멀티코어를 활용한 고속 병렬 분석 및 렌더링.
- **Resuming**: 단계별 분석 로그(JSON) 자동 저장으로 중단된 지점부터 재개 가능.

## 📦 설치 (Installation)

### 필요 조건 (Prerequisites)
- Python 3.8+
- OpenCV (`opencv-python`, `opencv-contrib-python`)
- FFmpeg (시스템 PATH에 설치)

### Python 패키지 설치
```bash
pip install -r requirements.txt
```

---

## 🚀 사용법 (Usage)

이 프로젝트의 메인 스크립트는 `timelapse_stabilizer.py`입니다.

### 1. 기본 실행 (전체 처리)
`input/` 루트 폴더 내의 모든 날짜 폴더를 분석합니다.
```bash
python timelapse_stabilizer.py --video
```

### 2. 특정 프로젝트(서브폴더) 실행
`input/MyProject/` 폴더 내의 이미지들을 안정화하고, 결과는 `output/MyProject/`에 저장합니다.
영상 파일명은 `MyProject_날짜범위_해상도_시간.mp4` 형식을 따릅니다.
```bash
python timelapse_stabilizer.py --subfolder MyProject --video
```

### 3. 날짜 범위 지정 및 고화질 설정
특정 기간만 선택하여 처리하고, 1080p 해상도로 영상을 생성합니다.
```bash
python timelapse_stabilizer.py --subfolder MyProject --start 2026-01-05 --end 2026-01-10 --resize-width 1920 --video
```

### 4. 결과 평가 (Evaluation)
안정화 결과를 정량적으로 평가하고(dX, dY, Rot), 로그와 파라미터를 `evaluation_reports/`에 백업합니다.
```bash
# 전체 평가
python evaluate_stabilization.py

# 특정 프로젝트(서브폴더) 평가
python evaluate_stabilization.py --subfolder MyProject
```

---

## ⚙️ 상세 프로세스 (Pipeline)

### 1. 알고리즘 다이어그램 (Algorithm)
이 프로젝트는 **Gradient-based ECC** (회전 정밀 감지)와 **Zero-Drift PID Control** 전략을 사용합니다.

**1. 개별 프레임 분석 (Analysis Logic):**
```text
[입력 이미지] 
    │
    ▼
[Center Crop 밝기 분석] ──(Dark)──> [제거/Skip]
    │ (Pass)
    ▼
[Gradient 변환 (조명 강인성 확보)]
    │
    ├───> [ECC 정밀 회전 감지 (500iter)]
    │           │
    │     [이미지 역회전]
    │           │
    └───────> [Phase Correlation] 
                      │
            [최종 변위(dx, dy) 계산]
```

**2. 전체 처리 파이프라인 (Parallel Pipeline):**
```text
[Step 1: Analysis]   -> 각 폴더 병렬 분석 (Dark Filtering Included) -> analysis_log.json
        │
[Step 2: Refinement] -> [Day 1 vs Day N] Global Anchor 정합 -> refine_log.json
        │
[Step 3: Integration]-> PID 제어기 적용 (Smooth & Zero Drift) -> full_log.txt
        │
[Step 4: Rendering]  -> [Warp] 변환 -> [Brightness Normalization] -> Output Images
```

---

## 📂 파일 구조 (File Structure)

```
project/
├── timelapse_stabilizer.py   # [Main] 병렬 안정화 스크립트
├── evaluate_stabilization.py # [New] 안정화 성능 평가 및 리포트 도구
├── create_video.py           # 비디오 생성 유틸리티 (모듈)
├── dep/                      # 구버전 스크립트 보관
├── input/                    # 입력 이미지 (구조: input/프로젝트명/날짜별폴더)
├── output/                   # 출력 결과물 (구조: output/프로젝트명/)
│   └── 프로젝트명/
│       ├── 2026-01-01/       # 안정화된 이미지들
│       ├── 프로젝트_analysis_log.json
│       ├── 프로젝트_refine_log.json
│       ├── 프로젝트_full_log.txt
│       └── 프로젝트_2026-01-01~2026-01-31_1920p_180000.mp4
└── evaluation_reports/       # 평가 리포트 아카이브
```

## 📝 로그 파일 (Log Files)
상세 분석 데이터는 `output/프로젝트명/` 폴더에 `[프로젝트]_` 접두어와 함께 저장됩니다.

**JSON 로그**: 재개(Resume)를 위한 중간 데이터입니다.
**최종 로그 (`full_log.txt`)**:
- **dx/dy**: 최종 보정된 이동량 (Absolute Pixels)
- **rot**: 최종 보정된 회전각 (Degree)
- **status**: `OK`, `DARK` (제거됨), `ROT(angle)` 등 상태 정보

---

## 🛠️ 고급 옵션 (Advanced Options)

| 옵션 | 설명 | 예시 |
|---|---|---|
| `--subfolder` (`-f`) | `input` 내의 특정 프로젝트 폴더 지정 | `-f MyProject` |
| `--start` / `--end` | 처리할 날짜 범위 지정 (YYYY-MM-DD) | `--start 2026-01-01` |
| `--resize-width` | 영상 가로 해상도 (세로 자동, 예: 1920) | `--resize-width 1920` |
| `--kp`, `--ki`, `--kd` | (코드상수) PID 제어 파라미터 튜닝 | (소스코드 수정 필요) |
| `--force-analyze` | 기존 로그 무시하고 전체 재분석 | `--force-analyze` |
| `--render-only` | 분석된 로그를 바탕으로 렌더링만 수행 | `--render-only` |
