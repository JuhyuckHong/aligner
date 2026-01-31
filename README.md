# Timelapse Auto-Stabilizer

타임랩스 이미지 시퀀스의 흔들림(Translation)과 회전(Rotation)을 자동으로 보정하여 부드러운 영상을 생성하는 도구입니다. 대량의 이미지를 병렬 처리하여 고속으로 안정화합니다.

## ✨ 주요 기능 (Key Features)
- **Hybrid Alignment**: Phase Correlation (이동) + ECC (회전) 알고리즘 결합.
- **Parallel Processing**: 멀티코어를 활용한 고속 병렬 분석 및 렌더링.
- **Virtual Refinement**: 가상 정합(Virtual Alignment)을 통해 정확한 Day-to-Day Drift 보정.
- **Early Day Refinement**: 아침 시간대에 누적 오차를 서서히 보정하여 영상 끊김 방지.
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

### 1. 기본 실행 (분석 + 렌더링 + 영상 생성)
```bash
python timelapse_stabilizer.py --video
```
- `input/` 폴더의 모든 이미지를 분석하고 안정화합니다.
- `output/` 폴더에 결과 이미지를 저장합니다.
- `output/combined.mp4` 영상을 생성합니다.

### 2. 고화질/1080p 영상 생성 (Resizing)
```bash
python timelapse_stabilizer.py --video --resize-width 1920
```
- 결과 영상을 1920px(FHD) 너비로 리사이징하여 생성합니다.

### 3. 고속 렌더링 모드 (Render Only)
코드를 수정했거나 렌더링 옵션만 바꾸고 싶을 때, 분석 과정(Phase 1, 2)을 건너뛰고 렌더링만 다시 수행합니다.
```bash
python timelapse_stabilizer.py --video --render-only
```
- 기존 `output/full_log.txt`를 읽어서 렌더링만 수행합니다.

### 4. 강제 재분석 (Force Analyze)
이미 로그 파일이 있더라도 처음부터 다시 분석하고 싶을 때 사용합니다.
```bash
python timelapse_stabilizer.py --force-analyze --video
```

---

## ⚙️ 상세 프로세스 (Pipeline)

### 1. 알고리즘 다이어그램 (Algorithm)
이 프로젝트는 **Hybrid Alignment** (Phase Correlation + ECC)와 **Early Day Refinement** 전략을 사용합니다.

**1. 개별 프레임 분석 (Analysis Logic):**
```text
[입력 이미지] -> [Rotation 감지 (ECC)]
                      │
            ┌─────────┴─────────┐
         (회전 발견)          (회전 없음)
            │                   │
    [이미지 역회전]        [원본 사용]
            │                   │
            └────> [Gradient Phase Correlation] 
                           │
                 [최종 변위(dx, dy) 계산]
```

**2. 전체 처리 파이프라인 (Parallel Pipeline):**
```text
[Step 1: Analysis]   -> 각 폴더 병렬 분석 (Frame-by-Frame Motion) -> analysis_log.json
        │
[Step 2: Refinement] -> 가상 정합(Virtual Warp)으로 Day Drift 측정 -> refine_log.json
        │
[Step 3: Integration]-> 데이터 연결 및 Early Correction 적용      -> full_log.txt
        │
[Step 4: Rendering]  -> 최종 좌표로 이미지 변환(Warp) 및 저장    -> Output Images
```

**Step 4: Rendering 상세 (Transformation Logic):**
```text
[원본 이미지]
    │
    ▼
[1. 회전 (Rotation)] : 중심점(Center) 기준 누적 회전각 적용
    │
    ▼
[2. 이동 (Translation)] : 회전된 이미지에 누적 이동량(dx, dy) 적용
    │
    ▼
[최종 이미지 저장]
```

---

## 📂 파일 구조 (File Structure)

```
project/
├── timelapse_stabilizer.py # [Main] 병렬 안정화 스크립트
├── create_video.py       # 비디오 생성 유틸리티
├── util/
│   └── manual_align_gui.py # 수동 정합 테스트/검증 도구
├── dep/
│   └── stabilize_phase.py # (구버전) 단일 스레드 안정화 스크립트
├── input/                # 입력 이미지 (날짜별 폴더 구조 권장)
├── output/      # 출력 결과물
    ├── analysis_log.json # Phase 1 분석 결과
    ├── refine_log.json   # Phase 2 Refine 결과
    ├── full_log.txt      # 최종 궤적 로그
    └── combined.mp4      # 최종 영상
└── requirements.txt      # 의존성 패키지
```

## 📝 로그 파일 (Log Files)
상세 분석 데이터는 `output/` 폴더에 저장됩니다.

**JSON 로그 (`analysis_log.json` / `refine_log.json`)**:
- 스크립트 내부 재개(Resume)를 위한 중간 데이터입니다.

**최종 로그 (`full_log.txt`)**:
```text
Folder        Filename                    dx=X.X    dy=Y.Y    rot=R.RRR    status=Status
2026-01-28    2026-01-28_14-00-00.jpg     dx=-19.8  dy=15.1   rot=-0.181   status=ROT(-0.18)
```
- **dx/dy**: 누적 이동량 (픽셀, Absolute)
- **rot**: 누적 회전각 (도, Degree)
- **status**: `OK` (일반), `ROT` (회전보정됨), `FIRST` (기준)

---

## 🛠️ 유틸리티 (Utilities)

### Manual Align GUI
알고리즘이 제대로 작동했는지 눈으로 확인하고 싶을 때 사용합니다. 두 이미지(Ref, Mov)를 겹쳐서 비교할 수 있습니다.

```bash
# 사용 예시
python util/manual_align_gui.py --ref output/day1/img.jpg --mov output/day2/img.jpg
```
- **WASD / 방향키**: 이동
- **U/O**: 회전
- **Z**: 깜빡임 비교 (Overlay Toggle)

---

## ⚠️ Deprecated (Legacy)

### `dep/stabilize_phase.py`
초기 버전의 단일 스레드 안정화 스크립트입니다. `Phase Correlation` 기반으로 `Day Refinement` 로직이 포함되어 있습니다.
현재는 `timelapse_stabilizer.py`로 대체되었으며, 참고용으로 남겨두었습니다.

사용법:
```bash
python dep/stabilize_phase.py --input input --output output_phase
```
