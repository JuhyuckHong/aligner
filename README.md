# 🎬 Timelapse Image Aligner

타임랩스 촬영 이미지의 흔들림을 보정하고 영상으로 만드는 도구 모음입니다.

---

## 📁 프로젝트 구조

```
aligner/
├── stabilize_phase.py     # 자동 흔들림 보정 (Phase Correlation)
├── create_video.py        # 메모리 효율적인 영상 생성
├── requirements.txt       # Python 의존성
├── util/
│   └── manual_align_gui.py  # 수동 정렬 GUI 도구
└── README.md
```

---

## 🔧 설치

### 필수 요구사항
- Python 3.8+
- FFmpeg (시스템 PATH에 설치)

### Python 패키지 설치
```bash
pip install -r requirements.txt
```

---

## 🛠️ 도구 설명

### 1. `stabilize_phase.py` - 자동 흔들림 보정

Phase Correlation 알고리즘을 사용하여 이미지 시퀀스의 흔들림을 자동으로 보정합니다.

**알고리즘:**
- Canny Edge Detection으로 엣지 이미지 생성
- Phase Correlation으로 프레임 간 이동량 계산
- 중간 프레임을 기준으로 정렬 (조명 변화에 강건)
- 이상치(outlier) 자동 필터링

**사용법:**
```bash
python stabilize_phase.py
```

**설정 (코드 내 수정):**
```python
input_dir = "input"      # 입력 폴더
output_dir = "output"  # 출력 폴더
```

**특징:**
| 항목 | 설명 |
|------|------|
| 보정 유형 | 평행이동 (Translation) |
| 기준 프레임 | 중간 프레임 (n // 2) |
| 최대 이동량 | 100px (초과 시 이전 값 사용) |
| 출력 품질 | JPEG Quality 98 |

---

### 2. `create_video.py` - 메모리 효율적 영상 생성

대량의 이미지를 배치로 나눠 처리하여 메모리 부족 문제를 해결합니다.

**동작 원리:**
1. 이미지를 배치(기본 200장)로 분할
2. 각 배치를 개별 MP4로 인코딩
3. 생성된 MP4들을 스트림 복사로 빠르게 병합
4. 임시 파일 자동 정리

**사용법:**
```bash
python create_video.py --input INPUT_FOLDER --output OUTPUT.mp4 [OPTIONS]
```

**옵션:**
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input`, `-i` | 입력 이미지 폴더 | (필수) |
| `--output`, `-o` | 출력 영상 파일 | `output.mp4` |
| `--fps` | 초당 프레임 수 | `30` |
| `--crf` | 품질 (0-51, 낮을수록 고품질) | `18` |
| `--batch` | 배치당 이미지 수 | `200` |
| `--ext` | 이미지 확장자 | `jpg` |

**예시:**
```bash
# 기본 설정
python create_video.py -i output -o timelapse.mp4

# 고품질 + 60fps
python create_video.py -i output -o timelapse_hq.mp4 --fps 60 --crf 12

# 메모리 부족 시 배치 크기 줄이기
python create_video.py -i output -o timelapse.mp4 --batch 100
```

---

### 3. `util/manual_align_gui.py` - 수동 정렬 GUI

두 이미지를 비교하며 수동으로 정렬 오프셋을 조정하는 GUI 도구입니다.

**기능:**
- **Main View**: 전체 이미지 표시
- **Zoom View**: 마우스 위치 기준 4배 확대
- 토글/오버레이로 정밀 비교 가능

**조작법:**
| 키 | 동작 |
|---|---|
| `W` / `A` / `S` / `D` | 1px 이동 (상/좌/하/우) |
| `I` / `J` / `K` / `L` | 10px 이동 (상/좌/하/우) |
| `SPACE` | Reference ↔ Aligned 토글 |
| `Z` | Overlay 모드 (반투명 겹침) |
| `마우스 이동` | 확대 위치 지정 |
| `Q` / `ESC` | 종료 (오프셋 출력) |

**사용법:**
```bash
# 두 이미지 직접 지정
python util/manual_align_gui.py --ref reference.jpg --mov moving.jpg

# 폴더의 처음 두 이미지 사용
python util/manual_align_gui.py --input-dir input

# 기본값 (input 폴더)
python util/manual_align_gui.py
```

---

## 📋 일반적인 워크플로우

### 1. 흔들림 보정 → 영상 생성 (자동)

```bash
# Step 1: 이미지 폴더를 input_dir에 복사

# Step 2: 자동 보정 실행
python stabilize_phase.py

# Step 3: 영상 생성
python create_video.py -i output -o timelapse.mp4
```

### 2. 수동 정렬 확인

```bash
# 보정 결과 확인
python util/manual_align_gui.py --ref original/img1.jpg --mov stabilized/img1.jpg
```

---

## ⚙️ CRF 품질 가이드

| CRF | 설명 | 용도 |
|-----|------|------|
| 0 | 무손실 | 아카이브 |
| 12-14 | 매우 고품질 | 전문가용 |
| **18** | 고품질 (기본값) | 일반 사용 |
| 23 | 중간 품질 | 웹 업로드 |
| 28+ | 저품질 | 미리보기 |

---

## 🚨 문제 해결

### FFmpeg 메모리 부족
```bash
# 배치 크기 줄이기
python create_video.py -i INPUT -o OUTPUT.mp4 --batch 100
```

### 보정이 잘 안 될 때
- 첫 번째 프레임이 흐리거나 이상한 경우 문제 발생 가능
- `stabilize_phase.py`는 중간 프레임을 기준으로 사용
- 극심한 흔들림은 수동 정렬 권장

### 이미지 파일명 형식
- 권장: `YYYY-MM-DD_HH-MM-SS.jpg`
- 정렬 순서: 파일명 알파벳순 (sorted)

---

## 📝 라이선스

개인 및 상업적 사용 자유
