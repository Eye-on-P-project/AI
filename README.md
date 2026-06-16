# Real-Time Drowsiness Detection Pipeline

실시간 졸음 감지를 위한 **전처리, 눈 상태 CNN 학습, 시계열 졸음 분류 모델 학습** 코드와 전처리된 sequence dataset을 정리한 프로젝트입니다.

전체 흐름은 다음과 같습니다.

```text
Driving video
→ MediaPipe FaceMesh landmark extraction
→ Eye open/closed CNN inference
→ Frame-level feature extraction
→ Frame sequence dataset generation
→ Hierarchical temporal drowsiness classification
→ Sequence / video / streaming-level evaluation
```

## 1. Project Overview

이 프로젝트는 UTA-RLDD 기반 졸음 감지 실험을 모바일/실시간 환경에 맞게 구성한 코드입니다.

주요 목표는 다음과 같습니다.

- MediaPipe FaceMesh로 얼굴 landmark를 추출합니다.
- 직접 학습한 eye open/closed CNN으로 `p_closed`를 계산합니다.
- EAR, MAR, head pose, PERCLOS, blink 통계 등 frame-level feature를 생성합니다.
- 10초 길이의 frame sequence를 만들어 temporal model 학습 데이터로 변환합니다.
- `normal`, `drowsy`, `sleepy` 3단계 졸음 상태를 분류합니다.
- 긴 눈 감김, 높은 PERCLOS, 고개 숙임 rule을 함께 평가합니다.

## 2. Upload Policy

이 저장소는 **원본 raw data는 제외**하고, **전처리 결과와 모델 결과는 포함할 수 있는 구조**로 관리합니다.

| 구분 | GitHub 포함 여부 | 위치 |
|---|---:|---|
| 소스 코드 | 포함 | `frame_sequence_preprocessing/`, `scripts/` |
| 전처리 결과 `.npy`, `.csv`, `.json` | 포함 | `data/processed/frame_sequences/` |
| eye state CNN 모델 파일 | 포함 | `outputs/eye_state_cnn/` |
| temporal classifier 모델 파일 | 포함 | `outputs/temporal_classifier/` |
| 원본 주행 영상 | 제외 | `data/raw/videos/` |
| eye open/closed 원본 이미지 데이터셋 | 제외 | `data/raw/eyes_set/` |
| 로컬 실험 폴더 | 제외 | `Senior Project/`, `runs/tmp/`, `logs/tmp/` |

> 중요: `X_all_inherited.npy`, `X_high_confidence.npy`, `.pth`, `.pt` 같은 파일은 100MB를 넘을 수 있으므로 Git LFS로 관리합니다. 이 저장소에는 `.gitattributes`가 포함되어 있습니다.

## 3. Repository Structure

```text
.
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ .gitattributes
├─ run_preprocessing.py
├─ frame_sequence_preprocessing/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ utils.py
│  ├─ eye_model.py
│  ├─ face_features.py
│  ├─ frame_extraction.py
│  ├─ normalization.py
│  ├─ blink_rolling.py
│  ├─ sequences.py
│  ├─ dataset.py
│  └─ main.py
├─ scripts/
│  ├─ train_eye_state_cnn_gray.py
│  └─ train_temporal_hierarchical_sleepy_rule.py
├─ data/
│  ├─ README.md
│  ├─ raw/
│  │  ├─ README.md
│  │  ├─ videos/          # local only, ignored by git
│  │  ├─ eyes_set/        # local only, ignored by git
│  │  └─ manifests/       # local only, ignored by git
│  └─ processed/
│     ├─ README.md
│     └─ frame_sequences/
│        ├─ README.md
│        ├─ feature_order.json
│        ├─ preprocess_frame_sequence_config.json
│        ├─ rebuild_highconf_binary_config.json
│        ├─ X_all_inherited.npy
│        ├─ y_all_inherited.npy
│        ├─ X_high_confidence.npy
│        ├─ y_high_confidence.npy
│        ├─ sequence_metadata_all_inherited.csv
│        └─ sequence_metadata_high_confidence.csv
├─ outputs/
│  ├─ README.md
│  ├─ eye_state_cnn/
│  │  ├─ README.md
│  │  ├─ best_eye_model_gray1.pth
│  │  ├─ last_eye_model_gray1.pth
│  │  ├─ config.txt
│  │  ├─ loss_curve.png
│  │  ├─ test_confusion_matrix.png
│  │  ├─ test_confusion_matrix_normalized.png
│  │  └─ classification_report.txt
│  └─ temporal_classifier/
│     ├─ README.md
│     └─ hierarchical_sleepy_rule/
│        ├─ global_train_config.json
│        ├─ all_cv_results.csv
│        └─ tcn_high_confidence_hierarchical_rule/
│           ├─ cv_summary_by_fold.csv
│           ├─ cv_summary_mean_std.csv
│           ├─ fold_*/
│           │  ├─ best_model.pt
│           │  ├─ last_model.pt
│           │  ├─ history.csv
│           │  ├─ test_sequence_metrics.json
│           │  ├─ test_video_metrics.json
│           │  ├─ test_streaming_metrics.json
│           │  ├─ test_sequence_predictions.csv
│           │  ├─ test_video_predictions.csv
│           │  └─ test_streaming_predictions.csv
│           └─ final_train/
│              ├─ best_model.pt
│              ├─ last_model.pt
│              ├─ history.csv
│              └─ run_summary.json
└─ docs/
   └─ pipeline_summary.md
```

## 4. Data Directory

### 4.1 Raw data: `data/raw/`

원본 데이터는 용량이 커서 GitHub에 올리지 않습니다.

로컬에서만 다음 구조로 배치합니다.

```text
data/raw/
├─ video_manifest.csv
├─ videos/
│  └─ original driving videos
├─ eyes_set/
│  ├─ train/
│  │  ├─ open/
│  │  └─ closed/
│  ├─ val/
│  │  ├─ open/
│  │  └─ closed/
│  └─ test/
│     ├─ open/
│     └─ closed/
└─ manifests/
   └─ optional backup manifest files
```

`video_manifest.csv`는 전처리를 다시 실행할 때 필요하며, 다음 column을 포함해야 합니다.

```text
video_id, fold, part, subject_id, score_label, video_path
```

### 4.2 Processed data: `data/processed/frame_sequences/`

현재 저장소에 포함되는 전처리 결과 파일은 다음과 같습니다.

| File | Shape / Role |
|---|---|
| `X_all_inherited.npy` | `(42280, 150, 13)`, 전체 video-label 상속 sequence |
| `y_all_inherited.npy` | `(42280,)`, 전체 sequence label |
| `X_high_confidence.npy` | `(17619, 150, 13)`, high-confidence sequence |
| `y_high_confidence.npy` | `(17619,)`, high-confidence label |
| `sequence_metadata_all_inherited.csv` | `42,280` rows, 전체 sequence metadata |
| `sequence_metadata_high_confidence.csv` | `17,619` rows, high-confidence metadata |
| `feature_order.json` | `13`개 frame feature 순서 |
| `preprocess_frame_sequence_config.json` | 전처리 실행 설정 |
| `rebuild_highconf_binary_config.json` | p_closed binary 변환 및 high-confidence 재구성 설정 |

전처리 sequence feature 순서는 다음과 같습니다.

```text
ear_z, p_closed, mar_z, pitch_z, yaw_z, roll_z, current_closed_duration, perclos_10s, avg_blink_duration_10s, avg_blink_amplitude_10s, avg_opening_velocity_10s, blink_frequency_10s, time_since_last_blink
```

## 5. Output Directory

모델 파일과 학습 결과는 `outputs/` 아래에 **모델 종류별로 분리**해서 올립니다.

### 5.1 Eye state CNN output

`scripts/train_eye_state_cnn_gray.py` 실행 결과는 다음 위치에 저장합니다.

```text
outputs/eye_state_cnn/
├─ best_eye_model_gray1.pth
├─ last_eye_model_gray1.pth
├─ config.txt
├─ loss_curve.png
├─ test_confusion_matrix.csv
├─ test_confusion_matrix.png
├─ test_confusion_matrix_normalized.png
└─ classification_report.txt
```

이 모델은 최종 졸음 분류기가 아니라, 전처리 단계에서 각 frame의 눈 감김 확률 `p_closed`를 뽑기 위한 모델입니다.

### 5.2 Temporal classifier output

`scripts/train_temporal_hierarchical_sleepy_rule.py` 실행 결과는 다음 위치에 저장합니다.

```text
outputs/temporal_classifier/hierarchical_sleepy_rule/
├─ global_train_config.json
├─ feature_order_used.json
├─ all_cv_results.csv
└─ tcn_high_confidence_hierarchical_rule/
   ├─ cv_summary_by_fold.csv
   ├─ cv_summary_mean_std.csv
   ├─ final_train_summary.csv
   ├─ fold_1/
   │  ├─ best_model.pt
   │  ├─ last_model.pt
   │  ├─ history.csv
   │  ├─ run_summary.json
   │  ├─ test_sequence_metrics.json
   │  ├─ test_video_metrics.json
   │  ├─ test_streaming_metrics.json
   │  ├─ test_sequence_confusion_matrix.png
   │  ├─ test_video_confusion_matrix.png
   │  ├─ test_streaming_confusion_matrix.png
   │  ├─ test_sequence_predictions.csv
   │  ├─ test_video_predictions.csv
   │  └─ test_streaming_predictions.csv
   └─ final_train/
      ├─ best_model.pt
      ├─ last_model.pt
      ├─ history.csv
      └─ run_summary.json
```

모델을 여러 개 실험할 경우 다음처럼 model/data mode 조합별 폴더를 따로 둡니다.

```text
outputs/temporal_classifier/hierarchical_sleepy_rule/
├─ gru_high_confidence_hierarchical_rule/
├─ lstm_high_confidence_hierarchical_rule/
├─ tcn_high_confidence_hierarchical_rule/
└─ hm_lstm_high_confidence_hierarchical_rule/
```

## 6. Pipeline

### Step 1. Eye open/closed CNN training

`scripts/train_eye_state_cnn_gray.py`는 눈 crop 이미지를 이용해 open/closed 이진 분류 모델을 학습합니다.

- Dataset: `ImageFolder` 기반 `train/val/test`
- Input: grayscale 1-channel image
- Model: `timm` 기반 MobileNetV4 small
- Task: eye open vs eye closed binary classification
- Output: checkpoint, confusion matrix, classification report, loss curve

주요 설정값:

```python
data_root = "data/raw/eyes_set"
save_dir = "outputs/eye_state_cnn"
batch_size = 64
epochs = 50
lr = 1e-5
```

실행:

```bash
python scripts/train_eye_state_cnn_gray.py
```

### Step 2. Frame sequence preprocessing

`frame_sequence_preprocessing/`은 기존 단일 전처리 스크립트를 기능별 모듈로 나눈 버전입니다.

전처리 과정:

1. `video_manifest.csv`를 읽습니다.
2. 각 영상에서 target FPS 기준으로 frame을 샘플링합니다.
3. MediaPipe FaceMesh로 face landmark를 추출합니다.
4. EAR, MAR, pitch, yaw, roll을 계산합니다.
5. 학습된 eye model로 양쪽 눈 crop의 `p_closed`를 계산합니다.
6. subject별 normal 영상 기준으로 z-score normalization을 수행합니다.
7. blink event, PERCLOS, rolling blink statistics를 계산합니다.
8. 고정 길이 frame sequence를 생성합니다.
9. high-confidence sequence를 선별하여 `.npy`와 metadata를 저장합니다.

주요 설정값:

```python
MANIFEST_CSV = "data/raw/video_manifest.csv"
OUT_ROOT = "data/processed/frame_sequences"
EYE_MODEL_WEIGHT_PATH = "outputs/eye_state_cnn/best_eye_model_gray1.pth"
TARGET_FPS = 15
SEQUENCE_SECONDS = 10
```

실행:

```bash
python run_preprocessing.py
```

또는:

```bash
python -m frame_sequence_preprocessing.main
```

### Step 3. Temporal drowsiness classification training

`scripts/train_temporal_hierarchical_sleepy_rule.py`는 전처리된 frame sequence를 이용해 졸음 상태를 분류합니다.

입력 파일:

```text
data/processed/frame_sequences/
├─ X_all_inherited.npy
├─ y_all_inherited.npy
├─ sequence_metadata_all_inherited.csv
├─ feature_order.json
└─ preprocess_frame_sequence_config.json
```

모델 구조:

```text
Stage 1: normal vs abnormal
Stage 2: drowsy vs sleepy
```

지원 encoder:

- GRU
- LSTM
- TCN
- HM-LSTM-style encoder

주요 설정값:

```python
DATA_ROOT = "data/processed/frame_sequences"
OUTPUT_ROOT = "outputs/temporal_classifier"
MODELS_TO_RUN = ["tcn"]
DATA_MODES_TO_RUN = ["high_confidence"]
```

실행:

```bash
python scripts/train_temporal_hierarchical_sleepy_rule.py
```

## 7. Feature Description

| Feature | Description |
|---|---|
| `ear_z` | subject별 기준으로 정규화한 Eye Aspect Ratio |
| `p_closed` | eye open/closed CNN이 예측한 눈 감김 확률 또는 binary eye closed 값 |
| `mar_z` | subject별 기준으로 정규화한 Mouth Aspect Ratio |
| `pitch_z` | 머리 상하 회전 정규화 값 |
| `yaw_z` | 머리 좌우 회전 정규화 값 |
| `roll_z` | 머리 기울기 정규화 값 |
| `current_closed_duration` | 현재 눈 감김 지속 시간 |
| `perclos_10s` | 최근 10초 기준 눈 감김 비율 |
| `avg_blink_duration_10s` | 최근 10초 평균 blink duration |
| `avg_blink_amplitude_10s` | 최근 10초 평균 blink amplitude |
| `avg_opening_velocity_10s` | 최근 10초 평균 eye opening velocity |
| `blink_frequency_10s` | 최근 10초 blink frequency |
| `time_since_last_blink` | 마지막 blink 이후 경과 시간 |

## 8. Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scriptsctivate
```

Install packages:

```bash
pip install -r requirements.txt
```

CUDA 환경에서는 자신의 CUDA 버전에 맞는 PyTorch wheel을 먼저 설치한 뒤 나머지 패키지를 설치하는 것을 권장합니다.

## 9. Git LFS Setup

전처리 결과 `.npy`와 모델 checkpoint는 일반 Git 파일 크기 제한에 걸릴 수 있으므로 Git LFS를 사용합니다.

처음 한 번만 실행합니다.

```bash
git lfs install
```

이 저장소에는 이미 `.gitattributes`가 포함되어 있으므로, 다음 확장자는 LFS 대상으로 추적됩니다.

```text
*.npy
*.npz
*.pth
*.pt
*.onnx
*.pkl
*.joblib
```

처음 커밋 예시:

```bash
git init
git lfs install
git add .
git commit -m "Add drowsiness detection pipeline"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

## 10. Notes

- `data/raw/` 내부 원본 데이터는 `.gitignore`로 제외됩니다.
- `data/processed/frame_sequences/` 내부 전처리 결과는 GitHub에 포함하는 구조입니다.
- `outputs/eye_state_cnn/`과 `outputs/temporal_classifier/` 내부 모델 파일 및 평가 결과도 포함할 수 있습니다.
- `.npy`, `.pth`, `.pt` 파일은 반드시 Git LFS로 올리는 것을 권장합니다.
- 전처리 결과만으로 temporal classifier 학습은 바로 실행할 수 있습니다.
- eye model을 다시 학습하거나 전처리를 다시 수행하려면 raw data가 로컬에 필요합니다.
