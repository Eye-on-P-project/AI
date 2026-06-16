from __future__ import annotations

import numpy as np

# ============================================================
# Global config
# ============================================================
CONFIG = {
    # ============================================================
    # 1. 기본 입출력 설정
    # ============================================================

    # 이전 단계에서 만든 video_manifest.csv 경로
    "MANIFEST_CSV": r"data/raw/video_manifest.csv",

    # 전처리 결과 저장 폴더
    # 기존 pre_data와 헷갈리지 않게 새 폴더 사용 권장
    "OUT_ROOT": r"data/processed/frame_sequences",

    # 처음 테스트할 때는 3 정도로 설정
    # 전체 영상을 처리할 때는 None으로 변경
    "PROCESS_ONLY_FIRST_N_VIDEOS": None,

    # True면 이미 만들어진 결과 파일은 다시 만들지 않고 건너뜀
    # 처음부터 다시 만들 때는 False
    "RESUME": False,

    # 병렬 처리 worker 수
    # GPU로 eye model을 사용할 경우 multiprocessing이 불안정할 수 있어서 1 권장
    "NUM_WORKERS": 1,


    # ============================================================
    # 2. 프레임 추출 및 얼굴 landmark 설정
    # ============================================================

    # 원본 영상에서 초당 몇 프레임을 사용할지 설정
    # 예: 원본 30fps 영상에서 TARGET_FPS=15면 약 2프레임마다 1개 처리
    "TARGET_FPS": 15,

    # MediaPipe 얼굴 landmark 추출 시 사용할 최대 이미지 너비
    # 값이 작을수록 빠르지만 landmark 정확도가 떨어질 수 있음
    "DETECT_MAX_WIDTH": 640,

    # MediaPipe 얼굴 검출 최소 신뢰도
    "MIN_DETECTION_CONFIDENCE": 0.5,

    # MediaPipe 얼굴 추적 최소 신뢰도
    "MIN_TRACKING_CONFIDENCE": 0.5,

    # True면 눈 입 주변 landmark가 더 정교해짐
    # 속도는 조금 느려질 수 있음
    "REFINE_LANDMARKS": True,


    # ============================================================
    # 3. 눈 open closed 모델 설정
    # ============================================================

    # True면 직접 학습한 눈 감김 모델을 사용
    # 이 모델은 최종 졸음 분류용이 아니라 p_closed 추출용
    "USE_EYE_MODEL": True,

    # 사용할 모델 로드 방식
    # timm: timm 모델 구조 사용
    # torchscript: torchscript 모델 사용
    # custom_python: 직접 만든 모델 클래스 사용
    "EYE_MODEL_BACKEND": "timm",

    # timm으로 학습한 MobileNetV4Small 모델 이름
    "EYE_MODEL_TIMM_NAME": "mobilenetv4_conv_small.e2400_r224_in1k",

    # 눈 모델 클래스 수
    # open closed 이진 분류라서 2
    "EYE_NUM_CLASSES": 2,

    # 직접 학습한 눈 모델 weight 경로
    "EYE_MODEL_WEIGHT_PATH": r"outputs/eye_state_cnn/best_eye_model_gray1.pth",

    # custom_python 사용 시 모델 코드 파일 경로
    "CUSTOM_MODEL_PY": r"",

    # custom_python 사용 시 모델 클래스 이름
    "CUSTOM_MODEL_CLASS": r"",

    # custom_python 모델 생성자에 넘길 인자
    "CUSTOM_MODEL_KWARGS": {},

    # checkpoint 안에서 state_dict를 찾을 후보 key
    "CHECKPOINT_STATE_KEYS": ["state_dict", "model_state_dict", "model", "net"],

    # False면 일부 layer 이름이 맞지 않아도 가능한 weight만 로드
    "LOAD_STRICT": False,

    # 눈 모델을 실행할 장치
    # auto cuda cpu 중 선택
    "EYE_MODEL_DEVICE": "cuda",

    # 눈 crop 이미지를 모델에 넣기 전에 resize할 크기
    "EYE_MODEL_INPUT_SIZE": 224,

    # 눈 모델 추론 batch size
    "EYE_MODEL_BATCH_SIZE": 64,

    # 모델 출력 해석 방식
    # auto: 출력 shape을 보고 자동 판단
    # softmax: [B,2] 출력
    # sigmoid: [B,1] 출력
    "EYE_MODEL_OUTPUT_MODE": "auto",

    # closed 클래스 인덱스
    # 보통 open=0 closed=1이면 1
    "EYE_CLOSED_CLASS_INDEX": 1,

    # 눈 모델 학습 때 사용한 normalization 값
    # ImageNet normalization으로 학습했으면 그대로 사용
    "EYE_MODEL_MEAN": [0.485, 0.456, 0.406],
    "EYE_MODEL_STD": [0.229, 0.224, 0.225],

    # 눈 crop bbox에 여유를 얼마나 줄지 설정
    # 값이 클수록 눈 주변 영역을 더 넓게 crop
    "EYE_CROP_PADDING": 0.65,


    # ============================================================
    # 4. p_closed 기반 눈 감김 및 blink 검출 설정
    # ============================================================

    # p_closed를 최근 몇 프레임 평균으로 부드럽게 만들지 설정
    # 실시간성을 고려해 현재 프레임 이전 값만 사용하는 causal smoothing 적용
    "P_CLOSED_SMOOTH_WINDOW": 5,

    # p_closed_smooth가 이 값 이상이면 눈 감김 시작으로 판단
    "P_CLOSED_START_THRESHOLD": 0.70,

    # p_closed_smooth가 이 값 이하로 내려가면 눈이 다시 떠졌다고 판단
    "P_CLOSED_END_THRESHOLD": 0.30,

    # 최근 구간에서 눈 감김 비율이 이 값 이상이면 강한 졸음 단서로 봄
    "PERCLOS_THRESHOLD": 0.50,

    # blink로 인정할 최소 지속 시간
    # 너무 짧으면 모델 오검출이나 노이즈일 가능성이 큼
    "MIN_BLINK_DURATION_SEC": 0.05,

    # blink로 인정할 최대 지속 시간
    # 이 값보다 길면 일반 blink가 아니라 긴 눈 감김 상태로 따로 반영
    "MAX_BLINK_DURATION_SEC": 2.00,

    # blink로 인정하기 위한 최소 EAR 변화량
    # z-score 기준으로 너무 작으면 실제 blink가 아닐 수 있음
    "MIN_BLINK_AMPLITUDE_Z": 0.10,

    # blink 사이 최소 간격
    # 너무 가까운 blink 후보는 중복 검출일 수 있음
    "MIN_BLINK_GAP_SEC": 0.05,

    # 이 시간 이상 눈을 감고 있으면 prolonged closure로 간주
    # blink가 끝나기 전이라도 current_closed_duration feature에 반영
    "PROLONGED_CLOSURE_SEC": 1.5,


    # ============================================================
    # 5. Subject별 normal 기준 calibration 설정
    # ============================================================

    # calibration 기준으로 사용할 영상 score
    # 0은 normal 영상
    "CALIBRATION_SCORE_LABEL": "0",

    # normal 영상 중 몇 비율을 calibration에 사용할지 설정
    # 1/3이면 앞쪽 또는 뒤쪽 1/3 사용
    "CALIBRATION_RATIO": 1 / 3,

    # calibration 구간 위치
    # first: 영상 앞부분
    # last: 영상 뒷부분
    "CALIBRATION_PART": "first",

    # calibration 계산에 필요한 최소 유효 프레임 수
    "MIN_CALIBRATION_VALID_FRAMES": 30,

    # 얼굴 검출 실패 등으로 빈 feature가 있을 때 보간할 최대 프레임 수
    "INTERPOLATE_LIMIT_FRAMES": 5,


    # ============================================================
    # 6. Frame sequence 데이터셋 생성 설정
    # ============================================================

    # sequence 하나가 몇 초 구간을 볼지 설정
    # TARGET_FPS=15 SEQUENCE_SECONDS=10이면 T=150
    "SEQUENCE_SECONDS": 10,

    # sequence를 몇 초 간격으로 만들지 설정
    # 학습 데이터의 중복을 줄이기 위해 5초 stride 사용
    # 실시간 추론은 여전히 sliding window로 1초마다 수행 가능
    "SEQUENCE_STRIDE_SECONDS": 5,

    # rolling feature 계산에 사용할 과거 시간 범위
    # perclos blink frequency 평균 blink duration 등이 이 구간 기준으로 계산됨
    "ROLLING_WINDOW_SECONDS": 10,

    # 마지막 blink 이후 경과 시간을 최대 몇 초로 제한할지 설정
    # 너무 큰 값이 feature를 지배하지 않도록 cap 적용
    "TIME_SINCE_LAST_BLINK_CAP_SEC": 10,

    # sequence 내 얼굴 검출 성공률이 이 값보다 낮으면 제외
    "MIN_FACE_RATE_PER_SEQUENCE": 0.50,

    # sequence별 npy 파일을 따로 저장할지 여부
    "SAVE_PER_SEQUENCE_NPY": True,

    # 전체 X y 배열을 하나의 npy로 저장할지 여부
    "SAVE_ALL_ARRAYS": True,

    # True면 전체 video-label 상속 데이터셋도 저장
    # 이번 버전은 high-confidence만 사용할 것이므로 False 권장
    "SAVE_ALL_INHERITED_DATASET": False,


    # ============================================================
    # 7. 신뢰도 높은 sequence 선별 설정 시작
    # ============================================================
    # 이 아래 설정들은 video label을 그대로 상속한 전체 데이터셋 중에서
    # 라벨과 feature 패턴이 잘 맞는 sequence만 따로 골라내기 위한 설정이다.
    #
    # 결과적으로 데이터셋은 두 가지가 생성된다.
    #
    # 1. 전체 video label 상속 데이터셋
    #    모든 sequence에 원본 video label을 그대로 부여
    #
    # 2. high-confidence 데이터셋
    #    drowsiness_score 기준으로 라벨 신뢰도가 높은 sequence만 사용
    # ============================================================

    # True면 high-confidence 데이터셋을 생성
    "MAKE_HIGH_CONFIDENCE_DATASET": True,

    # ------------------------------------------------------------
    # high-confidence 라벨 선별 기준
    # ------------------------------------------------------------
    # 핵심 변경점
    # - sleepy는 하품으로 고르지 않음
    # - sleepy는 눈을 오래 감은 상태 또는 고개가 아래로 푹 숙여진 상태로 선별
    # - drowsy는 하품 고개 변화 약한 눈 피로를 포함하지만 sleepy 조건은 제외
    # - normal은 눈 감김 하품 고개 변화가 모두 낮은 sequence만 선별

    # normal 선별 기준
    "HIGH_CONF_NORMAL_MAX_EYE_SLEEP_SCORE": 0.15,
    "HIGH_CONF_NORMAL_MAX_DROWSY_SCORE": 0.25,
    "HIGH_CONF_NORMAL_MAX_HEAD_DOWN_SCORE": 0.30,

    # drowsy 선별 기준
    "HIGH_CONF_DROWSY_MIN_SCORE": 0.25,
    "HIGH_CONF_DROWSY_MAX_SCORE": 0.70,
    "HIGH_CONF_DROWSY_MAX_EYE_SLEEP_SCORE": 0.55,
    "HIGH_CONF_DROWSY_MAX_CLOSED_DURATION": 2.0,
    "HIGH_CONF_DROWSY_MAX_HEAD_DOWN_SCORE": 0.70,

    # sleepy 선별 기준
    # sleepy는 하품이 아니라 장시간 눈 감김 또는 고개 숙임으로 선별한다.
    "HIGH_CONF_SLEEPY_MIN_EYE_SLEEP_SCORE": 0.55,
    "HIGH_CONF_SLEEPY_MIN_CLOSED_DURATION": 2.0,
    "HIGH_CONF_SLEEPY_MIN_PERCLOS": 0.70,
    "HIGH_CONF_SLEEPY_MIN_P_CLOSED": 0.75,
    "HIGH_CONF_SLEEPY_MIN_HEAD_DOWN_SCORE": 0.75,

    # 고개 숙임 방향 설정
    # MediaPipe solvePnP의 pitch 부호는 환경에 따라 다르게 해석될 수 있다.
    # 처음에는 abs를 권장하고 검증 이미지로 방향을 확인한 뒤 positive 또는 negative로 바꾸면 더 정확하다.
    # possible values: "abs", "positive", "negative"
    "HEAD_DOWN_DIRECTION": "abs",

    # sleepy 선별용 eye score 가중치
    # 하품은 포함하지 않는다.
    "EYE_SLEEP_SCORE_WEIGHTS": {
        "perclos": 0.40,
        "closed_duration": 0.35,
        "p_closed": 0.20,
        "blink_duration": 0.05,
    },

    # drowsy 선별용 score 가중치
    # 하품과 고개 변화는 drowsy 보조 단서로 사용한다.
    "DROWSY_SCORE_WEIGHTS": {
        "perclos": 0.20,
        "p_closed": 0.15,
        "yawn": 0.30,
        "head": 0.20,
        "blink_duration": 0.15,
    },

    # score 계산 시 각 요소를 0~1 범위로 정규화하기 위한 cap
    "SCORE_CAPS": {
        "perclos": 0.60,
        "closed_duration": 2.0,
        "p_closed": 0.80,
        "yawn_z": 3.0,
        "head_abs_z": 3.0,
        "head_down_z": 3.0,
        "blink_duration_sec": 0.60,
    },


    # ============================================================
    # 8. 라벨 매핑 설정
    # ============================================================

    # UTA-RLDD score를 프로젝트 라벨로 변환
    "SCORE_TO_LABEL": {
        "0": "normal",
        "5": "drowsy",
        "10": "sleepy",
    },

    # 학습에 사용할 라벨 id
    "LABEL_TO_ID": {
        "normal": 0,
        "drowsy": 1,
        "sleepy": 2,
    },
}

RAW_NORM_COLS = ["mean_ear", "mar", "pitch", "yaw", "roll"]

Z_COLS = ["ear_z", "mar_z", "pitch_z", "yaw_z", "roll_z"]

FRAME_FEATURE_COLS = [
    "ear_z",
    "p_closed",
    "mar_z",
    "pitch_z",
    "yaw_z",
    "roll_z",
    "current_closed_duration",
    "perclos_10s",
    "avg_blink_duration_10s",
    "avg_blink_amplitude_10s",
    "avg_opening_velocity_10s",
    "blink_frequency_10s",
    "time_since_last_blink",
]

LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]

RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

LEFT_EYE_CROP_IDX = [33, 133, 160, 158, 153, 144, 159, 145]

RIGHT_EYE_CROP_IDX = [362, 263, 385, 387, 373, 380, 386, 374]

MOUTH_LEFT = 61

MOUTH_RIGHT = 291

MOUTH_TOP = 13

MOUTH_BOTTOM = 14

HEAD_POSE_IDXS = {
    "nose": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

HEAD_MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],        # nose tip
    [0.0, -330.0, -65.0],   # chin
    [-225.0, 170.0, -135.0],# left eye outer
    [225.0, 170.0, -135.0], # right eye outer
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0],
], dtype=np.float64)
