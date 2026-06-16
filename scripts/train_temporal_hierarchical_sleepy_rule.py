# -*- coding: utf-8 -*-
"""
Hierarchical temporal training for real-time drowsiness classification.

목적:
    기존 3-class 단일 분류가 drowsy와 sleepy를 많이 혼동하는 문제를 줄이기 위해
    1단계 normal vs abnormal
    2단계 drowsy vs sleepy
    구조로 학습한다.

추가:
    실사용성을 위해 긴 눈 감김과 높은 PERCLOS가 나타나는 sequence는
    sleepy 확률을 보정하는 rule을 평가와 추론 결과에 함께 적용한다.

입력:
    X_all_inherited.npy                 [N, T, D]
    y_all_inherited.npy                 [N]
    sequence_metadata_all_inherited.csv
    feature_order.json

실행:
    python train_temporal_hierarchical_sleepy_rule.py
"""

import os
import json
import math
import time
import shutil
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss,
)

warnings.filterwarnings("ignore", category=UserWarning)


CONFIG = {
    # ------------------------------------------------------------
    # 1. 데이터 경로 설정
    # ------------------------------------------------------------
    "DATA_ROOT": r"data/processed/frame_sequences",

    "X_ALL_FILE": "X_all_inherited.npy",
    "Y_ALL_FILE": "y_all_inherited.npy",
    "METADATA_ALL_FILE": "sequence_metadata_all_inherited.csv",

    "FEATURE_ORDER_FILE": "feature_order.json",
    "PREPROCESS_CONFIG_FILE": "preprocess_frame_sequence_config.json",

    "OUTPUT_ROOT": r"outputs/temporal_classifier",
    "EXPERIMENT_TAG": "hierarchical_sleepy_rule",

    # ------------------------------------------------------------
    # 2. 실행 설정
    # ------------------------------------------------------------
    # 현재 코드는 hierarchical task 전용이다.
    # 모델 구조만 gru, lstm, tcn, hm_lstm 중에서 바꿔 실험 가능하다.
    "DATA_MODES_TO_RUN": ["high_confidence"],
    "MODELS_TO_RUN": ["tcn"],

    # high_confidence로 학습하더라도 test fold는 전체 sequence로 평가한다.
    # 실제 사용 상황을 보기 위해 True 권장.
    "EVAL_HIGH_CONF_ON_FULL_TEST": True,

    # ------------------------------------------------------------
    # 3. Fold 기반 학습 설정
    # ------------------------------------------------------------
    "RUN_FOLD_CV": True,
    "RUN_FINAL_TRAIN_AFTER_CV": True,
    "FOLD_COLUMN": "fold",
    "TEST_FOLDS": None,
    "VAL_GROUP_COLUMN": "subject_id",
    "VAL_RATIO": 0.2,
    "ALLOW_SEQUENCE_FALLBACK_SPLIT": True,

    # ------------------------------------------------------------
    # 4. 학습 하이퍼파라미터
    # ------------------------------------------------------------
    "SEED": 42,
    "DEVICE": "auto",
    "NUM_WORKERS": 8,
    "PIN_MEMORY": True,

    "BATCH_SIZE": 128,
    "EPOCHS": 80,
    "LEARNING_RATE": 5e-4,
    "WEIGHT_DECAY": 5e-4,
    "USE_AMP": True,
    "GRAD_CLIP_NORM": 1.0,

    "EARLY_STOPPING_PATIENCE": 12,
    "EARLY_STOPPING_MIN_DELTA": 1e-4,
    # 실사용 목적에서는 rule 적용 전 모델 자체의 streaming macro-F1을 기준으로 best model을 고른다.
    # rule은 별도 결과로 저장해서 sleepy recall 보정 효과만 확인한다.
    "EARLY_STOPPING_MONITOR": "val_streaming_macro_f1",
    "EARLY_STOPPING_MODE": "max",

    "USE_LR_SCHEDULER": True,
    "LR_SCHEDULER_FACTOR": 0.5,
    "LR_SCHEDULER_PATIENCE": 4,

    # high-confidence 데이터가 이미 선별되어 있고 fold별 불균형이 크지 않으면 False가 더 안정적이다.
    # drowsy recall이 너무 낮으면 True로 바꿔 재실험한다.
    "USE_CLASS_WEIGHT": False,

    # train set 기준 feature standardization
    "STANDARDIZE_FEATURES": True,
    "FEATURE_STD_EPS": 1e-6,

    # weak label 완화
    "LABEL_SMOOTHING": 0.03,

    # ------------------------------------------------------------
    # 5. 클래스 설정
    # ------------------------------------------------------------
    "NUM_CLASSES": 3,
    "CLASS_NAMES": ["normal", "drowsy", "sleepy"],

    # ------------------------------------------------------------
    # 6. 계층형 학습 설정
    # ------------------------------------------------------------
    # 1단계: normal vs abnormal loss
    # 2단계: drowsy vs sleepy loss
    # drowsy와 sleepy 분리가 약하면 STAGE_LOSS_WEIGHT를 1.2~2.0 범위로 올려볼 수 있다.
    "ABNORMAL_LOSS_WEIGHT": 1.0,
    "STAGE_LOSS_WEIGHT": 1.0,

    # stage head는 y>0인 샘플만 loss 계산
    # abnormal 샘플이 batch에 없을 때는 stage loss를 0으로 처리

    # ------------------------------------------------------------
    # 7. 공통 모델 설정
    # ------------------------------------------------------------
    "HIDDEN_DIM": 128,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.3,
    "BIDIRECTIONAL": False,
    "RNN_POOLING": "last",

    # ------------------------------------------------------------
    # 8. TCN 설정
    # ------------------------------------------------------------
    # 32 64 64가 안 좋았다면 아래처럼 다시 조금 키워서 실험 권장
    "TCN_CHANNELS": [64, 128, 128],
    "TCN_KERNEL_SIZE": 5,
    "TCN_DROPOUT": 0.30,
    "TCN_POOLING": "mean",

    # ------------------------------------------------------------
    # 9. HM-LSTM-style 설정
    # ------------------------------------------------------------
    "HM_CHUNK_SIZE": 10,
    "HM_LOWER_HIDDEN_DIM": 96,
    "HM_UPPER_HIDDEN_DIM": 128,

    # ------------------------------------------------------------
    # 10. Sleepy 보정 rule 설정
    # ------------------------------------------------------------
    # rule은 최종 3-class 성능을 올리는 목적보다 실사용 경고 보정용이다.
    # 따라서 best model 선정 기준에는 rule 결과를 쓰지 않고 별도 metric으로 저장한다.
    "USE_SLEEPY_RULE": True,

    # boost: sleepy 확률을 특정 값 이상으로 올린 뒤 normalize
    # force: rule이 걸리면 무조건 sleepy로 예측하므로 오탐이 커질 수 있다.
    "SLEEPY_RULE_MODE": "boost",

    # 실시간 기준으로 sequence 마지막 frame 상태를 우선 사용한다.
    # sequence 내부에서 매우 긴 눈 감김이 있었으면 max duration도 함께 본다.
    "RULE_USE_LAST_FRAME": True,

    # p_closed는 전처리 v5에서 이진값으로 저장된다.
    # 그래서 threshold는 0.5 기준으로 둔다.
    # rule은 너무 쉽게 켜지지 않게 closed duration과 perclos 기준을 이전보다 보수적으로 설정했다.
    "RULE_CLOSED_DURATION_SEC": 2.0,
    "RULE_MAX_CLOSED_DURATION_SEC": 3.0,
    "RULE_PERCLOS_THRESHOLD": 0.70,
    "RULE_PCLOSED_THRESHOLD": 0.50,

    # sleepy는 하품이 아니라 긴 눈 감김 또는 고개 숙임으로만 보정한다.
    "RULE_USE_YAWN": False,
    "RULE_MAR_Z_THRESHOLD": 2.5,

    # 고개 숙임 rule은 오탐을 줄이기 위해 기준을 보수적으로 둔다.
    "RULE_USE_HEAD": True,
    "RULE_HEAD_ABS_Z_THRESHOLD": 2.8,

    # boost 방식일 때 sleepy 확률 최소값
    "RULE_SLEEPY_MIN_PROB": 0.65,

    # normal 확률이 매우 높으면 rule 보정을 막는다.
    "RULE_NORMAL_GUARD_PROB": 0.95,

    # ------------------------------------------------------------
    # 11. Streaming 평가 설정
    # ------------------------------------------------------------
    "STREAMING_PROB_SMOOTH_WINDOW": 3,
    "STREAMING_WARMUP_WINDOWS": 0,
    "VIDEO_AGGREGATION": "majority_vote",  # majority_vote or prob_mean
    "SAVE_PREDICTIONS": True,
    "MAX_SAMPLES": None,
}


# ============================================================
# Utilities
# ============================================================

def apply_default_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config)
    if cfg["EARLY_STOPPING_MONITOR"] == "val_loss":
        cfg["EARLY_STOPPING_MODE"] = "min"
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Use CPU.")
        return torch.device("cpu")
    return torch.device(device_cfg)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)


def copy_if_exists(src: Path, dst_dir: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst_dir / src.name)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return default
        return float(x)
    except Exception:
        return default


def sort_folds(folds: List[Any]) -> List[Any]:
    def key_func(x):
        s = str(x)
        digits = "".join([c for c in s if c.isdigit()])
        return (s.rstrip(digits), int(digits) if digits else 9999, s)
    return sorted(list(folds), key=key_func)


def read_feature_order(data_root: Path, config: Dict[str, Any]) -> List[str]:
    path = data_root / config["FEATURE_ORDER_FILE"]
    if not path.exists():
        print("[WARN] feature_order.json not found. Use fallback feature names.")
        return [
            "ear_z", "p_closed", "mar_z", "pitch_z", "yaw_z", "roll_z",
            "current_closed_duration", "perclos_10s", "avg_blink_duration_10s",
            "avg_blink_amplitude_10s", "avg_opening_velocity_10s",
            "blink_frequency_10s", "time_since_last_blink",
        ]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "frame_feature_order" in data:
        return list(data["frame_feature_order"])
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Unknown feature_order format: {path}")


# ============================================================
# Data loading and split
# ============================================================

def load_all_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Path, List[str]]:
    data_root = Path(config["DATA_ROOT"])
    x_path = data_root / config["X_ALL_FILE"]
    y_path = data_root / config["Y_ALL_FILE"]
    meta_path = data_root / config["METADATA_ALL_FILE"]

    if not x_path.exists():
        raise FileNotFoundError(f"X file not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"y file not found: {y_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata file not found: {meta_path}")

    print(f"[LOAD] {x_path}")
    X = np.load(x_path)
    y = np.load(y_path).astype(np.int64)
    meta = pd.read_csv(meta_path)
    feature_order = read_feature_order(data_root, config)

    if len(X) != len(y) or len(X) != len(meta):
        raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}, meta={len(meta)}")

    if config["MAX_SAMPLES"] is not None:
        n = int(config["MAX_SAMPLES"])
        X = X[:n]
        y = y[:n]
        meta = meta.iloc[:n].reset_index(drop=True)

    required_cols = [config["FOLD_COLUMN"], config["VAL_GROUP_COLUMN"], "video_id", "label_id", "label"]
    for col in required_cols:
        if col not in meta.columns:
            raise ValueError(f"metadata missing required column: {col}")

    if "is_high_confidence" not in meta.columns:
        print("[WARN] metadata has no is_high_confidence. high_confidence mode will use all sequences.")
        meta["is_high_confidence"] = 1

    print(f"[DATA] X shape: {X.shape}, y shape: {y.shape}, metadata: {meta.shape}")
    print("[DATA] class count:", dict(pd.Series(y).value_counts().sort_index()))
    print("[DATA] fold count:", dict(meta[config["FOLD_COLUMN"]].value_counts()))
    print("[DATA] high-confidence count:", int(meta["is_high_confidence"].astype(int).sum()))
    print("[DATA] feature order:", feature_order)
    return X, y, meta, data_root, feature_order


def get_eligible_mask(meta: pd.DataFrame, data_mode: str) -> np.ndarray:
    if data_mode == "all":
        return np.ones(len(meta), dtype=bool)
    if data_mode == "high_confidence":
        return meta["is_high_confidence"].astype(int).to_numpy() == 1
    raise ValueError(f"Unknown data mode: {data_mode}")


def choose_val_groups(
    meta_subset: pd.DataFrame,
    y_subset: np.ndarray,
    group_col: str,
    val_ratio: float,
    seed: int,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    rng = np.random.default_rng(seed)
    groups = np.array(meta_subset[group_col].astype(str).values)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return np.array([], dtype=int), np.array([], dtype=int), False

    target_counts = np.bincount(y_subset.astype(int), minlength=num_classes).astype(float)
    target_dist = target_counts / max(target_counts.sum(), 1.0)

    n_val_groups = max(1, int(round(len(unique_groups) * val_ratio)))
    n_val_groups = min(n_val_groups, len(unique_groups) - 1)

    best_score = float("inf")
    best_val_groups = None
    tries = min(1000, max(100, len(unique_groups) * 20))

    for _ in range(tries):
        val_groups = rng.choice(unique_groups, size=n_val_groups, replace=False)
        val_mask = np.isin(groups, val_groups)
        train_mask = ~val_mask
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        val_counts = np.bincount(y_subset[val_mask].astype(int), minlength=num_classes).astype(float)
        train_counts = np.bincount(y_subset[train_mask].astype(int), minlength=num_classes).astype(float)
        missing_penalty = np.sum(val_counts == 0) * 10.0 + np.sum(train_counts == 0) * 10.0
        val_dist = val_counts / max(val_counts.sum(), 1.0)
        score = np.abs(val_dist - target_dist).sum() + missing_penalty
        if score < best_score:
            best_score = score
            best_val_groups = val_groups

    if best_val_groups is None:
        return np.array([], dtype=int), np.array([], dtype=int), False

    val_mask = np.isin(groups, best_val_groups)
    train_mask = ~val_mask
    return np.where(train_mask)[0], np.where(val_mask)[0], True


def stratified_sequence_split(
    y_subset: np.ndarray,
    val_ratio: float,
    seed: int,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for c in range(num_classes):
        idx = np.where(y_subset == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio))) if len(idx) > 1 else 0
        val_indices.extend(idx[:n_val].tolist())
        train_indices.extend(idx[n_val:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.array(train_indices, dtype=int), np.array(val_indices, dtype=int)


def make_fold_split(
    meta: pd.DataFrame,
    y: np.ndarray,
    config: Dict[str, Any],
    test_fold: Any,
    data_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fold_col = config["FOLD_COLUMN"]
    fold_values = meta[fold_col].astype(str).to_numpy()
    test_fold_str = str(test_fold)
    base_eligible = get_eligible_mask(meta, data_mode)
    test_fold_mask = fold_values == test_fold_str

    if data_mode == "high_confidence" and config["EVAL_HIGH_CONF_ON_FULL_TEST"]:
        test_mask = test_fold_mask
    else:
        test_mask = test_fold_mask & base_eligible
    trainval_mask = (~test_fold_mask) & base_eligible

    trainval_indices = np.where(trainval_mask)[0]
    test_indices = np.where(test_mask)[0]
    if len(trainval_indices) == 0:
        raise ValueError(f"No trainval samples for fold={test_fold}, data_mode={data_mode}")
    if len(test_indices) == 0:
        raise ValueError(f"No test samples for fold={test_fold}, data_mode={data_mode}")

    meta_trainval = meta.iloc[trainval_indices].reset_index(drop=True)
    y_trainval = y[trainval_indices]

    train_local, val_local, ok = choose_val_groups(
        meta_subset=meta_trainval,
        y_subset=y_trainval,
        group_col=config["VAL_GROUP_COLUMN"],
        val_ratio=config["VAL_RATIO"],
        seed=config["SEED"],
        num_classes=config["NUM_CLASSES"],
    )
    if not ok:
        if not config["ALLOW_SEQUENCE_FALLBACK_SPLIT"]:
            raise RuntimeError("Group split failed and fallback is disabled.")
        print("[WARN] Group split failed. Use sequence-level stratified split fallback.")
        train_local, val_local = stratified_sequence_split(
            y_subset=y_trainval,
            val_ratio=config["VAL_RATIO"],
            seed=config["SEED"],
            num_classes=config["NUM_CLASSES"],
        )

    return trainval_indices[train_local], trainval_indices[val_local], test_indices


def make_final_train_val_split(
    meta: pd.DataFrame,
    y: np.ndarray,
    config: Dict[str, Any],
    data_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    eligible = get_eligible_mask(meta, data_mode)
    pool_indices = np.where(eligible)[0]
    meta_pool = meta.iloc[pool_indices].reset_index(drop=True)
    y_pool = y[pool_indices]
    train_local, val_local, ok = choose_val_groups(
        meta_subset=meta_pool,
        y_subset=y_pool,
        group_col=config["VAL_GROUP_COLUMN"],
        val_ratio=config["VAL_RATIO"],
        seed=config["SEED"] + 999,
        num_classes=config["NUM_CLASSES"],
    )
    if not ok:
        if not config["ALLOW_SEQUENCE_FALLBACK_SPLIT"]:
            raise RuntimeError("Final group split failed and fallback is disabled.")
        print("[WARN] Final group split failed. Use sequence-level stratified split fallback.")
        train_local, val_local = stratified_sequence_split(
            y_subset=y_pool,
            val_ratio=config["VAL_RATIO"],
            seed=config["SEED"] + 999,
            num_classes=config["NUM_CLASSES"],
        )
    return pool_indices[train_local], pool_indices[val_local]


# ============================================================
# Dataset
# ============================================================

class SequenceDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.X = X
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = self.X[idx].astype(np.float32, copy=True)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        y = int(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), idx


def compute_train_standardizer(
    X: np.ndarray,
    train_indices: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x_train = X[train_indices].astype(np.float32)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0).astype(np.float32)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return mean.reshape(1, -1), std.reshape(1, -1)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    config: Dict[str, Any],
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    shuffle: bool,
) -> DataLoader:
    ds = SequenceDataset(X, y, indices, mean=mean, std=std)
    return DataLoader(
        ds,
        batch_size=config["BATCH_SIZE"],
        shuffle=shuffle,
        num_workers=config["NUM_WORKERS"],
        pin_memory=bool(config["PIN_MEMORY"] and torch.cuda.is_available()),
        drop_last=False,
    )


# ============================================================
# Models
# ============================================================

class RNNEncoder(nn.Module):
    def __init__(self, input_dim: int, config: Dict[str, Any], rnn_type: str):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.pooling = config["RNN_POOLING"]
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=config["HIDDEN_DIM"],
            num_layers=config["NUM_LAYERS"],
            batch_first=True,
            dropout=config["DROPOUT"] if config["NUM_LAYERS"] > 1 else 0.0,
            bidirectional=config["BIDIRECTIONAL"],
        )
        self.out_dim = config["HIDDEN_DIM"] * (2 if config["BIDIRECTIONAL"] else 1)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        if self.pooling == "mean":
            h = out.mean(dim=1)
        elif self.pooling == "last":
            h = out[:, -1, :]
        else:
            raise ValueError(f"Unknown RNN pooling: {self.pooling}")
        return self.norm(h)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.downsample(x))


class TCNEncoder(nn.Module):
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__()
        channels = list(config["TCN_CHANNELS"])
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=int(config["TCN_KERNEL_SIZE"]),
                    dilation=2 ** i,
                    dropout=float(config["TCN_DROPOUT"]),
                )
            )
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.pooling = config["TCN_POOLING"]
        self.out_dim = channels[-1]
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.tcn(x.transpose(1, 2))
        if self.pooling == "mean":
            h = y.mean(dim=2)
        elif self.pooling == "last":
            h = y[:, :, -1]
        else:
            raise ValueError(f"Unknown TCN pooling: {self.pooling}")
        return self.norm(h)


class HMLSTMStyleEncoder(nn.Module):
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__()
        self.chunk_size = int(config["HM_CHUNK_SIZE"])
        self.lower = nn.LSTM(input_dim, config["HM_LOWER_HIDDEN_DIM"], batch_first=True)
        self.upper = nn.LSTM(config["HM_LOWER_HIDDEN_DIM"], config["HM_UPPER_HIDDEN_DIM"], batch_first=True)
        self.out_dim = config["HM_UPPER_HIDDEN_DIM"]
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        chunk = self.chunk_size
        pad_len = (chunk - (T % chunk)) % chunk
        if pad_len > 0:
            x = torch.cat([x, x.new_zeros(B, pad_len, D)], dim=1)
            T = x.shape[1]
        num_chunks = T // chunk
        x_chunks = x.view(B, num_chunks, chunk, D).reshape(B * num_chunks, chunk, D)
        lower_out, _ = self.lower(x_chunks)
        chunk_emb = lower_out[:, -1, :].view(B, num_chunks, -1)
        upper_out, _ = self.upper(chunk_emb)
        return self.norm(upper_out[:, -1, :])


class HierarchicalDrowsinessModel(nn.Module):
    def __init__(self, input_dim: int, config: Dict[str, Any], model_name: str):
        super().__init__()
        model_name = model_name.lower()
        if model_name == "gru":
            self.encoder = RNNEncoder(input_dim, config, rnn_type="gru")
        elif model_name == "lstm":
            self.encoder = RNNEncoder(input_dim, config, rnn_type="lstm")
        elif model_name == "tcn":
            self.encoder = TCNEncoder(input_dim, config)
        elif model_name == "hm_lstm":
            self.encoder = HMLSTMStyleEncoder(input_dim, config)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        out_dim = self.encoder.out_dim
        drop = float(config["DROPOUT"])
        self.dropout = nn.Dropout(drop)
        self.abnormal_head = nn.Linear(out_dim, 2)  # 0 normal, 1 abnormal
        self.stage_head = nn.Linear(out_dim, 2)     # 0 drowsy, 1 sleepy

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        h = self.dropout(h)
        abnormal_logits = self.abnormal_head(h)
        stage_logits = self.stage_head(h)
        return {
            "abnormal_logits": abnormal_logits,
            "stage_logits": stage_logits,
        }


def hierarchical_probs(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    abn = F.softmax(outputs["abnormal_logits"], dim=1)
    stage = F.softmax(outputs["stage_logits"], dim=1)
    p_normal = abn[:, 0:1]
    p_drowsy = abn[:, 1:2] * stage[:, 0:1]
    p_sleepy = abn[:, 1:2] * stage[:, 1:2]
    probs = torch.cat([p_normal, p_drowsy, p_sleepy], dim=1)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return probs


# ============================================================
# Loss and sleepy rule
# ============================================================

def compute_class_weight_binary(target: np.ndarray, device: torch.device) -> Optional[torch.Tensor]:
    counts = np.bincount(target.astype(int), minlength=2).astype(np.float32)
    if np.any(counts == 0):
        return None
    weights = counts.sum() / (2 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


class HierarchicalLoss(nn.Module):
    def __init__(self, config: Dict[str, Any], y_train: np.ndarray, device: torch.device):
        super().__init__()
        label_smoothing = float(config["LABEL_SMOOTHING"])
        abn_target = (y_train > 0).astype(np.int64)
        stage_target = (y_train[y_train > 0] - 1).astype(np.int64)

        abn_weight = compute_class_weight_binary(abn_target, device) if config["USE_CLASS_WEIGHT"] else None
        stage_weight = compute_class_weight_binary(stage_target, device) if config["USE_CLASS_WEIGHT"] and len(stage_target) > 0 else None

        self.abnormal_ce = nn.CrossEntropyLoss(weight=abn_weight, label_smoothing=label_smoothing)
        self.stage_ce = nn.CrossEntropyLoss(weight=stage_weight, label_smoothing=label_smoothing)
        self.abnormal_loss_weight = float(config["ABNORMAL_LOSS_WEIGHT"])
        self.stage_loss_weight = float(config["STAGE_LOSS_WEIGHT"])

    def forward(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        abnormal_target = (target > 0).long()
        abnormal_loss = self.abnormal_ce(outputs["abnormal_logits"], abnormal_target)

        abnormal_mask = target > 0
        if abnormal_mask.any():
            stage_target = (target[abnormal_mask] - 1).long()
            stage_loss = self.stage_ce(outputs["stage_logits"][abnormal_mask], stage_target)
        else:
            stage_loss = outputs["stage_logits"].sum() * 0.0

        loss = self.abnormal_loss_weight * abnormal_loss + self.stage_loss_weight * stage_loss
        parts = {
            "abnormal_loss": float(abnormal_loss.detach().cpu().item()),
            "stage_loss": float(stage_loss.detach().cpu().item()),
        }
        return loss, parts


def _get_feature_idx(feature_order: List[str], name: str) -> Optional[int]:
    return feature_order.index(name) if name in feature_order else None


def build_sleepy_rule_mask(
    raw_X: np.ndarray,
    probs: np.ndarray,
    feature_order: List[str],
    config: Dict[str, Any],
) -> np.ndarray:
    if not config["USE_SLEEPY_RULE"]:
        return np.zeros(len(raw_X), dtype=bool)

    X = np.nan_to_num(raw_X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    last = X[:, -1, :]

    idx_p_closed = _get_feature_idx(feature_order, "p_closed")
    idx_closed = _get_feature_idx(feature_order, "current_closed_duration")
    idx_perclos = _get_feature_idx(feature_order, "perclos_10s")
    idx_mar = _get_feature_idx(feature_order, "mar_z")
    idx_pitch = _get_feature_idx(feature_order, "pitch_z")
    idx_yaw = _get_feature_idx(feature_order, "yaw_z")
    idx_roll = _get_feature_idx(feature_order, "roll_z")

    mask = np.zeros(len(X), dtype=bool)

    if idx_closed is not None:
        last_closed = last[:, idx_closed]
        max_closed = X[:, :, idx_closed].max(axis=1)
        mask |= last_closed >= float(config["RULE_CLOSED_DURATION_SEC"])
        mask |= max_closed >= float(config["RULE_MAX_CLOSED_DURATION_SEC"])

    if idx_perclos is not None:
        last_perclos = last[:, idx_perclos]
        mask |= last_perclos >= float(config["RULE_PERCLOS_THRESHOLD"])

    if idx_p_closed is not None and idx_perclos is not None:
        last_p = last[:, idx_p_closed]
        last_perclos = last[:, idx_perclos]
        mask |= (last_p >= float(config["RULE_PCLOSED_THRESHOLD"])) & (last_perclos >= 0.40)

    if config["RULE_USE_YAWN"] and idx_mar is not None:
        # 하품만으로 sleepy를 강제하지 않고 눈 감김이나 perclos와 같이 나타날 때만 반영
        mar_max = X[:, :, idx_mar].max(axis=1)
        eye_related = mask.copy()
        if idx_p_closed is not None:
            eye_related |= last[:, idx_p_closed] >= 0.60
        mask |= (mar_max >= float(config["RULE_MAR_Z_THRESHOLD"])) & eye_related

    if config["RULE_USE_HEAD"]:
        head_mask = np.zeros(len(X), dtype=bool)
        for idx in [idx_pitch, idx_yaw, idx_roll]:
            if idx is not None:
                head_mask |= np.abs(last[:, idx]) >= float(config["RULE_HEAD_ABS_Z_THRESHOLD"])
        mask |= head_mask

    # normal guard
    # normal 확률이 아주 높은데 긴 눈감김 조건이 없으면 불필요한 false alarm 방지
    normal_guard = probs[:, 0] >= float(config["RULE_NORMAL_GUARD_PROB"])
    if idx_closed is not None:
        strong_closed = X[:, :, idx_closed].max(axis=1) >= float(config["RULE_CLOSED_DURATION_SEC"])
        mask = mask & (~normal_guard | strong_closed)
    else:
        mask = mask & (~normal_guard)

    return mask


def apply_sleepy_rule(
    probs: np.ndarray,
    raw_X: np.ndarray,
    feature_order: List[str],
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    probs = probs.astype(np.float64, copy=True)
    rule_mask = build_sleepy_rule_mask(raw_X, probs, feature_order, config)
    if not config["USE_SLEEPY_RULE"]:
        preds = probs.argmax(axis=1).astype(int)
        return probs.astype(np.float32), preds

    if config["SLEEPY_RULE_MODE"] == "force":
        probs[rule_mask, :] = 0.0
        probs[rule_mask, 2] = 1.0
    elif config["SLEEPY_RULE_MODE"] == "boost":
        min_p = float(config["RULE_SLEEPY_MIN_PROB"])
        for i in np.where(rule_mask)[0]:
            if probs[i, 2] < min_p:
                remain = max(1e-8, 1.0 - min_p)
                other_sum = probs[i, 0] + probs[i, 1]
                if other_sum <= 1e-8:
                    probs[i, 0] = 0.0
                    probs[i, 1] = remain
                else:
                    probs[i, 0] = probs[i, 0] / other_sum * remain
                    probs[i, 1] = probs[i, 1] / other_sum * remain
                probs[i, 2] = min_p
    else:
        raise ValueError(f"Unknown SLEEPY_RULE_MODE: {config['SLEEPY_RULE_MODE']}")

    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)
    preds = probs.argmax(axis=1).astype(int)
    return probs.astype(np.float32), preds


# ============================================================
# Training and evaluation
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: HierarchicalLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    use_amp: bool,
    grad_clip_norm: Optional[float],
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_abn_loss = 0.0
    total_stage_loss = 0.0
    all_preds = []
    all_targets = []

    for x, target, _idx in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            outputs = model(x)
            loss, parts = criterion(outputs, target)

        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        with torch.no_grad():
            probs = hierarchical_probs(outputs)
            preds = torch.argmax(probs, dim=1)

        bs = len(target)
        total_loss += float(loss.item()) * bs
        total_abn_loss += parts["abnormal_loss"] * bs
        total_stage_loss += parts["stage_loss"] * bs
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return {
        "loss": total_loss / max(len(y_true), 1),
        "abnormal_loss": total_abn_loss / max(len(y_true), 1),
        "stage_loss": total_stage_loss / max(len(y_true), 1),
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: HierarchicalLoss,
    device: torch.device,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Dict[str, float]]:
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []
    all_indices = []
    total_loss = 0.0
    total_abn_loss = 0.0
    total_stage_loss = 0.0

    for x, target, idx in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            outputs = model(x)
            loss, parts = criterion(outputs, target)
            probs = hierarchical_probs(outputs)
        preds = torch.argmax(probs, dim=1)
        bs = len(target)
        total_loss += float(loss.item()) * bs
        total_abn_loss += parts["abnormal_loss"] * bs
        total_stage_loss += parts["stage_loss"] * bs
        all_probs.append(probs.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())
        all_indices.append(idx.numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    loss = total_loss / max(len(targets), 1)
    parts = {
        "abnormal_loss": total_abn_loss / max(len(targets), 1),
        "stage_loss": total_stage_loss / max(len(targets), 1),
    }
    return probs, preds, targets, indices, loss, parts


def sequence_metrics(y_true: np.ndarray, y_pred: np.ndarray, loss: float) -> Dict[str, float]:
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    out = {}
    for i, name in enumerate(class_names):
        mask = y_true == i
        out[f"{name}_recall"] = float((y_pred[mask] == i).mean()) if mask.sum() > 0 else float("nan")
    normal_mask = y_true == 0
    out["normal_false_alarm_rate"] = float((y_pred[normal_mask] != 0).mean()) if normal_mask.sum() > 0 else float("nan")
    return out


def video_level_predictions(
    probs: np.ndarray,
    y_true: np.ndarray,
    indices: np.ndarray,
    meta: pd.DataFrame,
    config: Dict[str, Any],
    suffix: str = "",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = meta.iloc[indices].copy().reset_index(drop=True)
    df["true_label_id"] = y_true.astype(int)
    for c in range(config["NUM_CLASSES"]):
        df[f"prob_{c}"] = probs[:, c]
    df["pred_label_id"] = probs.argmax(axis=1).astype(int)

    rows = []
    for video_id, g in df.groupby("video_id"):
        true_label = int(g["true_label_id"].value_counts().idxmax())
        if config["VIDEO_AGGREGATION"] == "majority_vote":
            pred_label = int(g["pred_label_id"].value_counts().idxmax())
            prob_mean = g[[f"prob_{c}" for c in range(config["NUM_CLASSES"])]].mean().to_numpy()
        elif config["VIDEO_AGGREGATION"] == "prob_mean":
            prob_mean = g[[f"prob_{c}" for c in range(config["NUM_CLASSES"])]].mean().to_numpy()
            pred_label = int(np.argmax(prob_mean))
        else:
            raise ValueError(f"Unknown VIDEO_AGGREGATION: {config['VIDEO_AGGREGATION']}")
        row = {"video_id": video_id, "num_sequences": int(len(g)), "true_label_id": true_label, "pred_label_id": pred_label}
        for c in range(config["NUM_CLASSES"]):
            row[f"prob_mean_{c}"] = float(prob_mean[c])
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out, {"loss": float("nan"), "accuracy": float("nan"), "macro_f1": float("nan")}
    yt = out["true_label_id"].to_numpy().astype(int)
    yp = out["pred_label_id"].to_numpy().astype(int)
    prob_mat = out[[f"prob_mean_{c}" for c in range(config["NUM_CLASSES"])]].to_numpy()
    try:
        vloss = log_loss(yt, prob_mat, labels=list(range(config["NUM_CLASSES"])))
    except Exception:
        vloss = float("nan")
    metrics = {
        "loss": float(vloss),
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
    }
    metrics.update(class_recalls(yt, yp, config["CLASS_NAMES"]))
    return out, metrics


def streaming_predictions(
    probs: np.ndarray,
    y_true: np.ndarray,
    indices: np.ndarray,
    meta: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    smooth_k = int(config["STREAMING_PROB_SMOOTH_WINDOW"])
    warmup = int(config["STREAMING_WARMUP_WINDOWS"])
    num_classes = config["NUM_CLASSES"]

    df = meta.iloc[indices].copy().reset_index(drop=True)
    df["true_label_id"] = y_true.astype(int)
    for c in range(num_classes):
        df[f"prob_{c}"] = probs[:, c]

    sort_cols = [c for c in ["video_id", "start_time", "end_time", "start_frame", "sequence_id"] if c in df.columns]
    if "video_id" not in sort_cols:
        sort_cols = ["video_id"]

    all_rows = []
    for video_id, g in df.groupby("video_id"):
        g = g.sort_values(sort_cols).reset_index(drop=True)
        prob_arr = g[[f"prob_{c}" for c in range(num_classes)]].to_numpy()
        smooth_probs = []
        for i in range(len(g)):
            left = max(0, i - smooth_k + 1)
            smooth_probs.append(prob_arr[left:i + 1].mean(axis=0))
        smooth_probs = np.asarray(smooth_probs)
        preds = smooth_probs.argmax(axis=1).astype(int)
        for i, row in g.iterrows():
            if i < warmup:
                continue
            out = {
                "video_id": video_id,
                "sequence_id": row.get("sequence_id", ""),
                "start_time": safe_float(row.get("start_time", np.nan), np.nan),
                "end_time": safe_float(row.get("end_time", np.nan), np.nan),
                "true_label_id": int(row["true_label_id"]),
                "pred_label_id": int(preds[i]),
            }
            for c in range(num_classes):
                out[f"smooth_prob_{c}"] = float(smooth_probs[i, c])
            all_rows.append(out)

    out = pd.DataFrame(all_rows)
    if len(out) == 0:
        return out, {"accuracy": float("nan"), "macro_f1": float("nan"), "normal_false_alarm_rate": float("nan"), "prediction_change_rate": float("nan")}

    yt = out["true_label_id"].to_numpy().astype(int)
    yp = out["pred_label_id"].to_numpy().astype(int)
    change_rates = []
    for _vid, g in out.groupby("video_id"):
        p = g["pred_label_id"].to_numpy()
        if len(p) > 1:
            change_rates.append(float((p[1:] != p[:-1]).mean()))
    metrics = {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "prediction_change_rate": float(np.mean(change_rates)) if change_rates else 0.0,
    }
    metrics.update(class_recalls(yt, yp, config["CLASS_NAMES"]))
    return out, metrics


def evaluate_all_levels(
    model: nn.Module,
    loader: DataLoader,
    criterion: HierarchicalLoss,
    device: torch.device,
    use_amp: bool,
    X_raw: np.ndarray,
    meta: pd.DataFrame,
    feature_order: List[str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    probs, preds, targets, indices, loss, loss_parts = predict_loader(model, loader, criterion, device, use_amp)

    raw_X_eval = X_raw[indices]
    rule_probs, rule_preds = apply_sleepy_rule(probs, raw_X_eval, feature_order, config)

    seq_m = sequence_metrics(targets, preds, loss)
    seq_m.update(loss_parts)
    seq_m.update(class_recalls(targets, preds, config["CLASS_NAMES"]))

    seq_rule_m = sequence_metrics(targets, rule_preds, loss)
    seq_rule_m.update(class_recalls(targets, rule_preds, config["CLASS_NAMES"]))

    video_df, video_m = video_level_predictions(probs, targets, indices, meta, config)
    video_rule_df, video_rule_m = video_level_predictions(rule_probs, targets, indices, meta, config)

    stream_df, stream_m = streaming_predictions(probs, targets, indices, meta, config)
    stream_rule_df, stream_rule_m = streaming_predictions(rule_probs, targets, indices, meta, config)

    return {
        "probs": probs,
        "preds": preds,
        "rule_probs": rule_probs,
        "rule_preds": rule_preds,
        "targets": targets,
        "indices": indices,
        "sequence_metrics": seq_m,
        "sequence_metrics_rule": seq_rule_m,
        "video_predictions": video_df,
        "video_metrics": video_m,
        "video_predictions_rule": video_rule_df,
        "video_metrics_rule": video_rule_m,
        "streaming_predictions": stream_df,
        "streaming_metrics": stream_m,
        "streaming_predictions_rule": stream_rule_df,
        "streaming_metrics_rule": stream_rule_m,
    }


# ============================================================
# Save and plot
# ============================================================

def plot_history(history: pd.DataFrame, out_dir: Path) -> None:
    if len(history) == 0:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_acc"], label="train_acc")
    plt.plot(history["epoch"], history["val_sequence_acc"], label="val_sequence_acc")
    plt.plot(history["epoch"], history["val_streaming_acc"], label="val_streaming_acc")
    plt.plot(history["epoch"], history["val_streaming_acc_rule"], label="val_streaming_acc_rule")
    plt.plot(history["epoch"], history["val_video_acc"], label="val_video_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_macro_f1"], label="train_macro_f1")
    plt.plot(history["epoch"], history["val_sequence_macro_f1"], label="val_sequence_macro_f1")
    plt.plot(history["epoch"], history["val_streaming_macro_f1"], label="val_streaming_macro_f1")
    plt.plot(history["epoch"], history["val_streaming_macro_f1_rule"], label="val_streaming_macro_f1_rule")
    plt.plot(history["epoch"], history["val_video_macro_f1"], label="val_video_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Macro-F1 Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "macro_f1_curve.png", dpi=150)
    plt.close()


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], out_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], out_path: Path) -> None:
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, digits=4, zero_division=0)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)


def save_predictions(result: Dict[str, Any], meta: pd.DataFrame, config: Dict[str, Any], out_path: Path) -> None:
    indices = result["indices"]
    df = meta.iloc[indices].copy().reset_index(drop=True)
    df["true_label_id"] = result["targets"].astype(int)
    df["pred_label_id"] = result["preds"].astype(int)
    df["pred_label_id_rule"] = result["rule_preds"].astype(int)
    for c in range(config["NUM_CLASSES"]):
        df[f"prob_{c}"] = result["probs"][:, c]
        df[f"prob_rule_{c}"] = result["rule_probs"][:, c]
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, config: Dict[str, Any], metrics: Dict[str, Any], mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> None:
    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": to_jsonable(config),
        "metrics": to_jsonable(metrics),
    }
    if mean is not None and std is not None:
        ckpt["feature_mean"] = mean.astype(np.float32)
        ckpt["feature_std"] = std.astype(np.float32)
    torch.save(ckpt, path)


def load_best_model(path: Path, model: nn.Module, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


# ============================================================
# Runner
# ============================================================

def get_monitor_value(row: Dict[str, Any], monitor: str) -> float:
    if monitor not in row:
        raise KeyError(f"Monitor metric not found: {monitor}")
    return float(row[monitor])


def is_better(current: float, best: Optional[float], mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < best - min_delta
    if mode == "max":
        return current > best + min_delta
    raise ValueError(f"Unknown early stopping mode: {mode}")


def save_eval_outputs(prefix: str, result: Dict[str, Any], out_dir: Path, config: Dict[str, Any], meta: pd.DataFrame) -> None:
    save_json(out_dir / f"{prefix}_sequence_metrics.json", result["sequence_metrics"])
    save_json(out_dir / f"{prefix}_sequence_metrics_rule.json", result["sequence_metrics_rule"])
    save_json(out_dir / f"{prefix}_video_metrics.json", result["video_metrics"])
    save_json(out_dir / f"{prefix}_video_metrics_rule.json", result["video_metrics_rule"])
    save_json(out_dir / f"{prefix}_streaming_metrics.json", result["streaming_metrics"])
    save_json(out_dir / f"{prefix}_streaming_metrics_rule.json", result["streaming_metrics_rule"])

    save_confusion_matrix(result["targets"], result["preds"], config["CLASS_NAMES"], out_dir / f"{prefix}_sequence_confusion_matrix.png", f"{prefix} Sequence Confusion Matrix")
    save_confusion_matrix(result["targets"], result["rule_preds"], config["CLASS_NAMES"], out_dir / f"{prefix}_sequence_confusion_matrix_rule.png", f"{prefix} Sequence Confusion Matrix With Sleepy Rule")
    save_classification_report(result["targets"], result["preds"], config["CLASS_NAMES"], out_dir / f"{prefix}_sequence_classification_report.txt")
    save_classification_report(result["targets"], result["rule_preds"], config["CLASS_NAMES"], out_dir / f"{prefix}_sequence_classification_report_rule.txt")

    if len(result["video_predictions"]) > 0:
        vdf = result["video_predictions"]
        save_confusion_matrix(vdf["true_label_id"].to_numpy().astype(int), vdf["pred_label_id"].to_numpy().astype(int), config["CLASS_NAMES"], out_dir / f"{prefix}_video_confusion_matrix.png", f"{prefix} Video Confusion Matrix")
        vdf.to_csv(out_dir / f"{prefix}_video_predictions.csv", index=False, encoding="utf-8-sig")
    if len(result["video_predictions_rule"]) > 0:
        vdf = result["video_predictions_rule"]
        save_confusion_matrix(vdf["true_label_id"].to_numpy().astype(int), vdf["pred_label_id"].to_numpy().astype(int), config["CLASS_NAMES"], out_dir / f"{prefix}_video_confusion_matrix_rule.png", f"{prefix} Video Confusion Matrix With Sleepy Rule")
        vdf.to_csv(out_dir / f"{prefix}_video_predictions_rule.csv", index=False, encoding="utf-8-sig")

    if len(result["streaming_predictions"]) > 0:
        sdf = result["streaming_predictions"]
        save_confusion_matrix(sdf["true_label_id"].to_numpy().astype(int), sdf["pred_label_id"].to_numpy().astype(int), config["CLASS_NAMES"], out_dir / f"{prefix}_streaming_confusion_matrix.png", f"{prefix} Streaming Confusion Matrix")
        sdf.to_csv(out_dir / f"{prefix}_streaming_predictions.csv", index=False, encoding="utf-8-sig")
    if len(result["streaming_predictions_rule"]) > 0:
        sdf = result["streaming_predictions_rule"]
        save_confusion_matrix(sdf["true_label_id"].to_numpy().astype(int), sdf["pred_label_id"].to_numpy().astype(int), config["CLASS_NAMES"], out_dir / f"{prefix}_streaming_confusion_matrix_rule.png", f"{prefix} Streaming Confusion Matrix With Sleepy Rule")
        sdf.to_csv(out_dir / f"{prefix}_streaming_predictions_rule.csv", index=False, encoding="utf-8-sig")

    if config["SAVE_PREDICTIONS"]:
        save_predictions(result, meta, config, out_dir / f"{prefix}_sequence_predictions.csv")


def run_single_training(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    feature_order: List[str],
    config: Dict[str, Any],
    model_name: str,
    data_mode: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: Optional[np.ndarray],
    out_dir: Path,
    run_name: str,
    is_final_train: bool = False,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    device = get_device(config["DEVICE"])
    use_amp = bool(config["USE_AMP"] and device.type == "cuda")
    input_dim = int(X.shape[-1])

    model = HierarchicalDrowsinessModel(input_dim=input_dim, config=config, model_name=model_name).to(device)

    if config["STANDARDIZE_FEATURES"]:
        mean, std = compute_train_standardizer(X, train_indices, config["FEATURE_STD_EPS"])
    else:
        mean, std = None, None

    train_loader = make_loader(X, y, train_indices, config, mean, std, shuffle=True)
    val_loader = make_loader(X, y, val_indices, config, mean, std, shuffle=False)

    criterion = HierarchicalLoss(config=config, y_train=y[train_indices], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["LEARNING_RATE"]), weight_decay=float(config["WEIGHT_DECAY"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=float(config["LR_SCHEDULER_FACTOR"]), patience=int(config["LR_SCHEDULER_PATIENCE"])) if config["USE_LR_SCHEDULER"] else None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_json(out_dir / "train_config.json", config)
    save_json(out_dir / "feature_order_used.json", {"frame_feature_order": feature_order})

    split_df = pd.DataFrame({
        "index": np.concatenate([train_indices, val_indices] + ([] if test_indices is None else [test_indices])),
        "split": ["train"] * len(train_indices) + ["val"] * len(val_indices) + ([] if test_indices is None else ["test"] * len(test_indices)),
    })
    split_meta = split_df.merge(meta.reset_index().rename(columns={"index": "index"}), on="index", how="left")
    split_meta.to_csv(out_dir / "split_metadata.csv", index=False, encoding="utf-8-sig")

    print(f"\n[RUN] {run_name}")
    print(f"  model={model_name}, data_mode={data_mode}, final={is_final_train}")
    print(f"  train={len(train_indices)}, val={len(val_indices)}, test={0 if test_indices is None else len(test_indices)}")
    print(f"  device={device}, amp={use_amp}, input_dim={input_dim}")
    print(f"  monitor={config['EARLY_STOPPING_MONITOR']} ({config['EARLY_STOPPING_MODE']})")
    print(f"  sleepy_rule={config['USE_SLEEPY_RULE']} mode={config['SLEEPY_RULE_MODE']}")

    history_rows = []
    best_monitor = None
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, int(config["EPOCHS"]) + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, config["GRAD_CLIP_NORM"])

        val_result = evaluate_all_levels(model, val_loader, criterion, device, use_amp, X, meta, feature_order, config)
        val_seq = val_result["sequence_metrics"]
        val_seq_rule = val_result["sequence_metrics_rule"]
        val_vid = val_result["video_metrics"]
        val_vid_rule = val_result["video_metrics_rule"]
        val_str = val_result["streaming_metrics"]
        val_str_rule = val_result["streaming_metrics_rule"]

        if scheduler is not None:
            scheduler.step(val_seq["loss"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_m["loss"],
            "train_abnormal_loss": train_m["abnormal_loss"],
            "train_stage_loss": train_m["stage_loss"],
            "train_acc": train_m["acc"],
            "train_macro_f1": train_m["macro_f1"],

            "val_loss": val_seq["loss"],
            "val_abnormal_loss": val_seq["abnormal_loss"],
            "val_stage_loss": val_seq["stage_loss"],
            "val_sequence_acc": val_seq["accuracy"],
            "val_sequence_macro_f1": val_seq["macro_f1"],
            "val_sequence_acc_rule": val_seq_rule["accuracy"],
            "val_sequence_macro_f1_rule": val_seq_rule["macro_f1"],

            "val_video_acc": val_vid["accuracy"],
            "val_video_macro_f1": val_vid["macro_f1"],
            "val_video_acc_rule": val_vid_rule["accuracy"],
            "val_video_macro_f1_rule": val_vid_rule["macro_f1"],

            "val_streaming_acc": val_str["accuracy"],
            "val_streaming_macro_f1": val_str["macro_f1"],
            "val_streaming_normal_false_alarm_rate": val_str["normal_false_alarm_rate"],
            "val_streaming_sleepy_recall": val_str.get("sleepy_recall", float("nan")),
            "val_streaming_acc_rule": val_str_rule["accuracy"],
            "val_streaming_macro_f1_rule": val_str_rule["macro_f1"],
            "val_streaming_normal_false_alarm_rate_rule": val_str_rule["normal_false_alarm_rate"],
            "val_streaming_sleepy_recall_rule": val_str_rule.get("sleepy_recall", float("nan")),
            "time_sec": time.time() - t0,
        }
        history_rows.append(row)

        monitor_value = get_monitor_value(row, config["EARLY_STOPPING_MONITOR"])
        better = is_better(monitor_value, best_monitor, config["EARLY_STOPPING_MODE"], float(config["EARLY_STOPPING_MIN_DELTA"]))
        if better:
            best_monitor = monitor_value
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(out_dir / "best_model.pt", model, optimizer, epoch, config, row, mean, std)
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} f1 {row['train_macro_f1']:.4f} | "
            f"val stream f1 {row['val_streaming_macro_f1']:.4f} acc {row['val_streaming_acc']:.4f} sleepyR {row['val_streaming_sleepy_recall']:.4f} | "
            f"rule f1 {row['val_streaming_macro_f1_rule']:.4f} acc {row['val_streaming_acc_rule']:.4f} sleepyR {row['val_streaming_sleepy_recall_rule']:.4f} | "
            f"monitor {monitor_value:.4f} | time {row['time_sec']:.1f}s"
        )

        if epochs_no_improve >= int(config["EARLY_STOPPING_PATIENCE"]):
            print(f"[EARLY STOP] best_epoch={best_epoch}, best_monitor={best_monitor:.6f}")
            break

    save_checkpoint(out_dir / "last_model.pt", model, optimizer, history_rows[-1]["epoch"] if history_rows else 0, config, history_rows[-1] if history_rows else {}, mean, std)
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
    plot_history(history_df, out_dir)

    best_model = HierarchicalDrowsinessModel(input_dim=input_dim, config=config, model_name=model_name).to(device)
    ckpt = load_best_model(out_dir / "best_model.pt", best_model, device)
    ckpt_mean = ckpt.get("feature_mean", mean)
    ckpt_std = ckpt.get("feature_std", std)
    if ckpt_mean is not None:
        ckpt_mean = np.asarray(ckpt_mean, dtype=np.float32)
    if ckpt_std is not None:
        ckpt_std = np.asarray(ckpt_std, dtype=np.float32)

    val_eval_loader = make_loader(X, y, val_indices, config, ckpt_mean, ckpt_std, shuffle=False)
    final_val_result = evaluate_all_levels(best_model, val_eval_loader, criterion, device, use_amp, X, meta, feature_order, config)
    save_eval_outputs("val", final_val_result, out_dir, config, meta)

    test_summary = {}
    if test_indices is not None and len(test_indices) > 0:
        test_loader = make_loader(X, y, test_indices, config, ckpt_mean, ckpt_std, shuffle=False)
        test_result = evaluate_all_levels(best_model, test_loader, criterion, device, use_amp, X, meta, feature_order, config)
        save_eval_outputs("test", test_result, out_dir, config, meta)
        test_summary = {
            "test_sequence_acc": test_result["sequence_metrics"]["accuracy"],
            "test_sequence_macro_f1": test_result["sequence_metrics"]["macro_f1"],
            "test_sequence_acc_rule": test_result["sequence_metrics_rule"]["accuracy"],
            "test_sequence_macro_f1_rule": test_result["sequence_metrics_rule"]["macro_f1"],
            "test_video_acc": test_result["video_metrics"]["accuracy"],
            "test_video_macro_f1": test_result["video_metrics"]["macro_f1"],
            "test_video_acc_rule": test_result["video_metrics_rule"]["accuracy"],
            "test_video_macro_f1_rule": test_result["video_metrics_rule"]["macro_f1"],
            "test_streaming_acc": test_result["streaming_metrics"]["accuracy"],
            "test_streaming_macro_f1": test_result["streaming_metrics"]["macro_f1"],
            "test_streaming_sleepy_recall": test_result["streaming_metrics"].get("sleepy_recall", float("nan")),
            "test_streaming_normal_false_alarm_rate": test_result["streaming_metrics"].get("normal_false_alarm_rate", float("nan")),
            "test_streaming_acc_rule": test_result["streaming_metrics_rule"]["accuracy"],
            "test_streaming_macro_f1_rule": test_result["streaming_metrics_rule"]["macro_f1"],
            "test_streaming_sleepy_recall_rule": test_result["streaming_metrics_rule"].get("sleepy_recall", float("nan")),
            "test_streaming_normal_false_alarm_rate_rule": test_result["streaming_metrics_rule"].get("normal_false_alarm_rate", float("nan")),
        }

    summary = {
        "run_name": run_name,
        "model_name": model_name,
        "data_mode": data_mode,
        "is_final_train": is_final_train,
        "best_epoch": int(ckpt["epoch"]),
        "best_monitor": best_monitor,
        "num_train": int(len(train_indices)),
        "num_val": int(len(val_indices)),
        "num_test": int(0 if test_indices is None else len(test_indices)),
    }
    summary.update(test_summary)
    save_json(out_dir / "run_summary.json", summary)
    return summary


def run_cv_for_model_data(X, y, meta, data_root, feature_order, config, model_name, data_mode, base_out: Path) -> pd.DataFrame:
    fold_col = config["FOLD_COLUMN"]
    test_folds = sort_folds(meta[fold_col].astype(str).unique().tolist()) if config["TEST_FOLDS"] is None else [str(x) for x in config["TEST_FOLDS"]]
    summaries = []
    for fold in test_folds:
        fold_out = base_out / f"fold_{fold}"
        ensure_dir(fold_out)
        train_idx, val_idx, test_idx = make_fold_split(meta, y, config, fold, data_mode)
        copy_if_exists(data_root / config["FEATURE_ORDER_FILE"], fold_out)
        copy_if_exists(data_root / config["PREPROCESS_CONFIG_FILE"], fold_out)
        summary = run_single_training(
            X=X,
            y=y,
            meta=meta,
            feature_order=feature_order,
            config=config,
            model_name=model_name,
            data_mode=data_mode,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            out_dir=fold_out,
            run_name=f"{model_name}_{data_mode}_{fold}",
            is_final_train=False,
        )
        summary["fold"] = fold
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(base_out / "cv_summary_by_fold.csv", index=False, encoding="utf-8-sig")
    metric_cols = [c for c in summary_df.columns if c.startswith("test_")]
    rows = []
    for col in metric_cols:
        values = pd.to_numeric(summary_df[col], errors="coerce")
        rows.append({"metric": col, "mean": float(values.mean()), "std": float(values.std(ddof=0)), "min": float(values.min()), "max": float(values.max())})
    cv_stats = pd.DataFrame(rows)
    cv_stats.to_csv(base_out / "cv_summary_mean_std.csv", index=False, encoding="utf-8-sig")
    print("\n[CV SUMMARY]")
    print(cv_stats)
    return summary_df


def run_final_train(X, y, meta, data_root, feature_order, config, model_name, data_mode, base_out: Path) -> Dict[str, Any]:
    final_out = base_out / "final_train"
    ensure_dir(final_out)
    train_idx, val_idx = make_final_train_val_split(meta, y, config, data_mode)
    copy_if_exists(data_root / config["FEATURE_ORDER_FILE"], final_out)
    copy_if_exists(data_root / config["PREPROCESS_CONFIG_FILE"], final_out)
    return run_single_training(
        X=X,
        y=y,
        meta=meta,
        feature_order=feature_order,
        config=config,
        model_name=model_name,
        data_mode=data_mode,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=None,
        out_dir=final_out,
        run_name=f"{model_name}_{data_mode}_final",
        is_final_train=True,
    )


def main():
    config = apply_default_config(CONFIG)
    set_seed(int(config["SEED"]))
    device = get_device(config["DEVICE"])
    print("[ENV]")
    print("  torch:", torch.__version__)
    print("  cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  cuda device:", torch.cuda.get_device_name(0))
    print("  selected device:", device)

    X, y, meta, data_root, feature_order = load_all_data(config)

    output_root = Path(config["OUTPUT_ROOT"]) / config["EXPERIMENT_TAG"]
    ensure_dir(output_root)
    save_json(output_root / "global_train_config.json", config)
    save_json(output_root / "feature_order_used.json", {"frame_feature_order": feature_order})
    copy_if_exists(data_root / config["FEATURE_ORDER_FILE"], output_root)
    copy_if_exists(data_root / config["PREPROCESS_CONFIG_FILE"], output_root)

    all_run_summaries = []
    for data_mode in config["DATA_MODES_TO_RUN"]:
        for model_name in config["MODELS_TO_RUN"]:
            combo_name = f"{model_name}_{data_mode}_hierarchical_rule"
            combo_out = output_root / combo_name
            ensure_dir(combo_out)
            print("\n" + "=" * 80)
            print(f"[COMBO] model={model_name}, data_mode={data_mode}, task=hierarchical+sleepy_rule")
            print("=" * 80)
            if config["RUN_FOLD_CV"]:
                cv_summary = run_cv_for_model_data(X, y, meta, data_root, feature_order, config, model_name, data_mode, combo_out)
                cv_summary["model_name"] = model_name
                cv_summary["data_mode"] = data_mode
                all_run_summaries.append(cv_summary)
            if config["RUN_FINAL_TRAIN_AFTER_CV"]:
                final_summary = run_final_train(X, y, meta, data_root, feature_order, config, model_name, data_mode, combo_out)
                pd.DataFrame([final_summary]).to_csv(combo_out / "final_train_summary.csv", index=False, encoding="utf-8-sig")

    if all_run_summaries:
        total = pd.concat(all_run_summaries, ignore_index=True)
        total.to_csv(output_root / "all_cv_results.csv", index=False, encoding="utf-8-sig")

    print("\n[DONE]")
    print("Output:", output_root)


if __name__ == "__main__":
    main()
