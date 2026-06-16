from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

import cv2
import pandas as pd

try:
    import torch
except ImportError as e:
    raise ImportError("torch가 없습니다. pip install torch 로 설치하세요.") from e

from .config import CONFIG
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def get_dirs(out_root: Path) -> Dict[str, Path]:
    dirs = {
        "root": out_root,
        "frame_raw": out_root / "frame_features_raw",
        "frame_norm": out_root / "frame_features_norm",
        "blink_events": out_root / "blink_events_eye_model",
        "rolling": out_root / "rolling_features",
        "seq_all": out_root / "sequences_all" / "features",
        "seq_high": out_root / "sequences_high_confidence" / "features",
        "logs": out_root / "logs",
    }
    for p in dirs.values():
        ensure_dir(p)
    return dirs


def read_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["video_id", "fold", "part", "subject_id", "score_label", "video_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"manifest에 필요한 컬럼이 없습니다: {missing}")

    df = df.copy()
    for c in required:
        df[c] = df[c].astype(str)

    df["label"] = df["score_label"].map(CONFIG["SCORE_TO_LABEL"])
    df = df[df["label"].notna()].reset_index(drop=True)
    df["label_id"] = df["label"].map(CONFIG["LABEL_TO_ID"]).astype(int)

    if "original_label" not in df.columns:
        score_to_original = {"0": "Alert", "5": "Low Vigilant", "10": "Drowsy"}
        df["original_label"] = df["score_label"].map(score_to_original)

    if df["video_id"].duplicated().any():
        counts = {}
        new_ids = []
        for vid in df["video_id"]:
            counts[vid] = counts.get(vid, 0) + 1
            new_ids.append(vid if counts[vid] == 1 else f"{vid}_dup{counts[vid]}")
        df["video_id_original"] = df["video_id"]
        df["video_id"] = new_ids

    return df


def resize_for_detection(frame: np.ndarray, max_width: Optional[int]) -> np.ndarray:
    if max_width is None:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    new_h = int(round(h * scale))
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)


def choose_device() -> str:
    requested = str(CONFIG["EYE_MODEL_DEVICE"]).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA를 요청했지만 사용 불가합니다. CPU로 전환합니다.")
        return "cpu"
    return requested


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
