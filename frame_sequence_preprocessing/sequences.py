from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG, FRAME_FEATURE_COLS
from .utils import safe_name
def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _safe_mean(series: pd.Series, default: float = 0.0) -> float:
    value = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
    return default if pd.isna(value) else float(value)


def _safe_max(series: pd.Series, default: float = 0.0) -> float:
    value = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).max()
    return default if pd.isna(value) else float(value)


def _safe_abs_max(series: pd.Series, default: float = 0.0) -> float:
    value = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).abs().max()
    return default if pd.isna(value) else float(value)


def calc_confidence_scores(seq_df: pd.DataFrame) -> Dict[str, float]:
    """Compute class-specific confidence scores.

    sleepy는 하품을 사용하지 않고 눈 감김과 고개 숙임만 사용한다.
    drowsy는 하품 고개 변화 약한 눈 감김을 함께 사용한다.
    """
    caps = CONFIG["SCORE_CAPS"]

    mean_perclos_raw = _safe_mean(seq_df["perclos_10s"])
    max_closed_duration_raw = _safe_max(seq_df["current_closed_duration"])
    mean_p_closed_raw = _safe_mean(seq_df["p_closed"])
    max_yawn_z_raw = max(_safe_max(seq_df["mar_z"]), 0.0)
    max_abs_pitch_z_raw = _safe_abs_max(seq_df["pitch_z"])
    mean_blink_duration_raw = _safe_mean(seq_df["avg_blink_duration_10s"])

    direction = str(CONFIG.get("HEAD_DOWN_DIRECTION", "abs")).lower()
    pitch = pd.to_numeric(seq_df["pitch_z"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if direction == "positive":
        head_down_raw = max(float(pitch.max()), 0.0)
    elif direction == "negative":
        head_down_raw = max(float((-pitch).max()), 0.0)
    else:
        head_down_raw = float(pitch.abs().max())

    perclos = _clip01(mean_perclos_raw / float(caps["perclos"]))
    closed_duration = _clip01(max_closed_duration_raw / float(caps["closed_duration"]))
    p_closed = _clip01(mean_p_closed_raw / float(caps["p_closed"]))
    yawn = _clip01(max_yawn_z_raw / float(caps["yawn_z"]))
    head = _clip01(max_abs_pitch_z_raw / float(caps["head_abs_z"]))
    head_down = _clip01(head_down_raw / float(caps["head_down_z"]))
    blink_duration = _clip01(mean_blink_duration_raw / float(caps["blink_duration_sec"]))

    eye_w = CONFIG["EYE_SLEEP_SCORE_WEIGHTS"]
    drowsy_w = CONFIG["DROWSY_SCORE_WEIGHTS"]

    eye_sleep_score = _clip01(
        eye_w["perclos"] * perclos +
        eye_w["closed_duration"] * closed_duration +
        eye_w["p_closed"] * p_closed +
        eye_w["blink_duration"] * blink_duration
    )

    drowsy_score = _clip01(
        drowsy_w["perclos"] * perclos +
        drowsy_w["p_closed"] * p_closed +
        drowsy_w["yawn"] * yawn +
        drowsy_w["head"] * head +
        drowsy_w["blink_duration"] * blink_duration
    )

    # sleepy_score는 눈 감김 score와 고개 숙임 score 중 강한 쪽을 사용한다.
    # 즉 눈을 오래 감거나 고개가 아래로 푹 숙여져 있으면 sleepy 후보가 될 수 있다.
    sleepy_score = max(eye_sleep_score, head_down)

    return {
        "confidence_score": float(sleepy_score if sleepy_score >= drowsy_score else drowsy_score),
        "eye_sleep_score": float(eye_sleep_score),
        "drowsy_score": float(drowsy_score),
        "sleepy_score": float(sleepy_score),
        "head_down_score": float(head_down),
        "mean_perclos_raw": float(mean_perclos_raw),
        "max_closed_duration_raw": float(max_closed_duration_raw),
        "mean_p_closed_raw": float(mean_p_closed_raw),
        "max_yawn_z_raw": float(max_yawn_z_raw),
        "max_abs_pitch_z_raw": float(max_abs_pitch_z_raw),
        "max_head_down_z_raw": float(head_down_raw),
        "mean_blink_duration_raw": float(mean_blink_duration_raw),
    }


def high_confidence_decision(label: str, scores: Dict[str, float]) -> Tuple[bool, str]:
    eye = float(scores["eye_sleep_score"])
    drowsy = float(scores["drowsy_score"])
    head_down = float(scores["head_down_score"])
    closed_dur = float(scores["max_closed_duration_raw"])
    perclos = float(scores["mean_perclos_raw"])
    p_closed = float(scores["mean_p_closed_raw"])

    if label == "normal":
        keep = (
            eye <= float(CONFIG["HIGH_CONF_NORMAL_MAX_EYE_SLEEP_SCORE"]) and
            drowsy <= float(CONFIG["HIGH_CONF_NORMAL_MAX_DROWSY_SCORE"]) and
            head_down <= float(CONFIG["HIGH_CONF_NORMAL_MAX_HEAD_DOWN_SCORE"])
        )
        return keep, "normal_low_eye_drowsy_head_scores" if keep else "normal_confidence_rejected"

    if label == "drowsy":
        keep = (
            float(CONFIG["HIGH_CONF_DROWSY_MIN_SCORE"]) <= drowsy <= float(CONFIG["HIGH_CONF_DROWSY_MAX_SCORE"]) and
            eye < float(CONFIG["HIGH_CONF_DROWSY_MAX_EYE_SLEEP_SCORE"]) and
            closed_dur < float(CONFIG["HIGH_CONF_DROWSY_MAX_CLOSED_DURATION"]) and
            head_down < float(CONFIG["HIGH_CONF_DROWSY_MAX_HEAD_DOWN_SCORE"])
        )
        return keep, "drowsy_mid_drowsy_score_not_sleepy" if keep else "drowsy_confidence_rejected"

    if label == "sleepy":
        long_eye_closed = closed_dur >= float(CONFIG["HIGH_CONF_SLEEPY_MIN_CLOSED_DURATION"])
        high_perclos_closed = (
            perclos >= float(CONFIG["HIGH_CONF_SLEEPY_MIN_PERCLOS"]) and
            p_closed >= float(CONFIG["HIGH_CONF_SLEEPY_MIN_P_CLOSED"])
        )
        high_eye_score = eye >= float(CONFIG["HIGH_CONF_SLEEPY_MIN_EYE_SLEEP_SCORE"])
        head_down_sleepy = head_down >= float(CONFIG["HIGH_CONF_SLEEPY_MIN_HEAD_DOWN_SCORE"])
        keep = long_eye_closed or high_perclos_closed or high_eye_score or head_down_sleepy
        if keep:
            reasons = []
            if long_eye_closed:
                reasons.append("long_eye_closed")
            if high_perclos_closed:
                reasons.append("high_perclos_and_p_closed")
            if high_eye_score:
                reasons.append("high_eye_sleep_score")
            if head_down_sleepy:
                reasons.append("head_down")
            return True, "sleepy_" + "+".join(reasons)
        return False, "sleepy_confidence_rejected"

    return False, "unknown_label"


def build_sequences_for_video(
    row: pd.Series,
    rolling_df: pd.DataFrame,
    dirs: Dict[str, Path],
) -> Tuple[List[dict], List[dict]]:
    video_id = str(row["video_id"])
    label = str(row["label"])
    label_id = int(row["label_id"])

    target_fps = float(CONFIG["TARGET_FPS"])
    seq_len = int(round(target_fps * float(CONFIG["SEQUENCE_SECONDS"])))
    stride = max(1, int(round(target_fps * float(CONFIG["SEQUENCE_STRIDE_SECONDS"]))))
    save_all_inherited = bool(CONFIG.get("SAVE_ALL_INHERITED_DATASET", False))

    # all_rows는 전체 후보 sequence 수와 score 분포 확인용이다.
    # 이번 high-confidence 전용 버전에서는 SAVE_ALL_INHERITED_DATASET=False이면
    # 전체 sequence npy와 전체 X_all/y_all은 저장하지 않는다.
    all_rows = []
    high_rows = []

    if len(rolling_df) < seq_len:
        return all_rows, high_rows

    arr = rolling_df[FRAME_FEATURE_COLS].astype(float).values.astype(np.float32)
    n = len(arr)

    for seq_idx, start in enumerate(range(0, n - seq_len + 1, stride)):
        end = start + seq_len
        seq_df = rolling_df.iloc[start:end].copy()
        face_rate = float(seq_df["face_detected"].astype(float).mean())
        if face_rate < float(CONFIG["MIN_FACE_RATE_PER_SEQUENCE"]):
            continue

        seq = arr[start:end]
        seq_id = f"{safe_name(video_id)}_seq{seq_idx:05d}"

        scores = calc_confidence_scores(seq_df)
        is_high, reason = high_confidence_decision(label, scores)

        all_path = dirs["seq_all"] / f"{seq_id}.npy"
        meta = {
            "sequence_id": seq_id,
            "video_id": video_id,
            "fold": row["fold"],
            "part": row["part"],
            "subject_id": row["subject_id"],
            "score_label": row["score_label"],
            "original_label": row.get("original_label", ""),
            "label": label,
            "label_id": label_id,
            "label_source": "video_label_inherited_candidate",
            "start_row": int(start),
            "end_row": int(end - 1),
            "start_frame": int(seq_df["frame_idx"].iloc[0]),
            "end_frame": int(seq_df["frame_idx"].iloc[-1]),
            "start_time": float(seq_df["timestamp"].iloc[0]),
            "end_time": float(seq_df["timestamp"].iloc[-1]),
            "num_frames": int(seq_len),
            "feature_dim": int(len(FRAME_FEATURE_COLS)),
            "feature_order": ",".join(FRAME_FEATURE_COLS),
            "feature_path": "",
            "face_detection_rate": face_rate,
            "confidence_score": float(scores["confidence_score"]),
            "eye_sleep_score": float(scores["eye_sleep_score"]),
            "drowsy_score": float(scores["drowsy_score"]),
            "sleepy_score": float(scores["sleepy_score"]),
            "head_down_score": float(scores["head_down_score"]),
            "is_high_confidence": int(is_high),
            "filter_reason": reason,
            "mean_p_closed": float(scores["mean_p_closed_raw"]),
            "max_closed_duration": float(scores["max_closed_duration_raw"]),
            "mean_perclos": float(scores["mean_perclos_raw"]),
            "max_mar_z": float(scores["max_yawn_z_raw"]),
            "max_abs_pitch_z": float(scores["max_abs_pitch_z_raw"]),
            "max_head_down_z": float(scores["max_head_down_z_raw"]),
            "mean_blink_duration": float(scores["mean_blink_duration_raw"]),
            "mean_blink_frequency": float(seq_df["blink_frequency_10s"].mean()),
        }

        if save_all_inherited:
            if CONFIG["SAVE_PER_SEQUENCE_NPY"]:
                np.save(all_path, seq)
            meta["feature_path"] = str(all_path.resolve()) if CONFIG["SAVE_PER_SEQUENCE_NPY"] else ""
            meta["label_source"] = "video_label_inherited"

        all_rows.append(meta)

        if is_high and bool(CONFIG["MAKE_HIGH_CONFIDENCE_DATASET"]):
            high_path = dirs["seq_high"] / f"{seq_id}.npy"
            if CONFIG["SAVE_PER_SEQUENCE_NPY"]:
                np.save(high_path, seq)
            high_meta = dict(meta)
            high_meta["feature_path"] = str(high_path.resolve()) if CONFIG["SAVE_PER_SEQUENCE_NPY"] else ""
            high_meta["label_source"] = "video_label_inherited_high_confidence"
            high_rows.append(high_meta)

    return all_rows, high_rows


def save_stacked_arrays(metadata: pd.DataFrame, out_x: Path, out_y: Path) -> None:
    seq_len = int(round(float(CONFIG["TARGET_FPS"]) * float(CONFIG["SEQUENCE_SECONDS"])))
    feature_dim = int(len(FRAME_FEATURE_COLS))
    if metadata.empty:
        X = np.empty((0, seq_len, feature_dim), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        np.save(out_x, X)
        np.save(out_y, y)
        print(f"  saved empty {out_x.name}: {X.shape}")
        print(f"  saved empty {out_y.name}: {y.shape}")
        return
    paths = metadata["feature_path"].tolist()
    if not paths or any(not p for p in paths):
        X = np.empty((0, seq_len, feature_dim), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        np.save(out_x, X)
        np.save(out_y, y)
        print(f"  saved empty {out_x.name}: {X.shape} because feature_path is missing")
        print(f"  saved empty {out_y.name}: {y.shape}")
        return
    X = np.stack([np.load(p) for p in paths]).astype(np.float32)
    y = metadata["label_id"].values.astype(np.int64)
    np.save(out_x, X)
    np.save(out_y, y)
    print(f"  saved {out_x.name}: {X.shape}")
    print(f"  saved {out_y.name}: {y.shape}")
