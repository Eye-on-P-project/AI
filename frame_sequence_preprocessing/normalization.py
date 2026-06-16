from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG, RAW_NORM_COLS, Z_COLS
from .utils import read_csv_safe, safe_name
def select_calibration_rows(df: pd.DataFrame) -> pd.DataFrame:
    valid_mask = df["face_detected"].astype(int).eq(1)
    for col in RAW_NORM_COLS:
        valid_mask &= np.isfinite(df[col].astype(float))
    valid_df = df.loc[valid_mask].copy()
    if valid_df.empty:
        return valid_df
    k = max(1, int(len(valid_df) * float(CONFIG["CALIBRATION_RATIO"])))
    if str(CONFIG["CALIBRATION_PART"]).lower() == "last":
        return valid_df.tail(k)
    return valid_df.head(k)


def compute_subject_stats(manifest: pd.DataFrame, dirs: Dict[str, Path]) -> Tuple[Dict[str, dict], pd.DataFrame]:
    stats = {}
    rows = []
    for subject_id, sub_df in manifest.groupby("subject_id"):
        alert_rows = sub_df[sub_df["score_label"].astype(str) == str(CONFIG["CALIBRATION_SCORE_LABEL"])]
        if alert_rows.empty:
            rows.append({"subject_id": subject_id, "status": "no_normal_video"})
            continue

        alert_row = alert_rows.iloc[0]
        alert_vid = str(alert_row["video_id"])
        raw_path = dirs["frame_raw"] / f"{safe_name(alert_vid)}_frame_features_raw.csv"
        raw_df = read_csv_safe(raw_path)
        if raw_df.empty:
            rows.append({"subject_id": subject_id, "calib_video_id": alert_vid, "status": "empty_calib_video"})
            continue

        calib_df = select_calibration_rows(raw_df)
        if len(calib_df) < int(CONFIG["MIN_CALIBRATION_VALID_FRAMES"]):
            rows.append({
                "subject_id": subject_id,
                "calib_video_id": alert_vid,
                "status": "too_few_calibration_frames",
                "num_calibration_frames": len(calib_df),
            })
            continue

        values = calib_df[RAW_NORM_COLS].astype(float).values
        mean = np.nanmean(values, axis=0).astype(np.float32)
        std = np.nanstd(values, axis=0).astype(np.float32)
        std = np.where((~np.isfinite(std)) | (std < 1e-8), 1.0, std).astype(np.float32)
        stats[str(subject_id)] = {"mean": mean, "std": std, "calib_video_id": alert_vid}

        row = {
            "subject_id": subject_id,
            "calib_video_id": alert_vid,
            "status": "ok",
            "num_calibration_frames": len(calib_df),
        }
        for col, m, s in zip(RAW_NORM_COLS, mean, std):
            row[f"mean_{col}"] = float(m)
            row[f"std_{col}"] = float(s)
        rows.append(row)
    return stats, pd.DataFrame(rows)


def normalize_frame_df(raw_df: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> pd.DataFrame:
    df = raw_df.copy()
    for i, raw_col in enumerate(RAW_NORM_COLS):
        z_col = Z_COLS[i]
        vals = df[raw_col].astype(float).values
        z = (vals - mean[i]) / (std[i] if std[i] > 1e-8 else 1.0)
        df[z_col] = z.astype(np.float32)

    # short missing interpolation. Remaining missing values become normal baseline 0.
    interp_limit = int(CONFIG["INTERPOLATE_LIMIT_FRAMES"])
    for col in Z_COLS:
        s = pd.Series(df[col].astype(float))
        s = s.interpolate(method="linear", limit=interp_limit, limit_direction="both")
        df[col] = s.fillna(0.0).astype(np.float32)

    # p_closed is probability. Fill short missing then remaining with 0.
    p = pd.Series(df["p_closed"].astype(float))
    p = p.interpolate(method="linear", limit=interp_limit, limit_direction="both")
    df["p_closed"] = p.fillna(0.0).clip(0.0, 1.0).astype(np.float32)

    win = max(1, int(CONFIG["P_CLOSED_SMOOTH_WINDOW"]))
    df["p_closed_smooth"] = (
        df["p_closed"].astype(float)
        .rolling(window=win, min_periods=1, center=False)
        .mean()
        .clip(0.0, 1.0)
        .astype(np.float32)
    )
    return df
