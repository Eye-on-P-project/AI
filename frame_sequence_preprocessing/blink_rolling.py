from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .config import CONFIG
def compute_closed_state_and_duration(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    p = df["p_closed_smooth"].astype(float).values
    times = df["timestamp"].astype(float).values
    start_thr = float(CONFIG["P_CLOSED_START_THRESHOLD"])
    end_thr = float(CONFIG["P_CLOSED_END_THRESHOLD"])

    closed = np.zeros(len(df), dtype=np.int32)
    duration = np.zeros(len(df), dtype=np.float32)
    in_closed = False
    start_time = None

    for i, prob in enumerate(p):
        if not in_closed:
            if prob >= start_thr:
                in_closed = True
                start_time = times[i]
        else:
            if prob <= end_thr:
                in_closed = False
                start_time = None

        if in_closed:
            closed[i] = 1
            duration[i] = max(0.0, float(times[i] - start_time)) if start_time is not None else 0.0
        else:
            closed[i] = 0
            duration[i] = 0.0

    return closed, duration


def detect_blinks_from_pclosed(df: pd.DataFrame) -> pd.DataFrame:
    p = df["p_closed_smooth"].astype(float).values
    times = df["timestamp"].astype(float).values
    frame_idx = df["frame_idx"].astype(int).values
    ear_z = df["ear_z"].astype(float).values

    start_thr = float(CONFIG["P_CLOSED_START_THRESHOLD"])
    end_thr = float(CONFIG["P_CLOSED_END_THRESHOLD"])
    min_dur = float(CONFIG["MIN_BLINK_DURATION_SEC"])
    max_dur = float(CONFIG["MAX_BLINK_DURATION_SEC"])
    min_amp = float(CONFIG["MIN_BLINK_AMPLITUDE_Z"])
    min_gap = float(CONFIG["MIN_BLINK_GAP_SEC"])

    events = []
    in_closed = False
    s = None
    last_end_time = -1e9
    blink_count = 0

    for i, prob in enumerate(p):
        if not in_closed and prob >= start_thr:
            if times[i] - last_end_time >= min_gap:
                in_closed = True
                s = i
        elif in_closed and prob <= end_thr:
            e = i
            if s is None or e <= s:
                in_closed = False
                s = None
                continue

            duration_sec = float(times[e] - times[s])
            if min_dur <= duration_sec <= max_dur:
                seg = ear_z[s:e + 1]
                if len(seg) > 0 and np.isfinite(seg).any():
                    m = s + int(np.nanargmin(seg))
                    ear_start = float(ear_z[s])
                    ear_min = float(ear_z[m])
                    ear_end = float(ear_z[e])
                    amplitude = float(((ear_start + ear_end) / 2.0) - ear_min)
                    opening_time = max(float(times[e] - times[m]), 1e-6)
                    opening_velocity = float((ear_end - ear_min) / opening_time)
                    if np.isfinite(amplitude) and amplitude >= min_amp:
                        blink_count += 1
                        events.append({
                            "blink_id": len(events),
                            "video_id": str(df["video_id"].iloc[0]),
                            "start_idx": int(s),
                            "end_idx": int(e),
                            "min_idx": int(m),
                            "start_frame": int(frame_idx[s]),
                            "end_frame": int(frame_idx[e]),
                            "min_frame": int(frame_idx[m]),
                            "start_time": float(times[s]),
                            "end_time": float(times[e]),
                            "min_time": float(times[m]),
                            "duration_sec": duration_sec,
                            "duration": int(frame_idx[e] - frame_idx[s] + 1),
                            "amplitude": amplitude,
                            "opening_velocity": opening_velocity,
                            "frequency": float(blink_count / max(times[e], 1e-6)),
                            "p_closed_max": float(np.nanmax(p[s:e + 1])),
                            "ear_z_start": ear_start,
                            "ear_z_min": ear_min,
                            "ear_z_end": ear_end,
                        })
                        last_end_time = float(times[e])

            in_closed = False
            s = None

    return pd.DataFrame(events)


def compute_rolling_features(df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    times = out["timestamp"].astype(float).values
    closed_state, closed_duration = compute_closed_state_and_duration(out)
    out["closed_state"] = closed_state
    out["current_closed_duration"] = closed_duration

    window_sec = float(CONFIG["ROLLING_WINDOW_SECONDS"])
    cap_tslb = float(CONFIG["TIME_SINCE_LAST_BLINK_CAP_SEC"])

    # PERCLOS with two-pointer rolling window.
    perclos = np.zeros(len(out), dtype=np.float32)
    left = 0
    closed_sum = 0
    for i, t in enumerate(times):
        closed_sum += int(closed_state[i])
        while left <= i and times[left] < t - window_sec:
            closed_sum -= int(closed_state[left])
            left += 1
        denom = max(i - left + 1, 1)
        perclos[i] = closed_sum / denom
    out["perclos_10s"] = perclos

    # Blink rolling stats.
    if events_df.empty:
        out["avg_blink_duration_10s"] = 0.0
        out["avg_blink_amplitude_10s"] = 0.0
        out["avg_opening_velocity_10s"] = 0.0
        out["blink_frequency_10s"] = 0.0
        out["time_since_last_blink"] = cap_tslb
        return out

    ev_end = events_df["end_time"].astype(float).values
    ev_duration = events_df["duration_sec"].astype(float).values
    ev_amp = events_df["amplitude"].astype(float).values
    ev_vel = events_df["opening_velocity"].astype(float).values

    avg_dur = np.zeros(len(out), dtype=np.float32)
    avg_amp = np.zeros(len(out), dtype=np.float32)
    avg_vel = np.zeros(len(out), dtype=np.float32)
    freq = np.zeros(len(out), dtype=np.float32)
    tslb = np.full(len(out), cap_tslb, dtype=np.float32)

    last_blink_end = None
    for i, t in enumerate(times):
        mask = (ev_end <= t) & (ev_end >= t - window_sec)
        if mask.any():
            avg_dur[i] = float(np.nanmean(ev_duration[mask]))
            avg_amp[i] = float(np.nanmean(ev_amp[mask]))
            avg_vel[i] = float(np.nanmean(ev_vel[mask]))
            freq[i] = float(mask.sum() / window_sec)
        ended = ev_end[ev_end <= t]
        if len(ended) > 0:
            last_blink_end = float(ended[-1])
            tslb[i] = min(float(t - last_blink_end), cap_tslb)

    out["avg_blink_duration_10s"] = avg_dur
    out["avg_blink_amplitude_10s"] = avg_amp
    out["avg_opening_velocity_10s"] = avg_vel
    out["blink_frequency_10s"] = freq
    out["time_since_last_blink"] = tslb
    return out
