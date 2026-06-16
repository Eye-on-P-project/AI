from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from .blink_rolling import compute_rolling_features, detect_blinks_from_pclosed
from .config import CONFIG, FRAME_FEATURE_COLS
from .normalization import compute_subject_stats, normalize_frame_df
from .sequences import build_sequences_for_video, save_stacked_arrays
from .utils import read_csv_safe, safe_name
def build_frame_sequence_datasets(manifest: pd.DataFrame, dirs: Dict[str, Path]) -> None:
    stats, stats_df = compute_subject_stats(manifest, dirs)
    stats_df.to_csv(dirs["logs"] / "subject_frame_normalization_stats.csv", index=False, encoding="utf-8-sig")

    all_meta_rows = []
    high_meta_rows = []
    video_summary_rows = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Normalize + rolling + sequences"):
        video_id = str(row["video_id"])
        subject_id = str(row["subject_id"])
        raw_path = dirs["frame_raw"] / f"{safe_name(video_id)}_frame_features_raw.csv"
        raw_df = read_csv_safe(raw_path)

        if raw_df.empty:
            video_summary_rows.append({"video_id": video_id, "subject_id": subject_id, "status": "empty_raw_frame_csv"})
            continue
        if subject_id not in stats:
            video_summary_rows.append({"video_id": video_id, "subject_id": subject_id, "status": "no_subject_calibration"})
            continue

        mean = stats[subject_id]["mean"]
        std = stats[subject_id]["std"]
        norm_df = normalize_frame_df(raw_df, mean, std)
        norm_path = dirs["frame_norm"] / f"{safe_name(video_id)}_frame_features_norm.csv"
        norm_df.to_csv(norm_path, index=False, encoding="utf-8-sig")

        events_df = detect_blinks_from_pclosed(norm_df)
        events_path = dirs["blink_events"] / f"{safe_name(video_id)}_blink_events_eye_model.csv"
        events_df.to_csv(events_path, index=False, encoding="utf-8-sig")

        rolling_df = compute_rolling_features(norm_df, events_df)
        # Feature safety. Replace non-finite with 0 except time_since_last_blink.
        for col in FRAME_FEATURE_COLS:
            rolling_df[col] = pd.to_numeric(rolling_df[col], errors="coerce")
            if col == "time_since_last_blink":
                rolling_df[col] = rolling_df[col].replace([np.inf, -np.inf], np.nan).fillna(float(CONFIG["TIME_SINCE_LAST_BLINK_CAP_SEC"]))
            else:
                rolling_df[col] = rolling_df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        rolling_path = dirs["rolling"] / f"{safe_name(video_id)}_rolling_features.csv"
        rolling_df.to_csv(rolling_path, index=False, encoding="utf-8-sig")

        all_rows, high_rows = build_sequences_for_video(row, rolling_df, dirs)
        all_meta_rows.extend(all_rows)
        high_meta_rows.extend(high_rows)

        video_summary_rows.append({
            "video_id": video_id,
            "subject_id": subject_id,
            "label": row["label"],
            "num_frames": len(raw_df),
            "face_detection_rate": float(raw_df["face_detected"].astype(float).mean()) if len(raw_df) else 0.0,
            "num_blinks": len(events_df),
            "num_all_sequences": len(all_rows),
            "num_high_conf_sequences": len(high_rows),
            "status": "ok",
        })

    all_meta = pd.DataFrame(all_meta_rows)
    high_meta = pd.DataFrame(high_meta_rows)
    video_summary = pd.DataFrame(video_summary_rows)

    # high-confidence 전용 버전
    # - 실제 학습에 사용할 데이터는 high_meta만 저장한다.
    # - 학습 코드가 X_all_inherited 파일을 먼저 찾는 구조라서 all 파일은 dummy로 만든다.
    # - dummy all은 high-confidence와 완전히 같은 내용이다.
    all_meta.to_csv(dirs["logs"] / "sequence_metadata_candidates_with_confidence.csv", index=False, encoding="utf-8-sig")
    high_meta.to_csv(dirs["root"] / "sequence_metadata_high_confidence.csv", index=False, encoding="utf-8-sig")

    dummy_all_meta = high_meta.copy()
    if len(dummy_all_meta) > 0:
        dummy_all_meta["label_source"] = "dummy_all_mirrors_high_confidence"
        dummy_all_meta["is_dummy_all"] = 1
    else:
        dummy_all_meta = high_meta.copy()
    dummy_all_meta.to_csv(dirs["root"] / "sequence_metadata_all_inherited.csv", index=False, encoding="utf-8-sig")
    video_summary.to_csv(dirs["logs"] / "video_frame_sequence_summary.csv", index=False, encoding="utf-8-sig")

    print("\n[Dataset metadata]")
    print(f"  candidate sequences before filtering: {len(all_meta)}")
    if len(all_meta) > 0:
        print(all_meta["label"].value_counts())
    print(f"  high confidence sequences: {len(high_meta)}")
    if len(high_meta) > 0:
        print(high_meta["label"].value_counts())
    print(f"  dummy all sequences for training-code compatibility: {len(dummy_all_meta)}")

    if bool(CONFIG["SAVE_ALL_ARRAYS"]):
        print("\n[Stack arrays]")
        save_stacked_arrays(high_meta, dirs["root"] / "X_high_confidence.npy", dirs["root"] / "y_high_confidence.npy")
        # 학습 코드 호환용 dummy all 파일. 내용은 high-confidence와 동일하다.
        save_stacked_arrays(dummy_all_meta, dirs["root"] / "X_all_inherited.npy", dirs["root"] / "y_all_inherited.npy")
