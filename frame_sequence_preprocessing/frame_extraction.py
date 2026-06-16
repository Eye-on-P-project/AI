from __future__ import annotations

import math
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError("mediapipe가 없습니다. pip install mediapipe 로 설치하세요.") from e

from .config import (
    CONFIG,
    LEFT_EYE_CROP_IDX,
    LEFT_EYE_EAR_IDX,
    RIGHT_EYE_CROP_IDX,
    RIGHT_EYE_EAR_IDX,
)
from .eye_model import predict_p_closed
from .face_features import compute_eye_ear, compute_head_pose, compute_mar, crop_eye_rgb
from .utils import resize_for_detection, safe_name
def flush_eye_batch(rows: List[dict], pending_crops: List[np.ndarray], pending_meta: List[Tuple[int, str]]) -> None:
    if not pending_crops:
        return
    probs = predict_p_closed(pending_crops)
    for prob, (row_idx, side) in zip(probs, pending_meta):
        rows[row_idx][f"p_closed_{side}"] = float(prob)
    pending_crops.clear()
    pending_meta.clear()


def extract_frame_features(video_row: dict, frame_csv: Path) -> Dict[str, object]:
    video_id = str(video_row["video_id"])
    video_path = Path(str(video_row["video_path"]))
    if not video_path.exists():
        raise FileNotFoundError(f"영상 파일이 없습니다: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV가 영상을 열지 못했습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6 or math.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_fps = float(CONFIG["TARGET_FPS"])
    frame_step = max(1, int(round(fps / target_fps))) if target_fps > 0 else 1
    effective_fps = fps / frame_step

    rows: List[dict] = []
    pending_crops: List[np.ndarray] = []
    pending_meta: List[Tuple[int, str]] = []
    batch_size = int(CONFIG["EYE_MODEL_BATCH_SIZE"])

    frame_idx = 0
    processed_idx = 0
    face_detected_count = 0
    pad = float(CONFIG["EYE_CROP_PADDING"])

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=bool(CONFIG["REFINE_LANDMARKS"]),
        min_detection_confidence=float(CONFIG["MIN_DETECTION_CONFIDENCE"]),
        min_tracking_confidence=float(CONFIG["MIN_TRACKING_CONFIDENCE"]),
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / fps
            detect_frame = resize_for_detection(frame, CONFIG["DETECT_MAX_WIDTH"])
            rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
            h_det, w_det = rgb.shape[:2]
            result = face_mesh.process(rgb)

            row = {
                "video_id": video_id,
                "processed_idx": processed_idx,
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "source_fps": fps,
                "effective_fps": effective_fps,
                "frame_step": frame_step,
                "face_detected": 0,
                "left_ear": np.nan,
                "right_ear": np.nan,
                "mean_ear": np.nan,
                "mar": np.nan,
                "pitch": np.nan,
                "yaw": np.nan,
                "roll": np.nan,
                "p_closed_left": np.nan,
                "p_closed_right": np.nan,
                "p_closed": np.nan,
            }
            rows.append(row)
            row_idx = len(rows) - 1

            if result.multi_face_landmarks:
                face_detected_count += 1
                row["face_detected"] = 1
                landmarks = result.multi_face_landmarks[0].landmark

                row["left_ear"] = compute_eye_ear(landmarks, LEFT_EYE_EAR_IDX, w_det, h_det)
                row["right_ear"] = compute_eye_ear(landmarks, RIGHT_EYE_EAR_IDX, w_det, h_det)
                if np.isfinite(row["left_ear"]) and np.isfinite(row["right_ear"]):
                    row["mean_ear"] = (row["left_ear"] + row["right_ear"]) / 2.0

                row["mar"] = compute_mar(landmarks, w_det, h_det)
                pitch, yaw, roll = compute_head_pose(landmarks, w_det, h_det)
                row["pitch"] = pitch
                row["yaw"] = yaw
                row["roll"] = roll

                # crop from original frame. FaceMesh normalized coords are compatible with original image ratio.
                landmarks_orig = result.multi_face_landmarks[0].landmark
                left_crop = crop_eye_rgb(frame, landmarks_orig, LEFT_EYE_CROP_IDX, pad)
                right_crop = crop_eye_rgb(frame, landmarks_orig, RIGHT_EYE_CROP_IDX, pad)

                if left_crop is not None:
                    pending_crops.append(left_crop)
                    pending_meta.append((row_idx, "left"))
                if right_crop is not None:
                    pending_crops.append(right_crop)
                    pending_meta.append((row_idx, "right"))

                if len(pending_crops) >= batch_size:
                    flush_eye_batch(rows, pending_crops, pending_meta)

            processed_idx += 1
            frame_idx += 1

    cap.release()
    flush_eye_batch(rows, pending_crops, pending_meta)

    df = pd.DataFrame(rows)
    if not df.empty:
        p_left = df["p_closed_left"].astype(float)
        p_right = df["p_closed_right"].astype(float)
        df["p_closed"] = pd.concat([p_left, p_right], axis=1).mean(axis=1, skipna=True)

    df.to_csv(frame_csv, index=False, encoding="utf-8-sig")

    return {
        "video_id": video_id,
        "status": "ok",
        "source_fps": fps,
        "effective_fps": effective_fps,
        "frame_step": frame_step,
        "total_frames_reported": total_frames,
        "processed_frames": len(df),
        "face_detected_frames": face_detected_count,
        "face_detection_rate": face_detected_count / max(len(df), 1),
    }


def process_one_video(row_dict: dict, dirs_str: Dict[str, str]) -> Dict[str, object]:
    dirs = {k: Path(v) for k, v in dirs_str.items()}
    video_id = str(row_dict["video_id"])
    frame_csv = dirs["frame_raw"] / f"{safe_name(video_id)}_frame_features_raw.csv"

    try:
        logs = {"video_id": video_id, "video_path": row_dict.get("video_path", "")}
        if CONFIG["RESUME"] and frame_csv.exists():
            logs["frame_status"] = "skipped_exists"
            try:
                df = pd.read_csv(frame_csv, encoding="utf-8-sig")
                logs["processed_frames"] = len(df)
                logs["face_detection_rate"] = float(df["face_detected"].mean()) if len(df) else 0.0
            except Exception:
                pass
        else:
            out = extract_frame_features(row_dict, frame_csv)
            logs.update(out)
            logs["frame_status"] = out.get("status", "ok")
        logs["status"] = "ok"
        return logs
    except Exception as e:
        return {
            "video_id": video_id,
            "video_path": row_dict.get("video_path", ""),
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
