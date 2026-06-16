from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import (
    HEAD_MODEL_POINTS,
    HEAD_POSE_IDXS,
    MOUTH_BOTTOM,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    MOUTH_TOP,
)
def _lm_xy(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def compute_eye_ear(landmarks, eye_indices: List[int], image_w: int, image_h: int) -> float:
    pts = np.asarray([_lm_xy(landmarks, idx, image_w, image_h) for idx in eye_indices], dtype=np.float32)
    p1, p2, p3, p4, p5, p6 = pts
    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal <= 1e-6:
        return np.nan
    return float((vertical1 + vertical2) / (2.0 * horizontal))


def compute_mar(landmarks, image_w: int, image_h: int) -> float:
    left = _lm_xy(landmarks, MOUTH_LEFT, image_w, image_h)
    right = _lm_xy(landmarks, MOUTH_RIGHT, image_w, image_h)
    top = _lm_xy(landmarks, MOUTH_TOP, image_w, image_h)
    bottom = _lm_xy(landmarks, MOUTH_BOTTOM, image_w, image_h)
    horizontal = np.linalg.norm(left - right)
    vertical = np.linalg.norm(top - bottom)
    if horizontal <= 1e-6:
        return np.nan
    return float(vertical / horizontal)


def crop_eye_rgb(frame_bgr: np.ndarray, landmarks, eye_indices: List[int], pad_ratio: float) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    pts = np.asarray([_lm_xy(landmarks, idx, w, h) for idx in eye_indices], dtype=np.float32)
    if pts.size == 0 or not np.isfinite(pts).all():
        return None

    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    bw = x2 - x1
    bh = y2 - y1
    side = max(bw, bh) * (1.0 + float(pad_ratio))
    if side <= 2:
        return None
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    x1 = int(max(0, round(cx - side / 2.0)))
    x2 = int(min(w, round(cx + side / 2.0)))
    y1 = int(max(0, round(cy - side / 2.0)))
    y2 = int(min(h, round(cy + side / 2.0)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop_bgr = frame_bgr[y1:y2, x1:x2]
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def compute_head_pose(landmarks, image_w: int, image_h: int) -> Tuple[float, float, float]:
    try:
        image_points = np.array([
            _lm_xy(landmarks, HEAD_POSE_IDXS["nose"], image_w, image_h),
            _lm_xy(landmarks, HEAD_POSE_IDXS["chin"], image_w, image_h),
            _lm_xy(landmarks, HEAD_POSE_IDXS["left_eye_outer"], image_w, image_h),
            _lm_xy(landmarks, HEAD_POSE_IDXS["right_eye_outer"], image_w, image_h),
            _lm_xy(landmarks, HEAD_POSE_IDXS["left_mouth"], image_w, image_h),
            _lm_xy(landmarks, HEAD_POSE_IDXS["right_mouth"], image_w, image_h),
        ], dtype=np.float64)

        focal_length = float(image_w)
        center = (image_w / 2.0, image_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            HEAD_MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return np.nan, np.nan, np.nan
        rot_mat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
        pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])
        return pitch, yaw, roll
    except Exception:
        return np.nan, np.nan, np.nan
