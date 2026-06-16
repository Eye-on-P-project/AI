from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List

import cv2
import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError("torch가 없습니다. pip install torch 로 설치하세요.") from e

from .config import CONFIG
from .utils import choose_device

_EYE_MODEL_CACHE = None
_EYE_MODEL_DEVICE_CACHE = None
def _strip_module_prefix(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def _find_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in CONFIG["CHECKPOINT_STATE_KEYS"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # checkpoint 자체가 state_dict인 경우
        if all(hasattr(v, "shape") for v in ckpt.values()):
            return ckpt
    return ckpt


def _build_custom_model():
    py_path = CONFIG["CUSTOM_MODEL_PY"]
    cls_name = CONFIG["CUSTOM_MODEL_CLASS"]
    if not py_path or not cls_name:
        raise ValueError("custom_python backend는 CUSTOM_MODEL_PY와 CUSTOM_MODEL_CLASS가 필요합니다.")

    spec = importlib.util.spec_from_file_location("custom_eye_model_module", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"custom model file을 로드하지 못했습니다: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, cls_name)
    return cls(**CONFIG.get("CUSTOM_MODEL_KWARGS", {}))


def load_eye_model():
    global _EYE_MODEL_CACHE, _EYE_MODEL_DEVICE_CACHE
    if _EYE_MODEL_CACHE is not None:
        return _EYE_MODEL_CACHE, _EYE_MODEL_DEVICE_CACHE

    device = choose_device()
    backend = str(CONFIG["EYE_MODEL_BACKEND"]).lower()
    weight_path = Path(CONFIG["EYE_MODEL_WEIGHT_PATH"])

    if backend == "torchscript":
        if not weight_path.exists():
            raise FileNotFoundError(f"눈 모델 파일이 없습니다: {weight_path}")
        model = torch.jit.load(str(weight_path), map_location=device)
        model.eval().to(device)
    elif backend == "timm":
        try:
            import timm
        except ImportError as e:
            raise ImportError("timm이 없습니다. pip install timm 후 다시 실행하세요.") from e
        model = timm.create_model(
            CONFIG["EYE_MODEL_TIMM_NAME"],
            pretrained=False,
            num_classes=int(CONFIG["EYE_NUM_CLASSES"]),
        )
        if not weight_path.exists():
            raise FileNotFoundError(f"눈 모델 weight가 없습니다: {weight_path}")
        ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        state = _strip_module_prefix(_find_state_dict(ckpt))
        missing, unexpected = model.load_state_dict(state, strict=bool(CONFIG["LOAD_STRICT"]))
        if missing:
            print(f"[WARN] eye model missing keys: {len(missing)}")
        if unexpected:
            print(f"[WARN] eye model unexpected keys: {len(unexpected)}")
        model.eval().to(device)
    elif backend == "custom_python":
        model = _build_custom_model()
        if not weight_path.exists():
            raise FileNotFoundError(f"눈 모델 weight가 없습니다: {weight_path}")
        ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        state = _strip_module_prefix(_find_state_dict(ckpt))
        missing, unexpected = model.load_state_dict(state, strict=bool(CONFIG["LOAD_STRICT"]))
        if missing:
            print(f"[WARN] eye model missing keys: {len(missing)}")
        if unexpected:
            print(f"[WARN] eye model unexpected keys: {len(unexpected)}")
        model.eval().to(device)
    else:
        raise ValueError(f"지원하지 않는 EYE_MODEL_BACKEND: {backend}")

    _EYE_MODEL_CACHE = model
    _EYE_MODEL_DEVICE_CACHE = device
    return model, device


def preprocess_eye_crops(crops_rgb: List[np.ndarray]) -> torch.Tensor:
    size = int(CONFIG["EYE_MODEL_INPUT_SIZE"])
    mean = np.array(CONFIG["EYE_MODEL_MEAN"], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(CONFIG["EYE_MODEL_STD"], dtype=np.float32).reshape(1, 1, 3)

    arrs = []
    for crop in crops_rgb:
        if crop is None or crop.size == 0:
            crop = np.zeros((size, size, 3), dtype=np.uint8)
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        x = crop.astype(np.float32) / 255.0
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))
        arrs.append(x)
    return torch.from_numpy(np.stack(arrs, axis=0)).float()


def predict_p_closed(crops_rgb: List[np.ndarray]) -> np.ndarray:
    if len(crops_rgb) == 0:
        return np.empty((0,), dtype=np.float32)

    model, device = load_eye_model()
    x = preprocess_eye_crops(crops_rgb).to(device)
    with torch.no_grad():
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        mode = str(CONFIG["EYE_MODEL_OUTPUT_MODE"]).lower()
        if mode == "sigmoid" or (mode == "auto" and out.ndim == 2 and out.shape[1] == 1):
            p = torch.sigmoid(out.reshape(-1))
        else:
            prob = torch.softmax(out, dim=1)
            p = prob[:, int(CONFIG["EYE_CLOSED_CLASS_INDEX"])]
    return p.detach().cpu().numpy().astype(np.float32)
