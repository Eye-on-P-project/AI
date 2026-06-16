from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from .config import CONFIG, FRAME_FEATURE_COLS
from .dataset import build_frame_sequence_datasets
from .eye_model import load_eye_model
from .frame_extraction import process_one_video
from .utils import choose_device, get_dirs, read_manifest, save_json
def run_all() -> None:
    out_root = Path(CONFIG["OUT_ROOT"])
    dirs = get_dirs(out_root)

    # CUDA + multiprocessing은 모델 복제와 CUDA context 문제 때문에 안정성이 낮음.
    device = choose_device()
    if device.startswith("cuda") and int(CONFIG["NUM_WORKERS"]) > 1:
        print("[WARN] CUDA eye model 사용 시 NUM_WORKERS를 1로 낮춥니다.")
        CONFIG["NUM_WORKERS"] = 1

    save_json(CONFIG, out_root / "preprocess_frame_sequence_config.json")
    save_json({"frame_feature_order": FRAME_FEATURE_COLS}, out_root / "feature_order.json")

    manifest = read_manifest(CONFIG["MANIFEST_CSV"])
    if CONFIG["PROCESS_ONLY_FIRST_N_VIDEOS"] is not None:
        manifest = manifest.head(int(CONFIG["PROCESS_ONLY_FIRST_N_VIDEOS"])).copy()
    manifest.to_csv(out_root / "video_manifest_used.csv", index=False, encoding="utf-8-sig")

    print("[INFO] Frame-sequence preprocessing")
    print(f"  videos: {len(manifest)}")
    print(f"  subjects: {manifest['subject_id'].nunique()}")
    print(f"  output: {out_root}")
    print(f"  eye model backend: {CONFIG['EYE_MODEL_BACKEND']}")
    print(f"  eye model device: {device}")
    print(f"  workers: {CONFIG['NUM_WORKERS']}")
    print(f"  target fps: {CONFIG['TARGET_FPS']}")
    print(f"  sequence shape per sample: [{int(CONFIG['TARGET_FPS'] * CONFIG['SEQUENCE_SECONDS'])}, {len(FRAME_FEATURE_COLS)}]")

    # Load once early to fail fast.
    if bool(CONFIG["USE_EYE_MODEL"]):
        _ = load_eye_model()

    row_dicts = manifest.to_dict(orient="records")
    dirs_str = {k: str(v) for k, v in dirs.items()}

    logs = []
    if int(CONFIG["NUM_WORKERS"]) <= 1:
        for row in tqdm(row_dicts, desc="Frame feature extraction"):
            logs.append(process_one_video(row, dirs_str))
    else:
        # CPU 전처리용. 각 process에서 eye model을 따로 로드함.
        with ProcessPoolExecutor(max_workers=int(CONFIG["NUM_WORKERS"])) as ex:
            futures = [ex.submit(process_one_video, row, dirs_str) for row in row_dicts]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Frame feature extraction"):
                logs.append(fut.result())

    pd.DataFrame(logs).to_csv(dirs["logs"] / "video_processing_log.csv", index=False, encoding="utf-8-sig")

    print("\n[INFO] Build frame-sequence datasets")
    build_frame_sequence_datasets(manifest, dirs)

    print("\n[DONE] preprocessing finished")
    print(f"[SAVE ROOT] {out_root}")

if __name__ == "__main__":
    run_all()
