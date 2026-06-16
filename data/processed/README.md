# Processed Data Directory

이 폴더는 frame sequence preprocessing 결과를 저장하는 위치입니다.
전처리 결과 `.npy`, `.csv` 파일은 용량이 클 수 있으므로 GitHub에는 업로드하지 않습니다.

## Expected Files After Preprocessing

```text
data/processed/frame_sequences/
├─ X_high_confidence.npy
├─ y_high_confidence.npy
├─ X_all_inherited.npy
├─ y_all_inherited.npy
├─ sequence_metadata_high_confidence.csv
├─ sequence_metadata_all_inherited.csv
├─ preprocess_frame_sequence_config.json
├─ feature_order.json
├─ video_manifest_used.csv
├─ frame_features_raw/
├─ frame_features_norm/
├─ blink_events_eye_model/
├─ rolling_features/
├─ sequences_high_confidence/
└─ logs/
```

`X_all_inherited.npy` and `y_all_inherited.npy` may be dummy copies of the high-confidence dataset when the training code expects those filenames.
