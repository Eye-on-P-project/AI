# Raw Data Directory

이 폴더는 원본 데이터셋을 로컬에 배치하는 위치입니다.
영상, 이미지 데이터셋은 용량이 크기 때문에 GitHub에는 업로드하지 않습니다.

## Expected Structure

```text
data/raw/
├─ video_manifest.csv
├─ videos/
│  └─ original driving videos
├─ eyes_set/
│  ├─ train/
│  │  ├─ open/
│  │  └─ closed/
│  ├─ val/
│  │  ├─ open/
│  │  └─ closed/
│  └─ test/
│     ├─ open/
│     └─ closed/
└─ manifests/
   └─ optional backup manifest files
```

## Manifest Columns

`video_manifest.csv` should contain the following columns:

```text
video_id, fold, part, subject_id, score_label, video_path
```

The `video_path` column can be absolute paths or paths relative to the project root.
