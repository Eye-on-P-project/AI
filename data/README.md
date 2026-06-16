# Data Directory

이 폴더는 원본 데이터와 전처리 결과를 로컬에서 관리하기 위한 위치입니다.
대용량 파일은 `.gitignore`로 제외되어 GitHub에 업로드되지 않습니다.

```text
data/
├─ raw/          # 원본 영상, eye crop 이미지 데이터셋, manifest
└─ processed/    # 전처리로 생성된 frame sequence npy/csv
```

각 하위 폴더의 자세한 구조는 `data/raw/README.md`, `data/processed/README.md`를 참고하세요.
