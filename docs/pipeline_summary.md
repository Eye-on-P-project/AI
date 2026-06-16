# Pipeline Summary

1. `scripts/train_eye_state_cnn_gray.py`  
   눈 crop 이미지로 open/closed CNN을 학습합니다.

2. `run_preprocessing.py`  
   영상에서 MediaPipe landmark와 eye CNN 기반 `p_closed`를 추출하고 frame sequence dataset을 생성합니다.

3. `scripts/train_temporal_hierarchical_sleepy_rule.py`  
   생성된 sequence dataset으로 normal/drowsy/sleepy temporal classifier를 학습하고 sequence/video/streaming 평가를 수행합니다.
