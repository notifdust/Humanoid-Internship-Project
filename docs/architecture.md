# Architecture

Aligned with [`GUIDELINE.md`](GUIDELINE.md).

```
data/raw/ego/*.mp4  (4K, personal)
        │
        ▼  src/preprocess/resize_videos.py
data/processed/ego_1080/*.mp4
        │
        ▼  src/hands/extract_mediapipe.py
data/processed/hands/*.json   (wrist, landmarks, confidence)
        │
        ▼  src/retarget/wrist_to_panda.py
data/processed/retarget/*.npz  (EE deltas + gripper)
        │
        ├──────────────────────────────┐
        ▼                              ▼
LeRobot dataset              (optional) tiny video WM
        │
        ▼
SmolVLA LoRA finetune (4060)
        │
        ▼
LIBERO rollout MP4  →  README side-by-side with ego clip
```

## Modules

| Module | Role |
|--------|------|
| `preprocess` | 4K → 1080p, constant 30 fps |
| `hands` | MediaPipe right-hand landmarks |
| `retarget` | EgoVLA-inspired wrist → Panda deltas |
| `dataset` | Pack episodes for LeRobot |
| `train` | LoRA finetune entrypoint |
| `eval` | LIBERO rollout + metrics |
