# Architecture notes (placeholder)

High-level pipeline once implementation starts:

```
[Phone egocentric video]
        │
        ▼
[Pose / features] ── HaMeR / MediaPipe / learned encoder
        │
        ▼
[Retarget or latent align] ── human actions ↔ Panda / LIBERO actions
        │
        ├──► [VLA finetune] ── SmolVLA + LeRobot ──► LIBERO rollout
        │
        └──► [World model] ── UWM / iVideoGPT / small video predictor
                              └── optional planning / synthetic rollouts
```

Details TBD after choosing primary angle (see `PROJECT_OUTLINE.md`).
