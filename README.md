# Humanoid Internship Project

Submission workspace for the [Humanoid — Internship, Robot Learning Research](https://www.physicalai.jobs/humanoid/internship-robot-learning-research) challenge.

**Constraint:** use **personally collected** real data (phone egocentric manipulation video) to drive a simulated manipulator (LIBERO / Panda), creatively using VLA methods.

**Deadline:** Thursday, 16 October 2026, 23:59 BST.

## Pipeline

```
Ego 4K video → 1080p → MediaPipe Hands → wrist→Panda retarget
  → LeRobot episodes → SmolVLA finetune → LIBERO / MuJoCo rollout
```

Inspired by [EgoVLA](https://rchalyang.github.io/EgoVLA/) (human ego → robot) and [SmolVLA](https://arxiv.org/abs/2506.01844) + [LeRobot](https://github.com/huggingface/lerobot) / [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

## Status

| Stage | Result |
|-------|--------|
| Data | 42 personal ego clips (`data/raw/ego/` + `data/labels.csv`) |
| Hands / retarget | MediaPipe + wrist→EE deltas; overlays in `outputs/overlays/` |
| BC baseline | `outputs/eval/bc_rollout_41.mp4` |
| SmolVLA (2000 steps) | `outputs/train/smolvla_custom/checkpoint-2000` |
| Ablation (held-out MSE) | pretrained **0.029** → finetuned **0.023** → BC **0.021** (`outputs/eval/ablation_mse.json`) |
| Side-by-side | `outputs/eval/side_by_side_41.mp4` (ego \| pred \| LIBERO) |
| MuJoCo EE proxy | `outputs/eval/mujoco_ee_smolvla_41.mp4` |
| LIBERO open-loop | `outputs/eval/libero_openloop_smolvla_41.mp4` (personal-data actions) |
| LIBERO closed-loop | `outputs/eval/libero_closedloop_ego_ft.mp4` (domain-transfer; task success=False, expected) |
| WM teaser (future frame) | `outputs/wm/future_frame_strip_{30,41}.gif` — action-conditioned next-frame (IRASim-style tiny CNN) |
| Retarget QA plots | `outputs/viz/retarget/*_retarget.png` (wrist vs EE path) |
| LIBERO open-loop multi-seed | `outputs/eval/libero_openloop_multiseed.json` |

**Repo:** https://github.com/notifdust/Humanoid-Internship-Project/pull/2  

**How to inspect:** [`docs/HOW_TO_CHECK.md`](docs/HOW_TO_CHECK.md) · **Plan:** [`docs/GUIDELINE.md`](docs/GUIDELINE.md) · **Closed-loop research:** [`docs/RESEARCH_CLOSEDLOOP.md`](docs/RESEARCH_CLOSEDLOOP.md)

## Reproduce (high level)

```powershell
# Windows (Python 3.12 + CUDA)
.\.venv\Scripts\python.exe -m src.train.train_smolvla_custom --steps 2000 --out outputs/train/smolvla_custom
.\.venv\Scripts\python.exe -m src.eval.rollout_smolvla_overlay
.\.venv\Scripts\python.exe -m src.eval.ablation_mse
.\.venv\Scripts\python.exe -m src.viz.make_side_by_side
.\.venv\Scripts\python.exe -m src.wm.train_future_frame
.\.venv\Scripts\python.exe -m src.wm.make_wm_gif

# WSL2 LIBERO (mujoco==3.1.1, robosuite==1.4.1)
wsl -e bash scripts/wsl_run_libero_rollout.sh
wsl -e bash scripts/wsl_run_libero_closedloop.sh
```

## Design notes / honesty

- No robot teleop — actions are **retargeted from ego wrist** (EgoVLA-style; scale/depth approximate).
- Personal data → sim is strongest as **open-loop EE deltas** into LIBERO + offline ego MSE.
- Closed-loop with our ego-finetuned ckpt is **domain transfer** (ego `camera1`/6D ≠ LIBERO `image`/`image2`/8D/7D). Community closed-loop recipes use `HuggingFaceVLA/smolvla_libero`, 180° image rotate, `n_action_steps=1`, relative OSC ([notes](docs/RESEARCH_CLOSEDLOOP.md)).
- Ablation: finetuning on personal data beats pretrained SmolVLA on held-out action MSE; a tiny BC CNN still wins this proxy (overfit-friendly).
- Open-loop LIBERO multi-seed on ego-derived actions: **0/5** task success (expected — wrong objects/scene; demos motion transfer, not LIBERO demos).
- Windows LeRobot CLI tempfile bugs → custom trainer `src/train/train_smolvla_custom.py`.

## Docs / research

| Doc | Purpose |
|-----|---------|
| [`docs/GUIDELINE.md`](docs/GUIDELINE.md) | Ultimate plan, phases, data verdict |
| [`docs/RESEARCH_CLOSEDLOOP.md`](docs/RESEARCH_CLOSEDLOOP.md) | SmolVLA×LIBERO eval pins from online sources |
| [`resources/`](resources/) | GitHub survey, papers, books |

## License

MIT — see [`LICENSE`](LICENSE). Third-party papers and books remain under their own terms.
