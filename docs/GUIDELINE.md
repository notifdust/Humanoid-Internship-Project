# Ultimate guideline — Humanoid internship challenge

**Single source of truth for what we build, why, and in what order.**  
Deadline: **Thursday 16 Oct 2026, 23:59 BST**. Hardware: **RTX 4060 laptop (8 GB)**.

---

## 1. Data verdict — is it good enough?

**Yes — good enough to ship a strong challenge submission.** Do not re-record unless a clip is broken.

| Criterion | Your data | Verdict |
|-----------|-----------|---------|
| Personal collection | Phone egocentric, you filmed it | Satisfies constraint |
| Volume | 42 clips, ~5.6 min | Above the 20–25 target |
| Consistency | Same mug/plate/table/task | Excellent for small-data finetune |
| Contrast | Fuchsia mug, white plate, black table | Easy for vision / hand trackers |
| Viewpoint | True ego (chest/head-ish) | Matches EgoVLA / EgoMimic spirit |
| Hand | Right only | Good (no left/right mix) |
| Labels | `data/labels.csv` | Usable |
| Resolution | 4K @ 30 fps | Overkill → **downsample to 1080p** |
| Clip length | Some short (~4 s), some ~12 s | OK; drop only if grasp incomplete |

**Known limits (acknowledge in README, don’t block):**
- No robot teleop demos (expected for this challenge).
- Monocular ego → depth/scale are approximate (EgoVLA-style retargeting, not perfect IK).
- Not a LIBERO-identical scene (transfer is the creative part).
- First MediaPipe pass: mean hand visibility ~66% (ego mirror issue). **Mitigation:** detect on horizontally flipped frames, lower conf, interpolate gaps; prefer clips with `qa_pass` for val/test.

**Action for you:** none on data unless Phase A QA still leaves &lt;15 usable clips after flip fix — then we may ask for 5–10 cleaner reshoots of the worst IDs only.

---

## 2. Challenge north star

> Use **your** ego videos as a real signal to **drive a Panda-style arm in LIBERO** (and/or show a **world-model** prediction), with a clear, non-sloppy write-up.

Judges care about: creativity under the personal-data constraint, sim policy and/or WM quality, simple clear presentation.

---

## 3. Chosen technical story (locked)

**Primary path (must ship):**

```
Ego 4K video
  → 1080p preprocess
  → MediaPipe Hands (wrist / fingertips)     [fast on 4060; HaMeR optional upgrade]
  → Retarget wrist trajectory → Panda EE deltas (EgoVLA-inspired)
  → Build LeRobot-style episodes from retargeted actions + frames
  → LoRA / light finetune SmolVLA (or ACT/Diffusion if VLA blocked)
  → Rollout in LIBERO (pick/place-like task) + side-by-side video in README
```

**Secondary path (time-boxed, if primary works):**
- Small **action-conditioned / future-frame** predictor on your clips (WM teaser; cite IRASim / UWM / iVideoGPT ideas, keep model tiny).

**With the 16 Oct deadline — still primary-first; optional upgrades if Phase C is solid:**
- HaMeR upgrade for a subset of high-quality clips  
- Small world-model teaser (future-frame predictor)  
- Better retarget calibration / IK  

**Still out of scope:**
- Full UWM / iVideoGPT / GWM from scratch  
- Real robot  
- Isaac Sim  

---

## 4. Paper / code map → what we steal

| Idea | Source | How we use it |
|------|--------|----------------|
| Ego human video → robot via wrist/hand + IK/retarget | **EgoVLA** | Conceptual backbone of our pipeline |
| Treat human demos as first-class | **EgoMimic** | Narrative + co-training mindset |
| Hand from RGB | **HaMeR** paper; **MediaPipe** in practice | Pose front-end (MP first) |
| Human→robot hand mapping | **dex-retargeting** | Optional; we start with wrist→EE |
| Affordable VLA + LIBERO | **SmolVLA** + **LeRobot** | Policy + eval stack |
| LIBERO Panda tasks | **LIBERO** | Sim target |
| Low-VRAM finetune | SmolVLA-LIBERO LoRA forks (3060-class) | Pattern for 4060 |
| WM as learned simulator | **IRASim / UWM / iVideoGPT** | Secondary demo only |
| Classic WM intuition | Ha & Schmidhuber, DreamerV3 | README framing only |

Full lists: `resources/papers/READING_LIST.md`, `resources/github/SIMILAR_PROJECTS.md`.

---

## 5. Repo layout (target)

```
data/
  raw/ego/                 # your originals (4K) — gitignored
  processed/ego_1080/      # downscaled videos
  processed/hands/         # per-clip JSON (keypoints, wrist traj)
  processed/retarget/      # EE trajectories / pseudo-actions
  labels.csv
docs/
  GUIDELINE.md             # this file
  architecture.md
src/
  preprocess/              # resize, fps sanitize
  hands/                   # MediaPipe extraction
  retarget/                # wrist → Panda action
  dataset/                 # LeRobot / HDF5 builders
  train/                   # SmolVLA LoRA scripts
  eval/                    # LIBERO rollout helpers
scripts/                   # one-command entrypoints
outputs/                   # videos, plots, checkpoints — gitignored
```

---

## 6. Execution order (do in sequence)

### Phase A — Data ready (Day 0–0.5) ✅ / in progress
1. [x] Organize + label ego videos  
2. [x] Downsample all clips → `data/processed/ego_1080/` (1080p, 30 fps)  
3. [x] MediaPipe hand tracks → `data/processed/hands/*.json` + QA (ego-flip)  
4. [x] Flag low-visibility clips; keep all with interpolation, prefer `qa_pass` for eval  

### Phase B — Retarget (Day 0.5–1)
5. [x] Normalize wrist xy (image) + crude z from hand scale → 3D-ish EE path  
6. [x] Map to Panda **delta EE** / gripper open-close heuristic  
7. [x] Visualize overlay: ego frame + wrist trail (`src/viz/make_overlays.py`)  
7b. [x] Pack NPZ episodes (`src/dataset/build_episodes.py`)  

### Phase C — Policy in sim (through 16 Oct)
8. [x] Install PyTorch CUDA + LeRobot on 4060 (**Python 3.12** venv; CUDA 12.4)  
9. [x] Convert NPZ → LeRobot dataset (`data/processed/lerobot_pick_place_mug`, image mode)  
10. [x] BC baseline smoke-test on GPU → `outputs/checkpoints/bc_baseline/` + eval overlay  
11. [x] SmolVLA finetune smoke OK + **2000-step** done (`outputs/train/smolvla_custom/checkpoint-2000`)  
11b. [x] Offline SmolVLA overlay on held-out clip 41 (`outputs/eval/smolvla_rollout_41.mp4`, MSE≈0.013)  
11c. [x] Windows MuJoCo EE proxy sim from GT + SmolVLA actions (`outputs/eval/mujoco_ee_*.mp4`)  
12. [x] Full LIBERO Panda open-loop rollout in **WSL2** (`outputs/eval/libero_openloop_smolvla_41.mp4`)  
    - Venv: `~/hip_wsl_libero` (Linux home; not OneDrive)  
    - Pins: `robosuite==1.4.1`, `mujoco==3.1.1`, `PYTHONPATH=third_party/LIBERO`  
    - Runner: `scripts/wsl_run_libero_rollout.sh`

### Phase D — Polish + harden (revised 8 Oct — see §11)
**P0 (must before submit)**
13. [x] **Closed-loop LIBERO** ego-ft domain transfer (`outputs/eval/libero_closedloop_ego_ft.mp4`, success=False as expected) + `docs/RESEARCH_CLOSEDLOOP.md`  
14. [x] Side-by-side figure: ego | wrist overlay | LIBERO (`outputs/eval/side_by_side_41.mp4`)  
15. [x] Ablation: pretrained 0.029 → finetuned 0.023 → BC 0.021 (`outputs/eval/ablation_mse.json`)  
16. [~] README polished with honesty + cites; **public GitHub still pending (needs your push/URL)**

**P1 (time-boxed, after P0)**
17. [x] Tiny WM future-frame teaser (`src/wm/`, `outputs/wm/future_frame_strip_{30,41}.gif`)  
18. [ ] N-seed LIBERO success rate if closed-loop works  
19. [ ] Retarget qualitative plot (wrist path vs integrated EE)

**Cut (do not start unless P0 done with days left)**
- HaMeR / MANO, dex-retargeting IK, full UWM/iVideoGPT/GWM/Dreamer, Isaac, re-film

---

## 7. Training defaults (4060-safe)

| Knob | Default |
|------|---------|
| Input res | 224–256 (model), videos stored at 1080p |
| Batch size | 1–4 + grad accumulation |
| Precision | fp16 / bf16 |
| Method | LoRA on SmolVLA (not full FT) |
| Epochs | Short (overfit intentionally on 42 demos is OK for demo) |
| Early stop | Val instruction success / action MSE on held-out clips |

If SmolVLA install fails: fall back to **ACT or Diffusion Policy** on retargeted actions (still valid; cite Diffusion Policy).

---

## 8. Success criteria (submission)

**Must have**
- [x] Personal data clearly used (not decorative)  
- [x] Reproducible script: video → hands → retarget → (train) → sim  
- [x] At least one LIBERO (or equivalent) rollout video  
- [x] Honest README (open-loop vs closed-loop domain gap + ablation)  

**Nice to have**
- [x] Quantitative metric (ablation MSE: pretrained 0.029 / finetuned 0.023 / BC 0.021)  
- [x] WM prediction strip (`outputs/wm/future_frame_strip_*.gif`)  
- [x] Comparison: pretrained vs finetuned (personal data helps; BC still wins proxy)  
- [ ] Closed-loop LIBERO task success (ego-ft expected fail; optional libero-ref later)  

---

## 9. What you (human) still need to do

| When | Action |
|------|--------|
| Now | Nothing on filming — data is enough |
| If asked | Confirm a clip is success/fail if QA is ambiguous |
| Before submit | Review README narrative; fill application form + GitHub URL |
| Optional | Record 3 side-camera clips (not required) |

---

## 10. Decision log

| Decision | Choice | Reason |
|----------|--------|--------|
| Primary angle | Ego → retarget → SmolVLA → LIBERO | Matches posting example + EgoVLA |
| Pose frontend | MediaPipe first | Speed on 4060; HaMeR later if time |
| World models | Secondary only | Deadline; still cite lineage |
| Data reshoot | No | Quality/contrast/volume sufficient |
| LIBERO eval mode | Open-loop shipped; **closed-loop next** | Forks show image→action is what judges expect as “policy in sim” |
| HaMeR / full WM | Cut unless P0 done | 8 days left; diminishing return vs closed-loop + presentation |

---

## 11. Dynamic replan (8 Oct 2026)

**What the research says we still lack**
- **EgoVLA / EgoMimic:** we match the *story* (ego human → robot actions) but not MANO+IK fidelity — acceptable if we show retarget overlays and name the approximation.
- **SmolVLA×LIBERO forks:** real eval is **closed-loop** on `agentview` with relative OSC; our open-loop MP4 proves action transfer, not a reactive policy in sim.
- **UWM / IRASim / iVideoGPT / Sutton–Barto:** WM remains a *nice* second exhibit; do not derail P0 for a full dreamer-style stack.
- **Diffusion Policy / Octo:** already covered by BC + SmolVLA; no need for a third policy family.

**Priority order until 16 Oct:** closed-loop LIBERO → side-by-side + ablation → GitHub/README → (optional) WM GIF.

---

*Update this file when a Phase checkbox flips or a decision changes. Do not fork a second competing plan.*
