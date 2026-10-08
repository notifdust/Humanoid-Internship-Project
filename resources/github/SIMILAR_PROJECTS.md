# Similar GitHub implementations

Curated survey (Oct 2026) of public repos closest to the Humanoid challenge:
**personally collected egocentric / human manipulation data → VLA or world model → robot control in simulation (e.g. LIBERO / Panda).**

Stars are approximate at survey time.

---

## Tier A — closest to the challenge

| Repo | Why it matters | Stars (~) | URL |
|------|----------------|-----------|-----|
| **huggingface/lerobot** | Official stack for SmolVLA, LIBERO eval, datasets, training/eval scripts. Default baseline for the posting’s example. | 28k | https://github.com/huggingface/lerobot |
| **Lifelong-Robot-Learning/LIBERO** | Official LIBERO benchmark (Panda / robosuite / MuJoCo), demos, lifelong IL baselines. | 2.4k | https://github.com/Lifelong-Robot-Learning/LIBERO |
| **rchalyang/EgoVLA** (project) | VLA trained on **egocentric human video**, MANO hand params, IK + retargeting to robot — closest research match to personal phone data. | — | https://rchalyang.github.io/EgoVLA/ · [arXiv](https://arxiv.org/abs/2507.12440) |
| **dexsuite/dex-retargeting** | Human hand → robot hand retargeting (AnyTeleop lineage); video + pose-dataset pipelines. | 1.3k | https://github.com/dexsuite/dex-retargeting |
| **geopavlakos/hamer** | 3D hand mesh (MANO) from monocular RGB — practical front-end for phone egocentric video. | 1.2k | https://github.com/geopavlakos/hamer |
| **WEIRDLabUW/unified-world-model** | Unified video+action diffusion; trains on robot + **action-free video**; LIBERO eval path. | — | https://github.com/WEIRDLabUW/unified-world-model |
| **thuml/iVideoGPT** | Interactive world model pretrained on human + robot trajectories; action-conditioned video prediction. | ~190 | https://github.com/thuml/iVideoGPT |
| **Gaussian-World-Model/gaussianwm** | 3D Gaussian-splatting world model for manipulation (ICCV 2025). | — | https://github.com/Gaussian-World-Model/gaussianwm |

---

## Tier B — SmolVLA × LIBERO reproductions / tooling

These show how people actually wire SmolVLA into LIBERO (useful for avoiding eval pitfalls).

| Repo | Notes | URL |
|------|-------|-----|
| zuoxingdong/smolvla-libero-eval | Careful LIBERO-Spatial eval log; relative control, MuJoCo pins, `num_steps` ablations. | https://github.com/zuoxingdong/smolvla-libero-eval |
| kgaero/LeRobotSmolVLA | Keyboard teleop demos → HDF5 images → LeRobot dataset → SmolVLA eval. | https://github.com/kgaero/LeRobotSmolVLA |
| 2437buaa/smolvla-libero-repro | Low-VRAM LoRA finetune on RTX 3060 6GB. | https://github.com/2437buaa/smolvla-libero-repro |
| nikoxxxhr/smolvla-libero-lora | Parameter-efficient LoRA fine-tune + eval. | https://github.com/nikoxxxhr/smolvla-libero-lora |
| zkhong-kkk/smolvla-libero-finetuning | Finetuning experiments. | https://github.com/zkhong-kkk/smolvla-libero-finetuning |
| Brooks207/smolvla-libero-finetune | Official LeRobot train/eval pipeline. | https://github.com/Brooks207/smolvla-libero-finetune |
| alexsuw/smolvla-libero-fewshot | Few-shot style experiments. | https://github.com/alexsuw/smolvla-libero-fewshot |
| akinurfa/smolvla-libero-collector | CLI/web tool to run SmolVLA on LIBERO-10 and collect rollouts. | https://github.com/akinurfa/smolvla-libero-collector |
| rafiqul713/smolvla-libero-onnx | ONNX / inference-oriented. | https://github.com/rafiqul713/smolvla-libero-onnx |

---

## Tier C — egocentric human video → robot imitation

| Repo | Notes | URL |
|------|-------|-----|
| EgoMimic (project) | Egocentric video + 3D hand tracking co-trained with robot data. | https://egomimic.github.io/ · [arXiv](https://arxiv.org/abs/2410.24221) |
| IRMVLab/UniHand | HaMeR hand trajectories from human video → robot EE trajectories. | https://github.com/IRMVLab/UniHand |
| Siyuan-Liu99/somehand | MediaPipe / video → dexterous hand retargeting in MuJoCo. | https://github.com/Siyuan-Liu99/somehand |
| openvla/openvla | Large open VLA (heavier than SmolVLA; good reference). | https://github.com/openvla/openvla |
| octo-models/octo | Smaller generalist policy; diffusion action head. | https://github.com/octo-models/octo |
| real-stanford/diffusion_policy | Strong imitation baseline (not VLA); useful control / chunking ideas. | https://github.com/real-stanford/diffusion_policy |
| danijar/dreamerv3 | Classic latent world-model RL (less manipulation-native, still foundational). | https://github.com/danijar/dreamerv3 |
| worldmodels/worldmodels | Ha & Schmidhuber interactive paper + code lineage. | https://worldmodels.github.io/ |

---

## Suggested “steal from” map for this challenge

```
Phone video
    │
    ├─► HaMeR / MediaPipe / somehand     (extract hand pose)
    │         │
    │         └─► dex-retargeting / EgoVLA-style IK   (human → Panda actions)
    │
    ├─► LeRobot dataset format
    │         │
    │         └─► SmolVLA finetune + LIBERO eval      (policy path)
    │
    └─► action-free video → UWM / iVideoGPT / IRASim  (world-model path)
```

---

## Hugging Face hubs (not GitHub, but load-bearing)

- SmolVLA / LIBERO checkpoints via LeRobot docs: https://huggingface.co/docs/lerobot
- `HuggingFaceVLA/smolvla_libero` (commonly referenced in eval forks)
- Open X-Embodiment datasets (pretraining context)

---

*Survey method: GitHub code search (`SmolVLA LIBERO`, egocentric VLA, retargeting, world models), repo API metadata, and paper project pages. Re-check stars/READMEs before depending on a fork.*
