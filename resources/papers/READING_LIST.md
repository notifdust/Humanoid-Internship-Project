# Paper reading list

PDFs (when downloaded) live in [`pdf/`](pdf/). Re-fetch with `scripts/fetch_resources.ps1`.

Priority is **challenge relevance**, not citation count.

---

## P0 — directly on-mission

| Paper | Year | Why read | arXiv / PDF |
|-------|------|----------|-------------|
| **SmolVLA** — A VLA for affordable & efficient robotics | 2025 | Model named in the Humanoid posting; Colab/single-GPU friendly. | [2506.01844](https://arxiv.org/abs/2506.01844) · `pdf/smolvla_2506.01844.pdf` |
| **LIBERO** — Benchmarking knowledge transfer for lifelong robot learning | 2023 | Target sim environment (Panda manipulation suites). | [2312.08467](https://arxiv.org/abs/2312.08467) · `pdf/libero_2312.08467.pdf` |
| **EgoVLA** — Learning VLAs from egocentric human videos | 2025 | Human ego video → wrist/hand actions → IK/retarget → robot policy. | [2507.12440](https://arxiv.org/abs/2507.12440) · `pdf/egovla_2507.12440.pdf` |
| **EgoMimic** — Scaling imitation via egocentric video | 2024 | Treats human hand data as first-class demos with aligned action spaces. | [2410.24221](https://arxiv.org/abs/2410.24221) · `pdf/egomimic_2410.24221.pdf` |
| **HaMeR** — Reconstructing hands in 3D with transformers | 2024 | Practical monocular hand mesh for phone footage. | [2312.05251](https://arxiv.org/abs/2312.05251) · `pdf/hamer_2312.05251.pdf` |

---

## P1 — VLA / policy foundations

| Paper | Year | Why read | arXiv / PDF |
|-------|------|----------|-------------|
| **OpenVLA** | 2024 | Open 7B VLA; fine-tuning / LoRA patterns. | [2406.09246](https://arxiv.org/abs/2406.09246) · `pdf/openvla_2406.09246.pdf` |
| **Octo** | 2024 | Compact generalist; diffusion action head. | [2405.12213](https://arxiv.org/abs/2405.12213) · `pdf/octo_2405.12213.pdf` |
| **π₀** | 2024 | Flow-matching VLA; inspiration for SmolVLA’s action expert. | [2410.24164](https://arxiv.org/abs/2410.24164) · `pdf/pi0_2410.24164.pdf` |
| **RT-2** | 2023 | Canonical “actions as tokens” VLA. | [2307.15818](https://arxiv.org/abs/2307.15818) · `pdf/rt2_2307.15818.pdf` |
| **Diffusion Policy** | 2023 | Strong visuomotor IL baseline; action chunking. | [2303.04137](https://arxiv.org/abs/2303.04137) · `pdf/diffusion_policy_2303.04137.pdf` |

---

## P1 — world models (challenge suggestion #4)

| Paper | Year | Why read | arXiv / PDF |
|-------|------|----------|-------------|
| **IRASim** — Fine-grained WM for robot manipulation | 2024/25 | Action-conditioned long-horizon video; policy evaluation proxy. | [2406.14540](https://arxiv.org/abs/2406.14540) · `pdf/irasim_2406.14540.pdf` |
| **iVideoGPT** — Interactive VideoGPTs as scalable WMs | 2024 | Pretrained on human + robot trajs; planning / MBRL. | [2405.15223](https://arxiv.org/abs/2405.15223) · `pdf/ivideogpt_2405.15223.pdf` |
| **UWM** — Unified world models (video + action diffusion) | 2025 | Explicitly uses **action-free video** + robot data; LIBERO results. | [2504.02792](https://arxiv.org/abs/2504.02792) · `pdf/uwm_2504.02792.pdf` |
| **FLIP** — Flow-centric generative planning WM | 2024 | Flow + video generation for language-conditioned planning. | [2412.08261](https://arxiv.org/abs/2412.08261) · `pdf/flip_2412.08261.pdf` |
| **World Models** (Ha & Schmidhuber) | 2018 | Classic “learn in the dream” framing. | [1803.10122](https://arxiv.org/abs/1803.10122) · `pdf/world_models_ha_1803.10122.pdf` |
| **DreamerV3** | 2023 | General latent WM RL with fixed hyperparameters. | [2301.04104](https://arxiv.org/abs/2301.04104) · `pdf/dreamerv3_2301.04104.pdf` |

---

## P2 — optional depth

| Paper | Year | Why read | arXiv / PDF |
|-------|------|----------|-------------|
| **GWM** — Gaussian world models for robotic manipulation | 2025 | 3DGS world model; action-conditioned prediction + MBRL. | [2508.17600](https://arxiv.org/abs/2508.17600) · `pdf/gwm_2508.17600.pdf` |

- Open X-Embodiment / RT-X papers — data scale context for VLAs.
- AnyTeleop / DexMV — retargeting + human demo processing (paired with `dex-retargeting` code).

---

## How to use this list for the challenge

1. Skim **SmolVLA + LIBERO** (stack).
2. Deep-read **EgoVLA or EgoMimic** (personal ego data story).
3. Pick **one** of: HaMeR retarget path **or** UWM/iVideoGPT/IRASim world-model path **or** RL fine-tune after a bootstrap policy.
4. Cite what you borrow in the submission README.
