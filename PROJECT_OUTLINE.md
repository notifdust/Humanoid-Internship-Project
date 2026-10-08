# Project Outline — Humanoid Internship Challenge

## Context

This repository is the submission for the **Internship, Robot Learning Research** challenge at [Humanoid](https://www.physicalai.jobs/humanoid/internship-robot-learning-research) (London).

Humanoid builds commercially scalable humanoid robots (platform **HMND‑01**) for real industrial environments. Their research stack spans:

| Area | Focus |
|------|--------|
| **Reinforcement learning** | Language–vision conditioned manipulation; sim (Isaac Sim, MuJoCo) → real |
| **World models** | Action-conditioned video/dynamics prediction; learned simulators; fidelity metrics |
| **Pre- / post-training** | VLA models, memory, embodiment gap, data diversity |
| **Inference & optimisation** | Edge deployment, quantisation, latency, distributed training |

The challenge is the filter for that internship: a public GitHub repo demonstrating how you think about **robot learning with personally collected data**, not a polished production system.

**Deadline:** Friday, 9 October 2026, 23:59 BST  
**Format:** Public GitHub repository + README/presentation (run instructions, example outputs, design choices, what worked / what didn’t)

---

## The challenge (constraint)

**Core requirement:** Use **real data you personally collect** (e.g. phone egocentric hand-manipulation video) to **drive a robotic manipulator in a simple simulation** (e.g. LIBERO), creatively applying **VLA and/or world models**.

Reference framing from the posting:

- **Left:** egocentric hand-manipulation video  
- **Right:** policy driving a Panda arm in LIBERO (example path: post-train something like SmolVLA)

Suggested directions (non-exhaustive):

1. Post-train a policy on a simple task from your recorded egocentric data  
2. Creative **retargeting** (human hands → robot embodiment)  
3. Bootstrap a policy, then improve it with **RL**  
4. Emphasise **world modelling** (video/state prediction) over policy score  
5. **Optimise** a baseline policy for much faster inference  

Anything goes **as long as your own data plays a real role**. Hand-coded pipelines are fine; large compute is not required (Colab-scale is enough). Creativity and clear thinking matter more than a “standard” solution.

---

## Ultimate aim

Deliver a **reproducible, personally-data-driven demo** that shows:

1. **You collected** a small real manipulation dataset (phone / egocentric).  
2. That data **meaningfully shapes** the learning or modelling pipeline (not a decorative afterthought).  
3. The system **acts or predicts** in simulation (policy control and/or world-model rollouts).  
4. The write-up explains **design choices**, results, and failures with clarity — no filler.

Success for the application is judged on:

- Creativity under the personal-data constraint  
- Policy performance in sim **and/or** world-model prediction quality  
- Simple implementation and clear presentation  

For the internship itself, the broader aim is contributing to systems that let humanoids operate reliably in the real world — this challenge is a miniature of that loop: **real observation → learned model/policy → simulated (then eventually physical) control**.

---

## What “done” looks like for this repo

| Deliverable | Purpose |
|-------------|---------|
| Collected dataset (or clear pointer + sample) | Satisfies the personal-data constraint |
| Pipeline (VLA post-train / WM / RL / retarget / optimise — pick a focus) | Technical substance |
| Sim demo (e.g. LIBERO + manipulator) | Shows data → robot behaviour |
| README with run steps, outputs, design notes | What reviewers actually evaluate |

---

## Open decisions (next)

Not fixed yet — to decide before building:

- **Primary angle:** policy post-training vs world model vs retargeting vs RL fine-tune vs inference speed  
- **Task:** which simple manipulation to film and reproduce in sim  
- **Stack:** LIBERO (or alternative), base model (e.g. SmolVLA), training environment  
- **Data format:** how egocentric video becomes training signal (actions, goals, latents, etc.)

---

## Research library (this repo)

| Path | Contents |
|------|----------|
| [`resources/github/SIMILAR_PROJECTS.md`](resources/github/SIMILAR_PROJECTS.md) | Closest public GitHub implementations |
| [`resources/papers/`](resources/papers/) | Reading list, BibTeX, arXiv PDFs |
| [`resources/books/`](resources/books/) | Free textbooks & courses |
| [`resources/datasets/`](resources/datasets/) | Benchmarks + personal-data checklist |

## Sources

- [Internship, Robot Learning Research — Physical AI Jobs](https://www.physicalai.jobs/humanoid/internship-robot-learning-research)
