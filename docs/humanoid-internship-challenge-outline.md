# Humanoid Internship Challenge — Project Outline

**Source:** [Internship, Robot Learning Research at Humanoid](https://www.physicalai.jobs/humanoid/internship-robot-learning-research)  
**Submission deadline:** Friday, 9 October 2026, 23:59 BST  
**Deliverable:** Public GitHub repository (README/presentation, run instructions, example outputs, design notes)

---

## 1. Context

### Who Humanoid is

Humanoid is building commercially scalable, safe humanoid robots, with **HMND‑01** as their platform already being deployed in real industrial environments. Their stated mission is to build software systems that let robots operate effectively in the real world—expanding human capability and changing how physical work gets done.

### What the internship is about

The **Robot Learning Research** internship (London, full-time, 12–24 weeks) sits across their research stack:

| Area | Focus |
|------|--------|
| **Reinforcement learning** | Language–vision conditioned manipulation policies; sim task suites (Isaac Sim, MuJoCo); sim-to-real |
| **World models** | Action-conditioned video/dynamics prediction; using WMs as learned simulators; fidelity metrics |
| **Pre- / post-training** | In-context learning, memory, post-training VLA models on production use cases, closing the human↔robot embodiment gap |
| **Inference & optimisation** | Real-time edge inference (profiling, quantisation, latency); distributed training/data loading |

The challenge is the filter for that internship: it is designed to surface how applicants **think** about robot learning—not whether they can reproduce a standard tutorial. They explicitly say they are looking for creative, original submissions that push beyond the obvious, and that a “standard” task is now easy to hand off to AI agents.

---

## 2. The project (challenge statement)

### Hard constraint (non-negotiable)

> Use **real data you personally collect** to drive a robotic manipulator in a **simple simulation environment** (e.g. **LIBERO**).

Practically that means:

1. **Collect your own data** — typically egocentric hand-manipulation video from a phone (or similar).
2. **Use that data in the loop** — it must play a meaningful role in the approach (not just be decorative).
3. **Drive a simulated robot arm** — e.g. a Franka Panda in LIBERO, with a policy (example reference: **SmolVLA**-style VLA) and/or world-model machinery.

Their illustrative picture: left = egocentric hand video; right = LIBERO/Panda driven by a policy trained/adapted from that data.

### Soft / suggested directions (pick or invent)

They list possible development paths, without requiring any single one:

- **Post-train a policy** on a simple task using your recorded egocentric data  
- **Creative retargeting** — map human hand data onto challenging robot embodiments  
- **Bootstrap + RL** — start from the data-driven policy, then improve with RL  
- **World modelling focus** — video/state prediction quality over policy score  
- **Inference optimisation** — make a standard policy run much faster than a baseline  

Anything goes as long as the personal-data constraint holds. They also note many ideas fit on **Google Colab GPUs** and that hand-coded approaches are valid.

---

## 3. Ultimate aim

### What “success” means for this challenge

The ultimate aim is **not** “highest LIBERO score at all costs.” It is to demonstrate:

1. **You can close a human-data → robot-sim loop** — personally collected egocentric (or related) data becomes a real signal that affects how a simulated manipulator behaves or how a world model predicts.
2. **You understand modern robot learning levers** — at least one of: VLA post-training, retargeting/embodiment transfer, RL fine-tuning, world models, or edge/inference optimisation.
3. **Your thinking is clear and original** — creative framing, simple implementation, honest reporting of what worked and what did not (“without AI slop”).

### What they say they care about (evaluation axes)

| Criterion | Implication for design |
|-----------|------------------------|
| **Creativity + personal-data constraint** | Data must be load-bearing; novelty in how it is used matters |
| **Policy performance in sim and/or WM prediction quality** | Pick a measurable outcome and show it |
| **Implementation simplicity & clear presentation** | Prefer a clean, reproducible repo over a sprawling system |

### How this maps to the company’s longer-term aim

Humanoid’s research agenda is about robots that work in the **real world**: policies conditioned on language and vision, models that predict physically consistent futures, post-training for production tasks, and models that fit on robot hardware. This challenge is a **miniature of that stack**: take messy human-centric data, turn it into something a robot embodiment can use, and show that you can evaluate and explain the result.

---

## 4. Scope boundaries (for planning)

| In scope | Out of scope (for the challenge itself) |
|----------|-------------------------------------------|
| Phone-scale personal dataset | Large public datasets as the *only* training signal |
| LIBERO (or similar simple manipulator sim) | Full humanoid hardware deployment |
| VLA / WM / RL / retargeting / inference ideas | Matching every internship research pillar at once |
| Colab- or modest-GPU feasible work | Requiring multi-node clusters |
| Public GitHub + README/presentation | Formal paper-quality writeup |

---

## 5. Required submission shape

A **public GitHub repository** that includes:

- **README / presentation** — how to run the system  
- **Example outputs** — videos, plots, qualitative WM predictions, etc.  
- **Design notes** — choices made, what worked, what failed  
- Ability to paste the repo URL into their application form with name + CV  

**Timeline note:** submission window closes **9 October 2026, 23:59 BST** (two-week challenge window from their posting).

---

## 6. Open design decisions (to settle next)

These are deliberately left open until an approach is chosen:

1. **Task** — which simple manipulation skill (e.g. pick-and-place, push, open) and how it is defined in LIBERO  
2. **Data format** — raw video only vs. extracted poses / keypoints / language labels  
3. **Primary lever** — VLA post-train vs. retargeting vs. RL vs. world model vs. speed  
4. **Baseline** — what “before personal data” looks like, so the contribution of *your* data is visible  
5. **Metrics** — success rate, prediction error, latency, or a combination  

---

## 7. One-sentence summary

**Collect your own egocentric manipulation data, use it as a real part of training or modelling a policy/world-model that drives a simulated robot arm (e.g. LIBERO/Panda), and present a creative, simple, reproducible result that shows how you think about robot learning.**
