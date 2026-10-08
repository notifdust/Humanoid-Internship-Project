# Humanoid Internship Project

Submission workspace for the [Humanoid — Internship, Robot Learning Research](https://www.physicalai.jobs/humanoid/internship-robot-learning-research) challenge.

**Constraint:** use **personally collected** real data (e.g. phone egocentric manipulation video) to drive a simulated manipulator (e.g. LIBERO), creatively using VLA and/or world models.

**Deadline:** Friday, 9 October 2026, 23:59 BST.

## Docs

| Doc | Purpose |
|-----|---------|
| [`PROJECT_OUTLINE.md`](PROJECT_OUTLINE.md) | Project aim, context, success criteria |
| [`resources/`](resources/) | GitHub survey, papers, books, datasets |
| [`LICENSE`](LICENSE) | MIT |

## Status

Research / scaffolding stage. Implementation (data collection → train → sim demo) to follow.

## Quick links from research

- Stack: [LeRobot](https://github.com/huggingface/lerobot) · [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) · [SmolVLA paper](https://arxiv.org/abs/2506.01844)
- Ego-data path: [EgoVLA](https://rchalyang.github.io/EgoVLA/) · [HaMeR](https://github.com/geopavlakos/hamer) · [dex-retargeting](https://github.com/dexsuite/dex-retargeting)
- World models: [UWM](https://weirdlabuw.github.io/uwm/) · [iVideoGPT](https://thuml.github.io/iVideoGPT/)

```powershell
.\scripts\fetch_resources.ps1
```

## License

MIT — see [`LICENSE`](LICENSE). Third-party papers and books remain under their own terms; see `resources/books/README.md`.
