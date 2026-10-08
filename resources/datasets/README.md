# Datasets & benchmarks

Pointers only — download weights/data separately; do not commit large binaries.

| Name | Type | Use in this project | Link |
|------|------|---------------------|------|
| **LIBERO** demos | Sim manipulation demos | Eval / optional BC pretrain | https://github.com/Lifelong-Robot-Learning/LIBERO |
| **LeRobot** community datasets | HF datasets | Format target for your phone demos | https://huggingface.co/lerobot |
| **Open X-Embodiment** | Cross-robot trajs | Context for VLA pretraining (usually too big to fine-tune fully) | https://robotics-transformer-x.github.io/ |
| **Your phone egocentric set** | Self-collected | **Required** by the challenge | Store under `data/` (gitignored) |

## Minimal personal dataset checklist

- [ ] Short egocentric clips of one simple task (e.g. pick-and-place a cup)
- [ ] Consistent camera (phone, chest/head height if possible)
- [ ] Language label per episode (“put the red cup on the plate”)
- [ ] Enough takes for a toy fine-tune (often 10–50 short demos is a starting point)
- [ ] Document capture setup in the submission README
