# Free / open learning resources

This folder is for **legally free** textbooks and open courses useful for the challenge.
Large PDFs are **gitignored** (see root `.gitignore`); download locally with `scripts/fetch_resources.ps1`.

---

## Core free books

| Resource | License / access | Why it helps | Get it |
|----------|------------------|--------------|--------|
| **Reinforcement Learning: An Introduction** (Sutton & Barto, 2nd ed.) | Free official PDF (authors / MIT Press open access) | RL foundations if you bootstrap → improve with RL. | http://incompleteideas.net/book/the-book-2nd.html · local: `SuttonBarto_RL_2ndEd.pdf` |
| **Modern Robotics: Mechanics, Planning, and Control** (Lynch & Park) | Free **preprint** for personal use only — **do not redistribute** | Kinematics, Jacobians, IK — needed for hand→Panda retargeting. | http://modernrobotics.org/ · preprint PDF linked from that page |
| **Dive into Deep Learning** (Zhang et al.) | Open textbook (Apache-2.0 content) | PyTorch-oriented DL refresh. | https://d2l.ai/ |
| **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong) | Free PDF from authors | Linear algebra / probability refresher. | https://mml-book.github.io/ |

---

## Open courses (free to audit)

| Course | Relevance | URL |
|--------|-----------|-----|
| Coursera — Modern Robotics specialization (Lynch) | Matches the free textbook; IK & planning. | https://www.coursera.org/specializations/modernrobotics |
| Hugging Face — LeRobot docs / tutorials | SmolVLA, datasets, LIBERO eval in practice. | https://huggingface.co/docs/lerobot |
| Berkeley CS 285 / Deep RL (Levine) lectures | Policy gradients, IL, model-based RL. | https://rail.eecs.berkeley.edu/deeprlcourse/ |
| World Models interactive article | Classic WM intuition. | https://worldmodels.github.io/ |

---

## Notes on redistribution

- **Sutton & Barto**: keep the official PDF for personal study; respect the authors’ stated license (typically CC BY-NC-ND — no commercial reuse, no derivatives).
- **Modern Robotics preprint**: authors forbid further distribution — **link only**, do not commit the PDF to a public repo.
- Prefer citing publishers for anything not clearly open.

---

## Suggested reading order for this project

1. Modern Robotics ch. on forward/inverse kinematics (retargeting).
2. Sutton & Barto ch. 1–6 if doing RL fine-tuning; ch. on planning if doing model-based ideas.
3. D2L chapters on CNNs / Transformers as needed for VLA internals.
4. Papers in `../papers/READING_LIST.md` (P0 first).
