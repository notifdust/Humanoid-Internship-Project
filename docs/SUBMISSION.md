# Submission checklist (Humanoid challenge)

**Deadline:** Friday, 16 October 2026, 23:59 BST  
**Apply:** https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221  
**Public repo:** https://github.com/notifdust/Humanoid-Internship-Project

## Paste into the form

**GitHub URL**

```
https://github.com/notifdust/Humanoid-Internship-Project
```

**One-paragraph summary** (optional / cover note)

> I filmed 42 egocentric pick-and-place clips of a fuchsia mug, extracted MediaPipe hand tracks, retargeted wrist motion to Panda EE deltas (EgoVLA-inspired), finetuned SmolVLA on that personal dataset, and drove LIBERO / MuJoCo with the resulting actions. I also trained a tiny action-conditioned future-frame world-model teaser on the same ego clips. The README documents what worked (personal-data finetune beats pretrained on held-out action MSE; open-loop sim videos) and what did not (closed-loop domain transfer fails; open-loop LIBERO task success is 0/5 because the scene is not LIBERO-identical).

## Before you click submit

1. [ ] Repo is **public**
2. [ ] README opens cleanly on GitHub (no broken relative links to local-only videos — paths are documented)
3. [ ] CV uploaded in Ashby form
4. [ ] You watched `outputs/eval/side_by_side_41.mp4` and `outputs/wm/future_frame_strip_41.gif` locally once (gitignored large media)

## What judges should see in the README

- Personal data → pipeline → sim (constraint satisfied)
- VLA path (SmolVLA) **and** WM teaser
- Honest failure modes (closed-loop / open-loop SR)
- How to reproduce on RTX 4060 + WSL
