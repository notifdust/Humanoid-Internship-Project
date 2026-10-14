"""Build a short submission demo reel from existing ego / policy / sim / WM clips.

Sections (title card + clip):
  1. Ego + wrist overlay
  2. SmolVLA predicted actions on ego
  3. Side-by-side (ego | pred | LIBERO)
  4. LIBERO open-loop from personal actions
  5. MuJoCo EE proxy
  6. World-model future-frame strip
  7. Closing results card
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


W, H = 1280, 720
FPS = 20.0


def title_card(lines: list[str], frames: int = 40, bg: tuple[int, int, int] = (18, 18, 22)) -> list[np.ndarray]:
    out = []
    for _ in range(frames):
        img = np.full((H, W, 3), bg, dtype=np.uint8)
        y0 = H // 2 - 28 * len(lines)
        for i, line in enumerate(lines):
            scale = 1.15 if i == 0 else 0.75
            thickness = 2 if i == 0 else 1
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            x = max(20, (W - tw) // 2)
            y = y0 + i * 48
            cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), thickness, cv2.LINE_AA)
        out.append(img)
    return out


def letterbox(frame: np.ndarray, width: int = W, height: int = H) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), (12, 12, 14), dtype=np.uint8)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def read_clip(path: Path, max_frames: int = 0, every_n: int = 1) -> list[np.ndarray]:
    if not path.exists():
        raise SystemExit(f"Missing clip: {path}")
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % every_n == 0:
            frames.append(letterbox(fr))
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    if not frames:
        raise SystemExit(f"No frames in {path}")
    return frames


def results_card(ablation_path: Path, multiseed_path: Path, frames: int = 70) -> list[np.ndarray]:
    lines = [
        "Results (honest)",
        "Personal data -> SmolVLA finetune -> sim",
    ]
    if ablation_path.exists():
        abl = json.loads(ablation_path.read_text(encoding="utf-8"))
        by = {r["name"]: r for r in abl.get("results", [])}
        if "smolvla_pretrained" in by and "smolvla_finetuned" in by:
            lines.append(
                f"MSE  pretrained {by['smolvla_pretrained']['mse']:.3f}  ->  "
                f"finetuned {by['smolvla_finetuned']['mse']:.3f}"
            )
        if "bc_baseline" in by:
            lines.append(f"BC baseline MSE {by['bc_baseline']['mse']:.3f} (proxy control)")
    if multiseed_path.exists():
        ms = json.loads(multiseed_path.read_text(encoding="utf-8"))
        lines.append(
            f"LIBERO open-loop multi-seed  {ms.get('n_success', 0)}/{ms.get('n_episodes', 0)}  "
            "(scene mismatch; expected)"
        )
    lines.append("Closed-loop ego-ft: domain transfer fail (documented)")
    lines.append("github.com/notifdust/Humanoid-Internship-Project")
    return title_card(lines, frames=frames, bg=(16, 22, 28))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/demo/challenge_demo.mp4"))
    parser.add_argument("--max-section-frames", type=int, default=120)
    args = parser.parse_args()

    sections: list[tuple[list[str], Path | None, dict]] = [
        (["Humanoid challenge demo", "Ego phone data -> VLA -> LIBERO + WM"], None, {}),
        (
            ["1 / Ego + MediaPipe wrist overlay", "Personal fuchsia-mug pick/place"],
            Path("outputs/overlays/pick_place_mug_41_overlay.mp4"),
            {"every_n": 2},
        ),
        (
            ["2 / SmolVLA on held-out ego clip", "Predicted EE deltas overlaid"],
            Path("outputs/eval/smolvla_rollout_41.mp4"),
            {"every_n": 1},
        ),
        (
            ["3 / Side-by-side", "ego | SmolVLA pred | LIBERO open-loop"],
            Path("outputs/eval/side_by_side_41.mp4"),
            {"every_n": 1},
        ),
        (
            ["4 / LIBERO open-loop", "Personal-data actions drive Panda"],
            Path("outputs/eval/libero_openloop_smolvla_41.mp4"),
            {"every_n": 1},
        ),
        (
            ["5 / MuJoCo EE proxy", "Integrated deltas in a simple table scene"],
            Path("outputs/eval/mujoco_ee_smolvla_41.mp4"),
            {"every_n": 1},
        ),
        (
            ["6 / World-model teaser", "Action-conditioned future frame (t | pred | gt)"],
            Path("outputs/wm/future_frame_strip_41.mp4"),
            {"every_n": 1, "max_frames": 24},
        ),
    ]

    reel: list[np.ndarray] = []
    for title, path, opts in sections:
        reel.extend(title_card(title, frames=36 if path else 50))
        if path is None:
            continue
        max_f = int(opts.get("max_frames", args.max_section_frames))
        every_n = int(opts.get("every_n", 1))
        reel.extend(read_clip(path, max_frames=max_f, every_n=every_n))

    reel.extend(
        results_card(
            Path("outputs/eval/ablation_mse.json"),
            Path("outputs/eval/libero_openloop_multiseed.json"),
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for fr in reel:
        writer.write(fr)
    writer.release()

    meta = {
        "out": str(args.out),
        "frames": len(reel),
        "fps": FPS,
        "duration_sec": round(len(reel) / FPS, 2),
        "size": [W, H],
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} frames={len(reel)} duration={meta['duration_sec']}s")


if __name__ == "__main__":
    main()
