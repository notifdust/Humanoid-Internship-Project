"""Compose ego | overlay | LIBERO frames into one side-by-side MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_frames(path: Path, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise SystemExit(f"No frames in {path}")
    return frames


def resize_h(frame: np.ndarray, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    nw = max(1, int(round(w * (height / h))))
    return cv2.resize(frame, (nw, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ego", type=Path, default=Path("outputs/overlays/pick_place_mug_41.mp4"))
    parser.add_argument("--pred", type=Path, default=Path("outputs/eval/smolvla_rollout_41.mp4"))
    parser.add_argument("--sim", type=Path, default=Path("outputs/eval/libero_openloop_smolvla_41.mp4"))
    parser.add_argument("--out", type=Path, default=Path("outputs/eval/side_by_side_41.mp4"))
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    # Fallbacks if named overlay missing
    if not args.ego.exists():
        cand = sorted(Path("outputs/overlays").glob("*41*.mp4"))
        if not cand:
            cand = sorted(Path("outputs/overlays").glob("*.mp4"))
        if not cand:
            raise SystemExit("No ego overlay video found under outputs/overlays/")
        args.ego = cand[0]
        print(f"Using ego overlay {args.ego}")

    streams = [read_frames(p, args.max_frames) for p in (args.ego, args.pred, args.sim)]
    n = min(len(s) for s in streams)
    labels = ["ego + wrist", "SmolVLA pred", "LIBERO sim"]

    panels = []
    for i in range(n):
        cols = []
        for fr, lab in zip([s[i] for s in streams], labels, strict=True):
            fr = resize_h(fr, args.height)
            cv2.putText(fr, lab, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(fr, lab, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            cols.append(fr)
        # pad widths to max in row
        mh = args.height
        mw = max(c.shape[1] for c in cols)
        padded = []
        for c in cols:
            if c.shape[1] < mw:
                pad = np.zeros((mh, mw - c.shape[1], 3), dtype=np.uint8)
                c = np.concatenate([c, pad], axis=1)
            padded.append(c)
        panels.append(np.concatenate(padded, axis=1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    h, w = panels[0].shape[:2]
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for p in panels:
        writer.write(p)
    writer.release()
    print(f"Wrote {args.out} frames={n} size={w}x{h}")


if __name__ == "__main__":
    main()
