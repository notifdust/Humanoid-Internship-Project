"""Overlay wrist trail + gripper state on ego frames for QA / README."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def draw_overlay(frame_bgr: np.ndarray, wrist_xy: tuple[float, float] | None, trail: list, gripper: float) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    if len(trail) >= 2:
        pts = np.array([[int(x * w), int(y * h)] for x, y in trail], dtype=np.int32)
        cv2.polylines(out, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)
    if wrist_xy is not None:
        cx, cy = int(wrist_xy[0] * w), int(wrist_xy[1] * h)
        color = (0, 0, 255) if gripper > 0.5 else (0, 255, 0)
        cv2.circle(out, (cx, cy), 10, color, -1, cv2.LINE_AA)
        cv2.putText(
            out,
            "closed" if gripper > 0.5 else "open",
            (cx + 14, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def process_clip(
    video_path: Path,
    hand_json: Path,
    retarget_json: Path,
    out_path: Path,
    max_trail: int = 40,
) -> None:
    hand = json.loads(hand_json.read_text(encoding="utf-8"))
    ret = json.loads(retarget_json.read_text(encoding="utf-8"))
    gripper = ret["gripper_closed"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    trail: list[tuple[float, float]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fr = hand["frames"][idx] if idx < len(hand["frames"]) else {}
        g = float(gripper[idx]) if idx < len(gripper) else 0.0
        wrist = None
        if fr.get("detected") and fr.get("wrist"):
            wrist = (float(fr["wrist"][0]), float(fr["wrist"][1]))
            trail.append(wrist)
            if len(trail) > max_trail:
                trail.pop(0)
        writer.write(draw_overlay(frame, wrist, trail, g))
        idx += 1

    cap.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("data/processed/ego_1080"))
    parser.add_argument("--hands", type=Path, default=Path("data/processed/hands"))
    parser.add_argument("--retarget", type=Path, default=Path("data/processed/retarget"))
    parser.add_argument("--out", type=Path, default=Path("outputs/overlays"))
    parser.add_argument(
        "--clips",
        type=str,
        default="01,16,34,41",
        help="Comma-separated clip ids (e.g. 01,34) or 'all'",
    )
    args = parser.parse_args()

    if args.clips.strip().lower() == "all":
        ids = [f"{i:02d}" for i in range(1, 43)]
    else:
        ids = [c.strip().zfill(2) for c in args.clips.split(",") if c.strip()]

    for cid in tqdm(ids, desc="overlays"):
        name = f"pick_place_mug_{cid}"
        process_clip(
            args.videos / f"{name}.mp4",
            args.hands / f"{name}.json",
            args.retarget / f"{name}.json",
            args.out / f"{name}_overlay.mp4",
        )
    print(f"Wrote overlays -> {args.out}")


if __name__ == "__main__":
    main()
