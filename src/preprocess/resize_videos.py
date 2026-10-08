"""Downsample ego 4K clips to 1080p @ 30fps for the 4060 pipeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm


def resize_video(
    src: Path,
    dst: Path,
    width: int = 1920,
    height: int = 1080,
    fps_out: float = 30.0,
) -> dict:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {src}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps_out)
    n_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), fourcc, fps_out, (width, height))

    written = 0
    # Simple frame walk; 4K→1080 is the main win for later stages.
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    return {
        "filename": src.name,
        "src_fps": round(src_fps, 3),
        "frames_in": n_in,
        "frames_out": written,
        "width": width,
        "height": height,
        "fps_out": fps_out,
        "out_path": str(dst),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/raw/ego"),
        help="Folder with pick_place_mug_*.mp4",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/processed/ego_1080"),
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--limit", type=int, default=0, help="If >0, only first N clips")
    args = parser.parse_args()

    clips = sorted(args.src.glob("pick_place_mug_*.mp4"))
    if args.limit > 0:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"No clips found in {args.src}")

    rows = []
    for src in tqdm(clips, desc="resize→1080p"):
        dst = args.dst / src.name
        if dst.exists() and dst.stat().st_size > 0:
            rows.append({"filename": src.name, "skipped": 1, "out_path": str(dst)})
            continue
        rows.append(resize_video(src, dst, args.width, args.height))

    report = args.dst / "resize_report.csv"
    args.dst.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with report.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} videos -> {args.dst}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
