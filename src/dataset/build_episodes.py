"""Pack ego frames + retargeted actions into per-episode NPZ files.

Format (one file per clip):
  images: uint8 [T,H,W,3] RGB resized
  action: float32 [T,4]  = [dx, dy, dz, gripper_closed]
  state:  float32 [T,3]  = cumulative EE position proxy
  timestamps: float32 [T]
  instruction: str
  visibility: float
  qa_pass: bool
  split: str
  source_video: str

This is the intermediate dataset consumed by `to_lerobot.py` and training stubs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_labels(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["filename"]: r for r in rows}


def pack_episode(
    video_path: Path,
    retarget_path: Path,
    out_path: Path,
    meta: dict,
    image_size: int = 256,
) -> dict:
    ret = json.loads(retarget_path.read_text(encoding="utf-8"))
    deltas = np.asarray(ret["ee_delta"], dtype=np.float32)
    gripper = np.asarray(ret["gripper_closed"], dtype=np.float32).reshape(-1, 1)
    pos = np.asarray(ret["ee_pos"], dtype=np.float32)
    t = np.asarray(ret["t"], dtype=np.float32)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    cap.release()

    n = min(len(frames), len(deltas), len(gripper), len(pos), len(t))
    images = np.stack(frames[:n], axis=0).astype(np.uint8)
    action = np.concatenate([deltas[:n], gripper[:n]], axis=1).astype(np.float32)
    state = pos[:n].astype(np.float32)
    timestamps = t[:n]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        images=images,
        action=action,
        state=state,
        timestamps=timestamps,
        instruction=np.asarray(ret.get("instruction", meta.get("instruction", ""))),
        visibility=np.asarray(float(ret.get("visibility") or meta.get("visibility") or 0.0)),
        qa_pass=np.asarray(bool(ret.get("qa_pass") or meta.get("qa_pass") == "True" or meta.get("qa_pass") == "1" or meta.get("qa_pass") is True)),
        split=np.asarray(meta.get("split", "train")),
        source_video=np.asarray(video_path.name),
    )
    return {
        "episode": out_path.name,
        "frames": int(n),
        "split": meta.get("split", "train"),
        "visibility": float(ret.get("visibility") or 0.0),
        "qa_pass": bool(ret.get("qa_pass")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("data/processed/ego_1080"))
    parser.add_argument("--retarget", type=Path, default=Path("data/processed/retarget"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/episodes"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--min-visibility", type=float, default=0.0, help="Skip clips below this")
    args = parser.parse_args()

    labels = load_labels(args.labels) if args.labels.exists() else {}
    videos = sorted(args.videos.glob("pick_place_mug_*.mp4"))
    report = []

    for vp in tqdm(videos, desc="pack episodes"):
        rp = args.retarget / (vp.stem + ".json")
        if not rp.exists():
            continue
        meta = labels.get(vp.name, {"split": "train", "instruction": "pick up the fuchsia mug and place it on the white plate"})
        ret = json.loads(rp.read_text(encoding="utf-8"))
        vis = float(ret.get("visibility") or 0.0)
        if vis < args.min_visibility:
            continue
        out = args.out / (vp.stem + ".npz")
        report.append(pack_episode(vp, rp, out, meta, args.image_size))

    args.out.mkdir(parents=True, exist_ok=True)
    summary = args.out / "manifest.json"
    summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Packed {len(report)} episodes -> {args.out}")
    print(f"Manifest: {summary}")


if __name__ == "__main__":
    main()
