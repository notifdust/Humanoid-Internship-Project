"""Qualitative retarget check: wrist (image) path vs integrated EE path + gripper.

Writes PNG panels under outputs/viz/retarget/ for README / QA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_pair(hands_dir: Path, retarget_dir: Path, stem: str) -> tuple[dict, dict]:
    hand_path = hands_dir / f"{stem}.json"
    ret_path = retarget_dir / f"{stem}.json"
    if not hand_path.exists() or not ret_path.exists():
        raise FileNotFoundError(f"Missing {hand_path} or {ret_path}")
    hand = json.loads(hand_path.read_text(encoding="utf-8"))
    ret = json.loads(ret_path.read_text(encoding="utf-8"))
    return hand, ret


def wrist_xy(hand: dict) -> np.ndarray:
    frames = hand["frames"]
    xy = np.full((len(frames), 2), np.nan, dtype=np.float64)
    for i, fr in enumerate(frames):
        if fr.get("detected") and fr.get("wrist"):
            xy[i] = fr["wrist"][:2]
    # simple forward-fill for plot continuity
    for d in range(2):
        col = xy[:, d]
        idx = np.where(np.isfinite(col))[0]
        if len(idx) == 0:
            continue
        first, last = idx[0], idx[-1]
        col[:first] = col[first]
        col[last + 1 :] = col[last]
        bad = ~np.isfinite(col)
        if bad.any():
            good = np.isfinite(col)
            col[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), col[good])
        xy[:, d] = col
    return xy


def plot_clip(hand: dict, ret: dict, out: Path, stem: str) -> None:
    wrist = wrist_xy(hand)
    ee = np.asarray(ret["ee_pos"], dtype=np.float64)
    grip = np.asarray(ret["gripper_closed"], dtype=np.float64)
    t = np.asarray(ret.get("t", np.arange(len(ee))), dtype=np.float64)
    n = min(len(wrist), len(ee), len(grip), len(t))
    wrist, ee, grip, t = wrist[:n], ee[:n], grip[:n], t[:n]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(wrist[:, 0], wrist[:, 1], color="#1f77b4", lw=1.2)
    ax.scatter(wrist[0, 0], wrist[0, 1], c="green", s=28, zorder=3, label="start")
    ax.scatter(wrist[-1, 0], wrist[-1, 1], c="red", s=28, zorder=3, label="end")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # image y down
    ax.set_aspect("equal")
    ax.set_title("Wrist (image xy)")
    ax.set_xlabel("x (norm)")
    ax.set_ylabel("y (norm, down)")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.plot(ee[:, 0], ee[:, 1], color="#ff7f0e", lw=1.2)
    ax.scatter(ee[0, 0], ee[0, 1], c="green", s=28, zorder=3)
    ax.scatter(ee[-1, 0], ee[-1, 1], c="red", s=28, zorder=3)
    ax.set_title("EE proxy (xy)")
    ax.set_xlabel("forward")
    ax.set_ylabel("lateral")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, ee[:, 2], color="#2ca02c", lw=1.2, label="EE z")
    ax2 = ax.twinx()
    ax2.plot(t, grip, color="#d62728", lw=1.0, alpha=0.8, label="gripper")
    ax.set_title("Height + gripper")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("z proxy")
    ax2.set_ylabel("closed")
    ax2.set_ylim(-0.05, 1.05)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)

    vis = hand.get("visibility", ret.get("visibility"))
    fig.suptitle(f"{stem}  visibility={vis}", fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=Path, default=Path("data/processed/hands"))
    parser.add_argument("--retarget", type=Path, default=Path("data/processed/retarget"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/viz/retarget"))
    parser.add_argument(
        "--clips",
        type=str,
        default="pick_place_mug_01,pick_place_mug_30,pick_place_mug_41",
        help="comma-separated stems",
    )
    args = parser.parse_args()

    stems = [s.strip() for s in args.clips.split(",") if s.strip()]
    written = []
    for stem in stems:
        hand, ret = load_pair(args.hands, args.retarget, stem)
        out = args.out_dir / f"{stem}_retarget.png"
        plot_clip(hand, ret, out, stem)
        written.append(str(out))
        print(f"Wrote {out}")

    manifest = {"clips": stems, "files": written}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
