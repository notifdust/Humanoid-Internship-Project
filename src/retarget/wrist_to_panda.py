"""EgoVLA-inspired wrist trajectory → Panda-like EE deltas + gripper heuristic.

Maps normalized image wrist (x,y) and a crude depth proxy from hand scale
into delta end-effector actions suitable for later LeRobot / LIBERO packing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def hand_scale(landmarks: list[list[float]] | None) -> float:
    if not landmarks or len(landmarks) < 10:
        return float("nan")
    w = np.array(landmarks[0][:2])
    m = np.array(landmarks[9][:2])  # middle MCP
    return float(np.linalg.norm(m - w))


def pinch_distance(frame: dict) -> float:
    if not frame.get("thumb_tip") or not frame.get("index_tip"):
        return float("nan")
    a = np.array(frame["thumb_tip"][:2])
    b = np.array(frame["index_tip"][:2])
    return float(np.linalg.norm(a - b))


def interpolate_missing(xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = xy.copy()
    idx = np.arange(len(xy))
    for d in range(xy.shape[1]):
        good = valid & np.isfinite(xy[:, d])
        if good.sum() < 2:
            continue
        out[:, d] = np.interp(idx, idx[good], xy[good, d])
    return out


def retarget_clip(hand_json: Path) -> dict:
    with hand_json.open(encoding="utf-8") as f:
        data = json.load(f)

    frames = data["frames"]
    n = len(frames)
    wrist = np.full((n, 2), np.nan, dtype=np.float64)
    scale = np.full(n, np.nan, dtype=np.float64)
    pinch = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    for i, fr in enumerate(frames):
        if fr.get("detected") and fr.get("wrist"):
            wrist[i] = fr["wrist"][:2]
            scale[i] = hand_scale(fr.get("landmarks"))
            pinch[i] = pinch_distance(fr)
            valid[i] = True

    wrist = interpolate_missing(wrist, valid)
    # Depth proxy: larger hand scale ≈ closer to camera ≈ higher z in ego frame.
    # Invert & normalize for a crude lift signal.
    scale_filled = interpolate_missing(scale.reshape(-1, 1), np.isfinite(scale)).ravel()
    scale_n = (scale_filled - np.nanmean(scale_filled)) / (np.nanstd(scale_filled) + 1e-6)

    # Image coords: x right, y down. Robot table: x forward-ish, y left-right.
    # Map: robot_y ∝ +(0.5 - wrist_x), robot_x ∝ +(0.5 - wrist_y)  [ego looking down]
    pos = np.zeros((n, 3), dtype=np.float64)
    pos[:, 0] = (0.5 - wrist[:, 1]) * 2.0   # "forward"
    pos[:, 1] = (0.5 - wrist[:, 0]) * 2.0   # "lateral"
    pos[:, 2] = -scale_n * 0.15             # height proxy

    # Delta actions (finite difference), clipped for stability
    delta = np.zeros_like(pos)
    delta[1:] = pos[1:] - pos[:-1]
    delta = np.clip(delta, -0.05, 0.05)

    # Gripper: closed when pinch small relative to median pinch
    pinch_f = interpolate_missing(pinch.reshape(-1, 1), np.isfinite(pinch)).ravel()
    med = np.nanmedian(pinch_f[np.isfinite(pinch_f)]) if np.isfinite(pinch_f).any() else 0.05
    gripper = (pinch_f < 0.65 * med).astype(np.float64)  # 1 = closed

    t = np.array([fr["t"] for fr in frames], dtype=np.float64)
    return {
        "filename": data["filename"],
        "visibility": data.get("visibility"),
        "qa_pass": data.get("qa_pass"),
        "fps": data.get("fps"),
        "t": t.tolist(),
        "ee_pos": pos.tolist(),
        "ee_delta": delta.tolist(),
        "gripper_closed": gripper.tolist(),
        "instruction": "pick up the fuchsia mug and place it on the white plate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/processed/hands"))
    parser.add_argument("--dst", type=Path, default=Path("data/processed/retarget"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    files = sorted(args.src.glob("pick_place_mug_*.json"))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No hand JSON in {args.src}")

    args.dst.mkdir(parents=True, exist_ok=True)
    for path in tqdm(files, desc="retarget"):
        out = args.dst / path.name
        payload = retarget_clip(path)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
    print(f"Wrote {len(files)} retarget files -> {args.dst}")


if __name__ == "__main__":
    main()
