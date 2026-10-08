"""Build a SmolVLA-friendly LeRobot dataset from NPZ episodes.

- Camera key: observation.images.camera1
- State padded to 6D, action padded to 6D (zeros)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def pad_to(x: np.ndarray, dim: int) -> np.ndarray:
    if x.shape[-1] == dim:
        return x.astype(np.float32)
    out = np.zeros((*x.shape[:-1], dim), dtype=np.float32)
    out[..., : x.shape[-1]] = x.astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/lerobot_smolvla_mug"))
    parser.add_argument("--repo-id", type=str, default="local/pick_place_mug_smolvla")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-visibility", type=float, default=0.55)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--action-dim", type=int, default=6)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    files = sorted(args.episodes.glob("pick_place_mug_*.npz"))
    if not files:
        raise SystemExit(f"No NPZ in {args.episodes}")

    sample = np.load(files[0], allow_pickle=True)
    h, w = int(sample["images"].shape[1]), int(sample["images"].shape[2])

    if args.out.exists():
        print(f"Removing {args.out}")
        shutil.rmtree(args.out)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type="panda_retarget_proxy",
        root=args.out,
        use_videos=False,
        features={
            "observation.images.camera1": {
                "dtype": "image",
                "shape": (h, w, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (args.state_dim,),
                "names": [f"s{i}" for i in range(args.state_dim)],
            },
            "action": {
                "dtype": "float32",
                "shape": (args.action_dim,),
                "names": [f"a{i}" for i in range(args.action_dim)],
            },
        },
    )

    kept = 0
    for path in tqdm(files, desc="to lerobot smolvla"):
        ep = np.load(path, allow_pickle=True)
        vis = float(ep["visibility"]) if "visibility" in ep else 1.0
        if vis < args.min_visibility:
            continue
        images = ep["images"]
        actions = pad_to(ep["action"], args.action_dim)
        states = pad_to(ep["state"], args.state_dim)
        instruction = (
            str(ep["instruction"])
            if "instruction" in ep
            else "pick up the fuchsia mug and place it on the white plate"
        )
        n = min(len(images), len(actions), len(states))
        for i in range(n):
            dataset.add_frame(
                {
                    "observation.images.camera1": images[i],
                    "observation.state": states[i],
                    "action": actions[i],
                },
                task=instruction,
            )
        dataset.save_episode()
        kept += 1

    if hasattr(dataset, "finalize"):
        dataset.finalize()

    meta = {"episodes_written": kept, "repo_id": args.repo_id, "root": str(args.out)}
    (args.out / "conversion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"LeRobot SmolVLA dataset ready: {args.out} ({kept} episodes)")


if __name__ == "__main__":
    main()
