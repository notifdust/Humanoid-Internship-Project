"""Convert packed NPZ episodes into a local LeRobot dataset (when lerobot is installed)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/lerobot_pick_place_mug"))
    parser.add_argument("--repo-id", type=str, default="local/pick_place_mug_ego")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-visibility", type=float, default=0.55)
    args = parser.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        raise SystemExit(
            "lerobot is not installed. Run: pip install lerobot\n"
            f"Original error: {e}"
        ) from e

    files = sorted(args.episodes.glob("pick_place_mug_*.npz"))
    if not files:
        raise SystemExit(f"No NPZ episodes in {args.episodes}. Run build_episodes.py first.")

    # Peek one episode for shapes
    sample = np.load(files[0], allow_pickle=True)
    h, w = int(sample["images"].shape[1]), int(sample["images"].shape[2])
    action_dim = int(sample["action"].shape[1])
    state_dim = int(sample["state"].shape[1])

    if args.out.exists():
        import shutil

        print(f"Removing existing dataset at {args.out}")
        shutil.rmtree(args.out)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type="panda_retarget_proxy",
        root=args.out,
        # Images (not video): avoids a pyav/AVRational fps bug on Windows in lerobot 0.3.x
        use_videos=False,
        features={
            "observation.images.ego": {
                "dtype": "image",
                "shape": (h, w, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["x", "y", "z"][:state_dim],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["dx", "dy", "dz", "gripper"][:action_dim],
            },
        },
    )

    kept = 0
    for path in tqdm(files, desc="to lerobot"):
        ep = np.load(path, allow_pickle=True)
        vis = float(ep["visibility"]) if "visibility" in ep else 1.0
        if vis < args.min_visibility:
            continue
        images = ep["images"]
        actions = ep["action"]
        states = ep["state"]
        instruction = str(ep["instruction"]) if "instruction" in ep else "pick up the fuchsia mug and place it on the white plate"
        n = min(len(images), len(actions), len(states))
        for i in range(n):
            dataset.add_frame(
                {
                    "observation.images.ego": images[i],
                    "observation.state": states[i].astype(np.float32),
                    "action": actions[i].astype(np.float32),
                },
                task=instruction,
            )
        dataset.save_episode()
        kept += 1

    if hasattr(dataset, "finalize"):
        dataset.finalize()

    meta = {"episodes_written": kept, "repo_id": args.repo_id, "root": str(args.out)}
    (args.out / "conversion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"LeRobot dataset ready: {args.out} ({kept} episodes)")


if __name__ == "__main__":
    main()
