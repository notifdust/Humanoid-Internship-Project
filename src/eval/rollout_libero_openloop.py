"""Open-loop LIBERO rollout driven by retargeted / SmolVLA EE deltas.

Intended to run inside WSL after `scripts/wsl_install_libero.sh`.
Maps [dx, dy, dz, gripper, ...] -> robosuite OSC_POSE-style 7D actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_actions(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        return np.asarray(np.load(path, allow_pickle=True)["action"], dtype=np.float64)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64)
    raise SystemExit(f"Unsupported actions: {path}")


def to_libero_action(act: np.ndarray, scale: float) -> np.ndarray:
    """Build 7D [dx,dy,dz,0,0,0,gripper] with gripper in {-1 open, +1 close}."""
    out = np.zeros(7, dtype=np.float32)
    out[0] = float(np.clip(act[0] * scale, -1, 1))
    out[1] = float(np.clip(act[1] * scale, -1, 1))
    out[2] = float(np.clip(act[2] * scale, -1, 1))
    grip = float(act[3]) if act.shape[0] > 3 else 0.0
    out[6] = 1.0 if grip > 0.5 else -1.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/eval/libero_openloop.mp4"))
    parser.add_argument("--task-suite", type=str, default="libero_object")
    parser.add_argument("--task-id", type=int, default=0, help="index within suite")
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", type=str, default="agentview_image")
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    actions = load_actions(args.actions)
    bench_dict = benchmark.get_benchmark_dict()
    if args.task_suite not in bench_dict:
        raise SystemExit(f"Unknown suite {args.task_suite}; have {list(bench_dict)}")
    suite = bench_dict[args.task_suite]()
    task = suite.get_task(args.task_id)
    bddl = task.language  # human-readable
    bddl_file = suite.get_task_bddl_file_path(args.task_id)

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(args.seed)
    obs = env.reset()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    h, w = 256, 256
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))

    successes = 0
    for i, act in enumerate(actions):
        a = to_libero_action(act, args.scale)
        obs, reward, done, info = env.step(a)
        frame = obs.get(args.camera)
        if frame is None:
            # try common keys
            for k in ("agentview_image", "robot0_eye_in_hand_image"):
                if k in obs:
                    frame = obs[k]
                    break
        if frame is None:
            raise SystemExit(f"No camera in obs keys={list(obs.keys())}")
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, f"{i}/{len(actions)} {bddl[:40]}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        writer.write(bgr)
        if done:
            successes += int(bool(info.get("success", reward > 0)))
            break

    writer.release()
    env.close()
    meta = {
        "actions": str(args.actions),
        "suite": args.task_suite,
        "task_id": args.task_id,
        "language": bddl,
        "bddl_file": str(bddl_file),
        "frames": i + 1,
        "libero_path": get_libero_path("bddl_files"),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} task={bddl!r}")


if __name__ == "__main__":
    main()
