"""Run open-loop LIBERO rollouts over several seeds; write success summary JSON.

WSL/Linux. Uses GT or predicted actions (.npy / .npz). Does not claim closed-loop SR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.rollout_libero_openloop import load_actions, to_libero_action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/eval/libero_openloop_multiseed.json"))
    parser.add_argument("--task-suite", type=str, default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--max-steps", type=int, default=0, help="0 = all actions")
    args = parser.parse_args()

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    actions = load_actions(args.actions)
    if args.max_steps > 0:
        actions = actions[: args.max_steps]

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    language = suite.get_task(args.task_id).language
    bddl_file = suite.get_task_bddl_file_path(args.task_id)

    results = []
    for seed in seeds:
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=128,
            camera_widths=128,
        )
        env.seed(seed)
        env.reset()
        success = False
        steps = 0
        for act in actions:
            obs, reward, done, info = env.step(to_libero_action(act, args.scale))
            steps += 1
            if done or bool(info.get("success", False)) or reward > 0:
                success = bool(info.get("success", reward > 0))
                break
        env.close()
        results.append({"seed": seed, "success": success, "steps": steps})
        print(f"seed={seed} success={success} steps={steps}", flush=True)

    n_ok = sum(1 for r in results if r["success"])
    payload = {
        "actions": str(args.actions),
        "suite": args.task_suite,
        "task_id": args.task_id,
        "language": language,
        "scale": args.scale,
        "n_success": n_ok,
        "n_episodes": len(results),
        "success_rate": n_ok / max(len(results), 1),
        "results": results,
        "note": "Open-loop action replay; not closed-loop vision policy SR.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} success_rate={payload['success_rate']:.2f}")


if __name__ == "__main__":
    main()
