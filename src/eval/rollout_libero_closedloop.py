"""Closed-loop LIBERO rollout (WSL / Linux).

Two modes (see docs/RESEARCH_CLOSEDLOOP.md):

- ``ego-ft``: our ego-finetuned SmolVLA (camera1 / 6D) on LIBERO images — domain transfer.
- ``libero-ref``: HuggingFaceVLA/smolvla_libero with canonical 180-rotate + 8D state.

Requires LIBERO + robosuite + mujoco (pinned) and, for ego-ft, a Windows-trained
checkpoint readable from ``/mnt/c/...`` plus lerobot/torch in the WSL venv.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

_orig_ntf = tempfile.NamedTemporaryFile


def _ntf_windows_safe(*args, **kwargs):
    kwargs["delete"] = False
    return _orig_ntf(*args, **kwargs)


tempfile.NamedTemporaryFile = _ntf_windows_safe  # type: ignore


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = float(np.clip(q[3], -1.0, 1.0))
    den = math.sqrt(max(1.0 - q[3] * q[3], 0.0))
    if den < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def prep_image_180(img: np.ndarray) -> np.ndarray:
    """LIBERO OpenGL origin vs SmolVLA training: 180 rotate."""
    return np.ascontiguousarray(img[::-1, ::-1])


def libero_state_8(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )


def to_chw01(img_hwc_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(img_hwc_u8.astype(np.float32) / 255.0).permute(2, 0, 1)
    return t.unsqueeze(0).to(device)


def action_6_to_7(act6: np.ndarray, scale: float) -> np.ndarray:
    out = np.zeros(7, dtype=np.float32)
    out[0] = float(np.clip(act6[0] * scale, -1, 1))
    out[1] = float(np.clip(act6[1] * scale, -1, 1))
    out[2] = float(np.clip(act6[2] * scale, -1, 1))
    grip = float(act6[3]) if act6.shape[0] > 3 else 0.0
    out[6] = 1.0 if grip > 0.5 else -1.0
    return out


def load_ego_policy(ckpt: Path, dataset_root: Path, repo_id: str, device: torch.device):
    """Load ego-finetuned ckpt; prefer from_pretrained, else make_policy + dataset meta."""
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(str(ckpt))
        policy.config.device = str(device)
        if hasattr(policy.config, "empty_cameras"):
            policy.config.empty_cameras = 2
        policy.config.n_action_steps = 1
        policy.to(device)
        policy.eval()
        policy.reset()
        print("loaded via SmolVLAPolicy.from_pretrained")
        return policy
    except Exception as e:
        print(f"from_pretrained failed ({type(e).__name__}: {e}); falling back to make_policy")

    from lerobot.configs.types import NormalizationMode
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    raw = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    raw.pop("type", None)
    if "normalization_mapping" in raw:
        raw["normalization_mapping"] = {
            k: NormalizationMode(v) if not isinstance(v, NormalizationMode) else v
            for k, v in raw["normalization_mapping"].items()
        }
    allowed = {f.name for f in dataclasses.fields(SmolVLAConfig)}
    cfg = SmolVLAConfig(**{k: v for k, v in raw.items() if k in allowed})
    cfg.pretrained_path = str(ckpt)
    cfg.device = str(device)
    cfg.empty_cameras = 2
    cfg.push_to_hub = False
    cfg.n_action_steps = 1
    ds = LeRobotDataset(repo_id, root=str(dataset_root))
    policy = make_policy(cfg, ds_meta=ds.meta)
    policy.eval()
    policy.to(device)
    policy.reset()
    return policy


def load_libero_ref_policy(model_id: str, device: torch.device):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.config.n_action_steps = 1
    if hasattr(policy.config, "num_steps"):
        policy.config.num_steps = 1
    policy.to(device)
    policy.eval()
    policy.reset()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ego-ft", "libero-ref"), default="ego-ft")
    parser.add_argument("--ckpt", type=Path, default=Path("outputs/train/smolvla_custom/checkpoint-2000"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/lerobot_smolvla_mug"))
    parser.add_argument("--repo-id", type=str, default="local/pick_place_mug_smolvla")
    parser.add_argument("--ref-model", type=str, default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--out", type=Path, default=Path("outputs/eval/libero_closedloop_ego_ft.mp4"))
    parser.add_argument("--task-suite", type=str, default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--scale", type=float, default=5.0, help="ego-ft delta scale into LIBERO")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-text", type=str, default="", help="override language; default = BDDL lang")
    args = parser.parse_args()

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"mode={args.mode} device={device}")

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    language = args.task_text or suite.get_task(args.task_id).language
    bddl_file = suite.get_task_bddl_file_path(args.task_id)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(args.seed)
    obs = env.reset()

    if args.mode == "ego-ft":
        policy = load_ego_policy(args.ckpt, args.dataset_root, args.repo_id, device)
        # Prefer mug instruction for our finetune; still show suite language in overlay.
        task_for_policy = args.task_text or "pick up the fuchsia mug and place it on the white plate"
    else:
        policy = load_libero_ref_policy(args.ref_model, device)
        task_for_policy = language

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (256, 256))

    success = False
    steps = 0
    with torch.no_grad():
        for t in range(args.max_steps):
            agent = prep_image_180(obs["agentview_image"])
            wrist = prep_image_180(obs["robot0_eye_in_hand_image"])
            st8 = libero_state_8(obs)

            if args.mode == "ego-ft":
                batch = {
                    "observation.images.camera1": to_chw01(agent, device),
                    "observation.state": torch.from_numpy(st8[:6]).float().unsqueeze(0).to(device),
                    "task": [task_for_policy],
                }
                act = policy.select_action(batch).detach().cpu().numpy()[0]
                a7 = action_6_to_7(act, args.scale)
            else:
                batch = {
                    "observation.images.image": to_chw01(agent, device),
                    "observation.images.image2": to_chw01(wrist, device),
                    "observation.state": torch.from_numpy(st8).float().unsqueeze(0).to(device),
                    "task": [task_for_policy],
                }
                act = policy.select_action(batch).detach().cpu().numpy()[0]
                a7 = np.clip(act[:7].astype(np.float32), -1.0, 1.0)

            obs, reward, done, info = env.step(a7)
            frame = obs["agentview_image"]
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr,
                f"{args.mode} t={t} {language[:36]}",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )
            writer.write(bgr)
            steps = t + 1
            if done or bool(info.get("success", False)) or reward > 0:
                success = bool(info.get("success", reward > 0))
                break

    writer.release()
    env.close()
    meta = {
        "mode": args.mode,
        "suite": args.task_suite,
        "task_id": args.task_id,
        "language": language,
        "task_for_policy": task_for_policy,
        "steps": steps,
        "success": success,
        "ckpt": str(args.ckpt) if args.mode == "ego-ft" else args.ref_model,
        "note": "ego-ft is domain transfer (ego camera1/6D → LIBERO); see docs/RESEARCH_CLOSEDLOOP.md",
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} success={success} steps={steps}")


if __name__ == "__main__":
    main()
