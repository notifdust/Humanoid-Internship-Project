"""Run finetuned SmolVLA on a held-out ego episode; write overlay + action MSE."""

from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

# Windows: NamedTemporaryFile cannot be re-opened by path while the handle is open.
_orig_ntf = tempfile.NamedTemporaryFile


def _ntf_windows_safe(*args, **kwargs):
    kwargs["delete"] = False
    return _orig_ntf(*args, **kwargs)


tempfile.NamedTemporaryFile = _ntf_windows_safe  # type: ignore


def pad_to(x: np.ndarray, dim: int) -> np.ndarray:
    if x.shape[-1] == dim:
        return x.astype(np.float32)
    out = np.zeros((*x.shape[:-1], dim), dtype=np.float32)
    out[..., : x.shape[-1]] = x.astype(np.float32)
    return out


def load_policy(ckpt: Path, device: torch.device, dataset_root: Path, repo_id: str):
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

    # Meta only (for stats / feature shapes); no need to stream the full set.
    ds = LeRobotDataset(repo_id, root=str(dataset_root))
    policy = make_policy(cfg, ds_meta=ds.meta)
    policy.eval()
    policy.to(device)
    policy.reset()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=Path("data/processed/episodes/pick_place_mug_41.npz"))
    parser.add_argument("--ckpt", type=Path, default=Path("outputs/train/smolvla_custom/checkpoint-2000"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/lerobot_smolvla_mug"))
    parser.add_argument("--repo-id", type=str, default="local/pick_place_mug_smolvla")
    parser.add_argument("--out", type=Path, default=Path("outputs/eval/smolvla_rollout_41.mp4"))
    parser.add_argument("--task", type=str, default="pick up the fuchsia mug and place it on the white plate")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} ckpt={args.ckpt}")

    ep = np.load(args.episode, allow_pickle=True)
    images = ep["images"]
    gt = pad_to(ep["action"], 6)
    states = pad_to(ep["state"], 6)
    n = len(images) if args.max_frames <= 0 else min(len(images), args.max_frames)

    policy = load_policy(args.ckpt, device, args.dataset_root, args.repo_id)

    h, w = int(images[0].shape[0]), int(images[0].shape[1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))

    preds = []
    with torch.no_grad():
        for i in range(n):
            img = torch.from_numpy(images[i]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            st = torch.from_numpy(states[i]).float().unsqueeze(0)
            batch = {
                "observation.images.camera1": img.to(device),
                "observation.state": st.to(device),
                "task": [args.task],
            }
            action = policy.select_action(batch).detach().cpu().numpy()[0]
            preds.append(action)

            frame = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
            g_gt = float(gt[i, 3]) if i < len(gt) else 0.0
            g_pr = float(np.clip(action[3], 0, 1))
            cv2.rectangle(frame, (10, 10), (10 + int(80 * g_gt), 24), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, 30), (10 + int(80 * g_pr), 44), (0, 128, 255), -1)
            cv2.putText(
                frame,
                f"dx={action[0]:+.3f} dy={action[1]:+.3f} dz={action[2]:+.3f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            writer.write(frame)
            if (i + 1) % 50 == 0:
                print(f"frame {i + 1}/{n}", flush=True)

    writer.release()
    preds_a = np.asarray(preds, dtype=np.float32)
    actions_path = args.out.with_name(args.out.stem + "_actions.npy")
    np.save(actions_path, preds_a)
    m = min(len(preds_a), len(gt))
    mse = float(np.mean((preds_a[:m] - gt[:m]) ** 2))
    mse_xyz = float(np.mean((preds_a[:m, :3] - gt[:m, :3]) ** 2))
    meta = {
        "episode": str(args.episode),
        "ckpt": str(args.ckpt),
        "mse": mse,
        "mse_xyz": mse_xyz,
        "frames": m,
        "actions_npy": str(actions_path),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} mse={mse:.6f} mse_xyz={mse_xyz:.6f} actions={actions_path}")


if __name__ == "__main__":
    main()
