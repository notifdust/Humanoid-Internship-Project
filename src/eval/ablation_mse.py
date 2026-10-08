"""Compare held-out action MSE: BC vs pretrained SmolVLA vs finetuned SmolVLA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class TinyPolicy(nn.Module):
    def __init__(self, action_dim: int = 4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x))


def pad_to(x: np.ndarray, dim: int) -> np.ndarray:
    if x.shape[-1] == dim:
        return x.astype(np.float32)
    out = np.zeros((*x.shape[:-1], dim), dtype=np.float32)
    out[..., : x.shape[-1]] = x.astype(np.float32)
    return out


def mse_bc(episode: Path, ckpt: Path, device: torch.device) -> dict:
    ep = np.load(episode, allow_pickle=True)
    images, gt = ep["images"], pad_to(ep["action"], 4)
    model = TinyPolicy().to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(images)):
            x = torch.from_numpy(images[i]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            preds.append(model(x).cpu().numpy()[0])
    p = np.asarray(preds, dtype=np.float32)
    n = min(len(p), len(gt))
    return {
        "name": "bc_baseline",
        "mse": float(np.mean((p[:n] - gt[:n]) ** 2)),
        "mse_xyz": float(np.mean((p[:n, :3] - gt[:n, :3]) ** 2)),
        "frames": n,
    }


def mse_smolvla(episode: Path, ckpt: Path, dataset_root: Path, repo_id: str, device: torch.device, name: str) -> dict:
    from src.eval.rollout_smolvla_overlay import load_policy, pad_to as pad6

    ep = np.load(episode, allow_pickle=True)
    images = ep["images"]
    gt = pad6(ep["action"], 6)
    states = pad6(ep["state"], 6)
    policy = load_policy(ckpt, device, dataset_root, repo_id)
    task = "pick up the fuchsia mug and place it on the white plate"
    preds = []
    with torch.no_grad():
        for i in range(len(images)):
            batch = {
                "observation.images.camera1": (
                    torch.from_numpy(images[i]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                ),
                "observation.state": torch.from_numpy(states[i]).float().unsqueeze(0).to(device),
                "task": [task],
            }
            preds.append(policy.select_action(batch).detach().cpu().numpy()[0])
    p = np.asarray(preds, dtype=np.float32)
    n = min(len(p), len(gt))
    return {
        "name": name,
        "ckpt": str(ckpt),
        "mse": float(np.mean((p[:n] - gt[:n]) ** 2)),
        "mse_xyz": float(np.mean((p[:n, :3] - gt[:n, :3]) ** 2)),
        "frames": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=Path("data/processed/episodes/pick_place_mug_41.npz"))
    parser.add_argument("--bc-ckpt", type=Path, default=Path("outputs/checkpoints/bc_baseline/last.pt"))
    parser.add_argument("--pretrained", type=Path, default=Path("models/smolvla_base"))
    parser.add_argument("--finetuned", type=Path, default=Path("outputs/train/smolvla_custom/checkpoint-2000"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/lerobot_smolvla_mug"))
    parser.add_argument("--repo-id", type=str, default="local/pick_place_mug_smolvla")
    parser.add_argument("--out", type=Path, default=Path("outputs/eval/ablation_mse.json"))
    parser.add_argument("--skip-pretrained", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [mse_bc(args.episode, args.bc_ckpt, device)]
    if not args.skip_pretrained:
        rows.append(
            mse_smolvla(
                args.episode, args.pretrained, args.dataset_root, args.repo_id, device, "smolvla_pretrained"
            )
        )
    rows.append(
        mse_smolvla(args.episode, args.finetuned, args.dataset_root, args.repo_id, device, "smolvla_finetuned")
    )

    payload = {"episode": str(args.episode), "results": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for r in rows:
        print(f"{r['name']}: mse={r['mse']:.6f} mse_xyz={r['mse_xyz']:.6f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
