"""Run BC baseline on a held-out ego clip and write predicted-action overlay video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=Path("data/processed/episodes/pick_place_mug_41.npz"))
    parser.add_argument("--ckpt", type=Path, default=Path("outputs/checkpoints/bc_baseline/last.pt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/eval/bc_rollout_41.mp4"))
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ep = np.load(args.episode, allow_pickle=True)
    images = ep["images"]
    gt = ep["action"]

    model = TinyPolicy().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (args.image_size, args.image_size))

    preds = []
    with torch.no_grad():
        for i in range(len(images)):
            x = torch.from_numpy(images[i]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            x = x.to(device)
            pred = model(x).cpu().numpy()[0]
            preds.append(pred)
            frame = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
            # draw GT vs pred gripper as bars
            g_gt = float(gt[i, 3]) if i < len(gt) else 0.0
            g_pr = float(pred[3])
            cv2.rectangle(frame, (10, 10), (10 + int(80 * g_gt), 24), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, 30), (10 + int(80 * np.clip(g_pr, 0, 1)), 44), (0, 128, 255), -1)
            cv2.putText(frame, f"dx={pred[0]:+.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            writer.write(frame)
    writer.release()

    preds_a = np.asarray(preds)
    n = min(len(preds_a), len(gt))
    mse = float(np.mean((preds_a[:n] - gt[:n]) ** 2))
    meta = {"episode": str(args.episode), "mse": mse, "frames": n}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} mse={mse:.6f}")


if __name__ == "__main__":
    main()
