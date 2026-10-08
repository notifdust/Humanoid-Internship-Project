"""Train a tiny action-conditioned next-frame predictor on ego NPZ episodes.

WM teaser inspired by IRASim / Ha–Schmidhuber world-model framing — not a full UWM.
Predicts frame t+k from frame t and mean action over the horizon.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class EpisodeFrameDataset(Dataset):
    def __init__(self, episodes: Path, horizon: int = 5, image_size: int = 64, min_vis: float = 0.55):
        self.horizon = horizon
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        files = sorted(episodes.glob("pick_place_mug_*.npz"))
        for path in files:
            ep = np.load(path, allow_pickle=True)
            vis = float(ep["visibility"]) if "visibility" in ep else 1.0
            if vis < min_vis:
                continue
            images = ep["images"]
            actions = ep["action"].astype(np.float32)
            n = min(len(images), len(actions))
            for t in range(0, n - horizon):
                img_t = images[t]
                img_f = images[t + horizon]
                act = actions[t : t + horizon].mean(axis=0)
                self.samples.append((img_t, img_f, act))
        if not self.samples:
            raise SystemExit(f"No samples under {episodes}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        a, b, act = self.samples[idx]
        # downsample for tiny model
        import cv2

        a64 = cv2.resize(a, (64, 64), interpolation=cv2.INTER_AREA)
        b64 = cv2.resize(b, (64, 64), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(a64).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(b64).permute(2, 0, 1).float() / 255.0
        u = torch.from_numpy(act).float()
        return x, y, u


class TinyActionWM(nn.Module):
    """Encode image, fuse action, decode next frame."""

    def __init__(self, action_dim: int = 4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 8
            nn.ReLU(inplace=True),
        )
        self.act_proj = nn.Linear(action_dim, 128)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 64
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        a = self.act_proj(act).view(-1, 128, 1, 1)
        h = h + a
        return self.dec(h)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes"))
    parser.add_argument("--out", type=Path, default=Path("outputs/wm/future_frame"))
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = EpisodeFrameDataset(args.episodes, horizon=args.horizon)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = TinyActionWM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    print(f"device={device} samples={len(ds)} horizon={args.horizon}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, y, u in tqdm(loader, desc=f"wm ep{epoch}"):
            x, y, u = x.to(device), y.to(device), u.to(device)
            pred = model(x, u)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * x.size(0)
            n += x.size(0)
        avg = total / max(n, 1)
        history.append({"epoch": epoch, "l1": avg})
        print(f"epoch={epoch} l1={avg:.5f}", flush=True)

    ckpt = args.out / "last.pt"
    torch.save({"model": model.state_dict(), "horizon": args.horizon, "history": history}, ckpt)
    (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Wrote {ckpt}")


if __name__ == "__main__":
    main()
