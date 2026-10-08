"""Lightweight BC baseline on packed NPZ episodes (no LeRobot required).

Trains a tiny CNN+MLP to predict [dx,dy,dz,gripper] from ego frames.
Useful as a 4060 smoke-test and fallback if SmolVLA install fails.
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


class EpisodeDataset(Dataset):
    def __init__(self, episode_dir: Path, splits: set[str], min_vis: float = 0.55):
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        for path in sorted(episode_dir.glob("pick_place_mug_*.npz")):
            ep = np.load(path, allow_pickle=True)
            split = str(ep["split"]) if "split" in ep else "train"
            vis = float(ep["visibility"]) if "visibility" in ep else 1.0
            if split not in splits or vis < min_vis:
                continue
            images = ep["images"]
            actions = ep["action"]
            n = min(len(images), len(actions))
            # Subsample every 2nd frame to keep epoch manageable
            for i in range(0, n, 2):
                self.samples.append((images[i], actions[i]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img, act = self.samples[idx]
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(act).float()
        return x, y


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
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes"))
    parser.add_argument("--out", type=Path, default=Path("outputs/checkpoints/bc_baseline"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-visibility", type=float, default=0.55)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = EpisodeDataset(args.episodes, {"train"}, args.min_visibility)
    val_ds = EpisodeDataset(args.episodes, {"val", "test"}, args.min_visibility)
    if len(train_ds) == 0:
        raise SystemExit("No training frames. Run build_episodes.py first.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TinyPolicy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    args.out.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch} train", leave=False):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(x)
        tr_loss /= len(train_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                va_loss += loss_fn(model(x), y).item() * len(x)
        va_loss /= max(len(val_ds), 1)

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
        print(f"epoch {epoch}: train={tr_loss:.5f} val={va_loss:.5f}")
        torch.save({"model": model.state_dict(), "epoch": epoch}, args.out / "last.pt")

    (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint -> {args.out / 'last.pt'}")


if __name__ == "__main__":
    main()
