"""Render GT vs predicted future-frame strips / GIF for a held-out episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from src.wm.train_future_frame import TinyActionWM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=Path("data/processed/episodes/pick_place_mug_41.npz"))
    parser.add_argument("--ckpt", type=Path, default=Path("outputs/wm/future_frame/last.pt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/wm/future_frame_strip_41.gif"))
    parser.add_argument("--steps", type=int, default=8, help="number of prediction panels")
    parser.add_argument("--stride", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    horizon = int(blob.get("horizon", 5))
    model = TinyActionWM().to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    ep = np.load(args.episode, allow_pickle=True)
    images = ep["images"]
    actions = ep["action"].astype(np.float32)

    frames_out = []
    with torch.no_grad():
        for i in range(args.steps):
            t = i * args.stride
            if t + horizon >= len(images):
                break
            a = cv2.resize(images[t], (64, 64))
            gt = cv2.resize(images[t + horizon], (64, 64))
            act = actions[t : t + horizon].mean(axis=0)
            x = torch.from_numpy(a).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            u = torch.from_numpy(act).float().unsqueeze(0).to(device)
            pred = (model(x, u)[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            # scale up for readability
            scale = 3
            a_s = cv2.resize(a, (64 * scale, 64 * scale), interpolation=cv2.INTER_NEAREST)
            p_s = cv2.resize(pred, (64 * scale, 64 * scale), interpolation=cv2.INTER_NEAREST)
            g_s = cv2.resize(gt, (64 * scale, 64 * scale), interpolation=cv2.INTER_NEAREST)
            strip = np.concatenate([a_s, p_s, g_s], axis=1)
            strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
            cv2.putText(strip_bgr, "t", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(strip_bgr, "pred t+h", (64 * scale + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(strip_bgr, "gt t+h", (128 * scale + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frames_out.append(cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGB))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames_out, duration=0.6, loop=0)
    # also write an mp4 strip for players that dislike gif
    mp4 = args.out.with_suffix(".mp4")
    h, w = frames_out[0].shape[:2]
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (w, h))
    for fr in frames_out:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Wrote {args.out} and {mp4} panels={len(frames_out)} horizon={horizon}")


if __name__ == "__main__":
    main()
