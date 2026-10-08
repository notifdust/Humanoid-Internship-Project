"""Fine-tune SmolVLA without lerobot-train CLI (avoids Windows NamedTemporaryFile bug)."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Windows: NamedTemporaryFile cannot be re-opened by path while the handle is open.
_orig_ntf = tempfile.NamedTemporaryFile


def _ntf_windows_safe(*args, **kwargs):
    kwargs["delete"] = False
    return _orig_ntf(*args, **kwargs)


tempfile.NamedTemporaryFile = _ntf_windows_safe  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/lerobot_smolvla_mug"))
    parser.add_argument("--repo-id", type=str, default="local/pick_place_mug_smolvla")
    parser.add_argument("--pretrained", type=str, default="models/smolvla_base")
    parser.add_argument("--out", type=Path, default=Path("outputs/train/smolvla_custom"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--save-freq", type=int, default=500)
    args = parser.parse_args()

    import dataclasses

    from lerobot.configs.types import NormalizationMode
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} pretrained={args.pretrained}")

    # Avoid PreTrainedConfig.from_pretrained (draccus + tempfile issues on Windows).
    raw = json.loads((Path(args.pretrained) / "config.json").read_text(encoding="utf-8"))
    raw.pop("type", None)
    if "normalization_mapping" in raw:
        raw["normalization_mapping"] = {
            k: NormalizationMode(v) if not isinstance(v, NormalizationMode) else v
            for k, v in raw["normalization_mapping"].items()
        }
    allowed = {f.name for f in dataclasses.fields(SmolVLAConfig)}
    cfg = SmolVLAConfig(**{k: v for k, v in raw.items() if k in allowed})
    cfg.pretrained_path = args.pretrained
    cfg.device = str(device)
    cfg.empty_cameras = 2
    cfg.push_to_hub = False

    # Action chunk temporal context for the dataset sampler.
    delta_timestamps = {
        "action": [i / 30.0 for i in range(cfg.chunk_size)],
    }

    dataset = LeRobotDataset(
        args.repo_id,
        root=str(args.dataset_root),
        delta_timestamps=delta_timestamps,
    )
    policy = make_policy(cfg, ds_meta=dataset.meta)
    policy.train()

    frozen = 0
    for name, p in policy.named_parameters():
        if any(k in name.lower() for k in ("vision_tower", "vision_model", "visual")):
            p.requires_grad = False
            frozen += p.numel()
    print(f"froze ~{frozen} vision params")

    trainable = [p for p in policy.parameters() if p.requires_grad]
    if not trainable:
        for p in policy.parameters():
            p.requires_grad = True
        trainable = list(policy.parameters())
    print(f"trainable tensors: {sum(p.requires_grad for p in policy.parameters())}")

    opt = torch.optim.AdamW(trainable, lr=args.lr)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    it = iter(loader)

    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    log_path = args.out / "train.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    for step in tqdm(range(1, args.steps + 1), desc="smolvla"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        # Ensure task / language key exists
        if "task" not in batch and "task_index" in batch:
            batch["task"] = ["pick up the fuchsia mug and place it on the white plate"] * args.batch_size

        loss, _out = policy.forward(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % args.log_freq == 0 or step == 1:
            rec = {"step": step, "loss": float(loss.detach().cpu())}
            history.append(rec)
            log(json.dumps(rec))

        if step % args.save_freq == 0 or step == args.steps:
            ckpt_dir = args.out / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            log(f"saved {ckpt_dir}")

    log(f"Done. Checkpoints in {args.out}")


if __name__ == "__main__":
    main()
