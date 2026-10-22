"""Measure real CUDA training memory; do not infer it from parameter count."""

import argparse

from _common import PROJECT_ROOT, read_config
import torch
from torch.utils.data import DataLoader, TensorDataset

from cog_grouping.backbones import make_backbone
from cog_grouping.engine import train_epoch
from cog_grouping.grouping import GroupingConfig
from cog_grouping.model import build_classifier
from cog_grouping.provenance import atomic_json, environment, seed_everything


def main():
    """Profile a complete optimizer step including frozen-forward and adapter backward."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "runs" / "cuda_memory.json"))
    parser.add_argument("--random", action="store_true", help="same architecture, random weights; memory only")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        atomic_json(args.output, {"status": "unavailable", "reason": "CUDA is not available", "measured": False,
                                  "environment": environment()})
        raise SystemExit("CUDA is not available; no 24GB measurement has been made")
    config = read_config(args.config)
    seed_everything(config.get("seed", 42))
    device = torch.device("cuda")
    backbone = make_backbone(config["backbone"], not args.random, None if args.random else config.get("weights_path"))
    model = build_classifier(backbone, 100, config, PROJECT_ROOT).to(device)
    batch = config["batch_size"]
    images = torch.randn(batch * 3, 3, config.get("image_size", 224), config.get("image_size", 224))
    labels = torch.arange(batch * 3) % 100
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    precision = config.get("precision", "bf16")
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    train_epoch(model, loader, optimizer, device, precision, scaler=scaler)
    torch.cuda.synchronize()
    stats = train_epoch(model, loader, optimizer, device, precision, scaler=scaler)
    torch.cuda.synchronize()
    stats.update({"status": "measured", "measured": True, "microbatch": batch, "precision": precision,
                  "random_weights": args.random, "parameters": model.trainable_parameter_counts(),
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                  "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
                  "under_24_gib_allocated": stats["peak_cuda_allocated_bytes"] < 24 * 1024 ** 3,
                  "environment": environment(), "classification_efficacy_established": False})
    atomic_json(args.output, stats)
    print(stats)


if __name__ == "__main__":
    main()
