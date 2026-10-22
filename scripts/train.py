"""Train the proposed adapter and matched controls without opening final-test data."""

import argparse
from dataclasses import asdict
import random

from _common import PROJECT_ROOT, read_config
import numpy as np
import torch
from torch.utils.data import DataLoader

from cog_grouping.backbones import make_backbone
from cog_grouping.data import make_datasets
from cog_grouping.engine import epoch_learning_rate, evaluate, precision_context, train_epoch
from cog_grouping.grouping import GroupingConfig
from cog_grouping.model import build_classifier
from cog_grouping.provenance import atomic_json, atomic_torch_save, environment, seed_everything, seed_worker, source_fingerprint, state_fingerprint


def rng_state(generator):
    """Encode all RNG states using tensors and primitive types for safe loading."""
    numpy_state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "loader": generator.get_state(), "python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    """Restore epoch-boundary reproducibility, including data-loader shuffling."""
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def main():
    """Save resumable state and select the best checkpoint using validation only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs-this-invocation", type=int)
    parser.add_argument("--allow-random-backbone", action="store_true")
    args = parser.parse_args()
    if args.epochs_this_invocation is not None and args.epochs_this_invocation < 1:
        parser.error("--epochs-this-invocation must be positive")
    config = read_config(args.config)
    for name in ("mode", "seed", "device", "precision"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if not config.get("pretrained", True) and not args.allow_random_backbone:
        parser.error("random backbones are smoke tests only; explicitly pass --allow-random-backbone")
    device_name = config.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    precision = config.get("precision", "bf16" if device.type == "cuda" else "fp32")
    precision_context(device, precision)
    config["device"], config["precision"] = str(device), precision
    torch.set_num_threads(config.get("threads", 4))
    seed = config.get("seed", 42)
    seed_everything(seed)
    from pathlib import Path
    output = Path(args.output)
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite a run; use a new output or --resume")
    if args.resume and not (output / "last.pt").is_file():
        raise FileNotFoundError("--resume requires output/last.pt")
    if args.resume and (output / "test_accessed.json").exists():
        raise ValueError("training cannot resume after final-test access")
    output.mkdir(parents=True, exist_ok=True)
    backbone = make_backbone(config["backbone"], config.get("pretrained", True), config.get("weights_path"))
    backbone_hash = state_fingerprint(backbone)
    training, validation, _, classes, split = make_datasets(config, backbone.data_config, include_test=False)
    grouping_config = GroupingConfig(**config.get("grouping", {}))
    model = build_classifier(backbone, classes, config, PROJECT_ROOT).to(device)
    frozen_start = model.frozen_fingerprint()
    generator = torch.Generator().manual_seed(seed)
    options = {"batch_size": config["batch_size"], "num_workers": config.get("workers", 0),
               "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker}
    train_loader = DataLoader(training, shuffle=True, generator=generator, **options)
    val_loader = DataLoader(validation, shuffle=False, **options)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.get("learning_rate", 1e-3), weight_decay=config.get("weight_decay", 0.01))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    source = source_fingerprint(PROJECT_ROOT)
    manifest = {"config": config, "grouping_config": asdict(grouping_config), "num_classes": classes,
                "head_initialization_seed": seed + 1000003,
                "normalization": backbone.data_config, "weights": backbone.weights_info,
                "backbone_sha256": backbone_hash, "parameters": model.trainable_parameter_counts(),
                "frozen_model_sha256": frozen_start,
                "split_sha256": split["sha256"], "source": source, "environment": environment(),
                "test_accessed": False,
                "protocol": ("full distributed official ParaX; harmonized pilot" if config.get("mode") == "official_parax"
                             else "post-backbone single-module pilot; not published-score reproduction")}
    first_epoch, best_accuracy, history = 0, -1.0, []
    if args.resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        for key in ("config", "backbone_sha256", "frozen_model_sha256", "split_sha256", "source"):
            if saved["manifest"][key] != manifest[key]:
                raise ValueError("resume provenance mismatch: " + key)
        model.load_trainable_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        first_epoch, best_accuracy = saved["epoch"] + 1, saved["best_accuracy"]
        history = saved["history"]
        restore_rng(saved["rng"], generator)
    else:
        seed_everything(seed + 2000003)
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "data_roles.json", split)
    last_epoch = config["epochs"] if args.epochs_this_invocation is None else min(config["epochs"], first_epoch + args.epochs_this_invocation)
    for epoch in range(first_epoch, last_epoch):
        lr = epoch_learning_rate(config.get("learning_rate", 1e-3), epoch,
                                  config["epochs"], config.get("warmup_epochs", 0))
        for group in optimizer.param_groups:
            group["lr"] = lr
        train_result = train_epoch(model, train_loader, optimizer, device, precision,
                                    config.get("accumulation_steps", 1), scaler,
                                    config.get("max_grad_norm", 1.0))
        val_result = evaluate(model, val_loader, device, precision)
        entry = {"epoch": epoch, "learning_rate": lr, "train": train_result, "validation": val_result}
        history.append(entry)
        improved = val_result["accuracy"] > best_accuracy
        best_accuracy = max(best_accuracy, val_result["accuracy"])
        saved = {"epoch": epoch, "best_accuracy": best_accuracy, "model": model.trainable_state_dict(),
                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "rng": rng_state(generator), "manifest": manifest, "history": history}
        atomic_torch_save(output / "last.pt", saved)
        if improved:
            atomic_torch_save(output / "best.pt", saved)
        atomic_json(output / "history.json", history)
        print(__import__("json").dumps(entry, allow_nan=False), flush=True)
    final_hash = model.frozen_fingerprint()
    if final_hash != frozen_start:
        raise RuntimeError("frozen backbone changed during training")
    atomic_json(output / "status.json", {"completed_epochs": len(history), "best_validation_accuracy": best_accuracy,
                                        "planned_epochs": config["epochs"], "complete": len(history) == config["epochs"],
                                        "backbone_unchanged": True, "test_accessed": False,
                                        "efficacy_established": False})


if __name__ == "__main__":
    main()
