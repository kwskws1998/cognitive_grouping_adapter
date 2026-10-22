"""Project-local imports, cache paths, and configuration validation."""

import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "checkpoints" / "torch"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "checkpoints" / "huggingface"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def read_config(path):
    """Reject unknown keys to make experiment typos visible."""
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    return validate_config(config)


def validate_config(config):
    """Check both saved configurations and CLI-resolved experiment conditions."""
    allowed = {"backbone", "pretrained", "weights_path", "mode", "bottleneck", "grouping", "dataset",
               "data_root", "download", "image_size", "validation_fraction", "split_seed", "seed",
               "train_samples", "val_samples", "test_samples", "density", "neighbors", "distant_similarity",
               "max_train_samples", "max_val_samples", "max_test_samples", "batch_size", "accumulation_steps",
               "workers", "epochs", "warmup_epochs", "learning_rate", "weight_decay", "precision", "device",
               "max_grad_norm", "threads", "optimizer", "momentum", "nesterov", "min_lr", "schedule",
               "preprocess", "phase", "drop_last", "benchmark_protocol", "parax", "frozen_plan_sha256"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError("unknown configuration keys: " + str(sorted(unknown)))
    for key in ("backbone", "dataset", "epochs", "batch_size"):
        if key not in config:
            raise ValueError("missing required configuration key: " + key)
    for key in ("epochs", "batch_size", "accumulation_steps", "threads"):
        if key in config and (not isinstance(config[key], int) or config[key] < 1):
            raise ValueError(key + " must be a positive integer")
    if config.get("workers", 0) < 0:
        raise ValueError("workers must be nonnegative")
    if not 0 <= config.get("warmup_epochs", 0) < config["epochs"]:
        raise ValueError("warmup_epochs must be >= 0 and smaller than epochs")
    if config.get("phase", "development") not in ("development", "confirmation"):
        raise ValueError("phase must be development or confirmation")
    if config.get("phase") == "confirmation" and config.get("validation_fraction", 0) != 0:
        raise ValueError("confirmation requires validation_fraction=0 and a fixed final epoch")
    if config.get("phase") == "confirmation" and config["dataset"] not in ("cifar100", "svhn", "food101", "imagenet-r"):
        raise ValueError("full-training-pool confirmation is supported for benchmark image datasets only")
    if config.get("optimizer", "adamw") not in ("adamw", "sgd"):
        raise ValueError("optimizer must be adamw or sgd")
    if config.get("schedule", "epoch") not in ("epoch", "parax_iteration"):
        raise ValueError("unknown learning-rate schedule")
    if config.get("preprocess", "pilot") not in ("pilot", "parax"):
        raise ValueError("preprocess must be pilot or parax")
    if config.get("learning_rate", 1e-3) <= 0 or config.get("weight_decay", 0.01) < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    if not 0 <= config.get("min_lr", 0.0) <= config.get("learning_rate", 1e-3):
        raise ValueError("min_lr must be between zero and learning_rate")
    parax = config.get("parax", {})
    if set(parax) - {"rank", "enable_conv", "kernel_sizes", "router_hidden", "drop_path", "readout", "train_stochastic"}:
        raise ValueError("unknown ParaX configuration keys")
    if parax.get("readout", "matched") not in ("matched", "official_cls_bn"):
        raise ValueError("unknown ParaX readout")
    if parax.get("readout") == "official_cls_bn" and config["batch_size"] < 2:
        raise ValueError("official BatchNorm readout requires microbatch >= 2")
    for name, default in (("rank", 128), ("router_hidden", 16)):
        if not isinstance(parax.get(name, default), int) or parax.get(name, default) < 1:
            raise ValueError("ParaX " + name + " must be a positive integer")
    if not 0 <= parax.get("drop_path", 0.0) < 1:
        raise ValueError("ParaX drop_path must be in [0, 1)")
    if parax.get("enable_conv") and (not parax.get("kernel_sizes") or
            any(not isinstance(size, int) or size < 1 or size % 2 == 0 for size in parax["kernel_sizes"])):
        raise ValueError("ParaX convolution requires positive odd kernel sizes")
    return config
