"""Project-local imports, cache paths, and configuration validation."""

import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "checkpoints" / "torch"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "checkpoints" / "huggingface"))


def read_config(path):
    """Reject unknown keys to make experiment typos visible."""
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    allowed = {"backbone", "pretrained", "weights_path", "mode", "bottleneck", "grouping", "dataset",
               "data_root", "download", "image_size", "validation_fraction", "split_seed", "seed",
               "train_samples", "val_samples", "test_samples", "density", "neighbors", "distant_similarity",
               "max_train_samples", "max_val_samples", "max_test_samples", "batch_size", "accumulation_steps",
               "workers", "epochs", "warmup_epochs", "learning_rate", "weight_decay", "precision", "device",
               "max_grad_norm", "threads"}
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
    return config
