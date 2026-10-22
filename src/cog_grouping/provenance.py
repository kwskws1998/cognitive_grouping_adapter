"""Atomic artifacts and content fingerprints for experiment provenance."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
from typing import Any

import numpy as np
import torch


def atomic_json(path, value: Any):
    """Write complete JSON or leave the previous artifact intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".writing-", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(path, value):
    """Atomically replace a checkpoint without partially overwriting a usable file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=str(path.parent))
    try:
        with os.fdopen(handle, "wb") as stream:
            torch.save(value, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def state_fingerprint(module):
    """Hash ordered tensor names, shapes, dtypes and exact parameter/buffer bytes."""
    return tensor_mapping_fingerprint(module.state_dict())


def tensor_mapping_fingerprint(mapping):
    """Hash a named subset, for example frozen tensors in a distributed adapter model."""
    digest = hashlib.sha256()
    for name, tensor in sorted(mapping.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update((name + str(tuple(value.shape)) + str(value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path):
    """Hash a file incrementally, including large pretrained checkpoints."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment():
    """Record the runtime rather than inferring a GPU or precision setting."""
    versions = {}
    for package in ("torch", "torchvision", "timm", "numpy", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "python_executable": sys.executable, "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "packages": versions, "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def seed_everything(seed: int):
    """Control Python, NumPy, CPU and CUDA sampling; deterministic kernels are requested."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    """Make data-loader augmentations reproducible across worker processes."""
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def source_fingerprint(root):
    """Record source content while excluding dependencies, caches and run artifacts."""
    root = Path(root)
    files = {}
    for directory in ("src", "scripts", "configs", "tests"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json"):
                files[str(path.relative_to(root))] = file_sha256(path)
    payload = json.dumps(files, sort_keys=True).encode()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "files": files}
