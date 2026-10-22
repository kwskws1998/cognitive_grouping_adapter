"""Training/validation/test separation and deterministic crowding diagnostics."""

import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from .provenance import file_sha256


class CrowdingDataset(Dataset):
    """Render balanced target orientations with label-independent flanker geometry.

    These artificial images test an experimental prediction, not natural-image
    effectiveness or reproduction of the human peripheral-vision experiment.
    """

    def __init__(self, size=256, image_size=224, seed=42, density="dense",
                 neighbors="similar", distant_similarity=0.5, transform=None):
        """Seeds must be disjoint across data roles; condition is never a label."""
        if size < 2 or size % 2:
            raise ValueError("crowding size must be positive and even for exact label balance")
        if image_size < 64 or density not in ("sparse", "dense", "alone"):
            raise ValueError("invalid image size or density")
        if neighbors not in ("similar", "dissimilar", "mixed") or not 0 <= distant_similarity <= 1:
            raise ValueError("invalid neighbor condition or distant similarity")
        self.size, self.image_size, self.seed = size, image_size, seed
        self.density, self.neighbors = density, neighbors
        self.distant_similarity, self.transform = distant_similarity, transform
        self.targets = [index % 2 for index in range(size)]

    def __len__(self):
        """Return the fixed number of balanced trials."""
        return self.size

    def _draw_element(self, canvas, cx, cy, angle, contrast):
        """Add a windowed bar without changing label-dependent rendering nuisance."""
        radius = max(3, round(self.image_size * 0.032))
        xx, yy = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
        theta = math.radians(angle)
        across = xx * math.cos(theta) - yy * math.sin(theta)
        along = xx * math.sin(theta) + yy * math.cos(theta)
        patch = contrast * np.exp(-across ** 2 / (2 * (radius / 5) ** 2) - along ** 2 / (2 * (radius / 1.5) ** 2))
        left, top = round(cx) - radius, round(cy) - radius
        right, bottom = left + patch.shape[1], top + patch.shape[0]
        x0, y0, x1, y1 = max(0, left), max(0, top), min(canvas.shape[1], right), min(canvas.shape[0], bottom)
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] += patch[y0 - top:y1 - top, x0 - left:x1 - left]

    def __getitem__(self, index):
        """Deterministically render each trial; local versus distant changes are paired."""
        rng = np.random.default_rng(self.seed + index * 104729)
        size = self.image_size
        label = self.targets[index]
        canvas = np.full((size, size), 0.35, dtype=np.float32)
        jitter = rng.uniform(-0.01, 0.01, 2) * size
        center_x, center_y = size / 2 + jitter[0], size / 2 + jitter[1]
        contrast = float(rng.uniform(0.4, 0.6))
        background_noise = rng.normal(0, 0.005, canvas.shape)
        self._draw_element(canvas, center_x, center_y, -12 if label == 0 else 12, contrast)
        ring_count = {"alone": 0, "sparse": 1, "dense": 3}[self.density]
        for ring in range(1, ring_count + 1):
            for element in range(8 * ring):
                angle = element * 2 * math.pi / (8 * ring)
                x = center_x + ring * size * 0.12 * math.cos(angle)
                y = center_y + ring * size * 0.12 * math.sin(angle)
                if ring == 1:
                    similar = self.neighbors == "similar" or (self.neighbors == "mixed" and element % 2 == 0)
                else:
                    similar = rng.random() < self.distant_similarity
                self._draw_element(canvas, x, y, 0 if similar else 90, contrast)
        canvas += background_noise
        image = Image.fromarray(np.uint8(np.clip(canvas, 0, 1) * 255)).convert("RGB")
        return (self.transform(image) if self.transform else transforms.ToTensor()(image)), label


def stratified_indices(targets, validation_fraction, seed):
    """Split only the official training pool, with at least one item per class/role."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    labels = np.asarray(targets)
    rng = np.random.default_rng(seed)
    training, validation = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError("every class needs at least two training-pool examples")
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * validation_fraction)))
        validation.extend(indices[:count].tolist())
        training.extend(indices[count:].tolist())
    return sorted(training), sorted(validation)


def make_transforms(image_size, mean, std, training=False):
    """Specify identical preprocessing for all locally compared adapter modes."""
    if training:
        spatial = [transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
                   transforms.RandomHorizontalFlip()]
    else:
        spatial = [transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC)]
    return transforms.Compose(spatial + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def _subset_limit(dataset, maximum):
    """Use an explicit deterministic debug subset without changing the data split."""
    if maximum is None:
        return dataset
    if maximum < 1:
        raise ValueError("sample limits must be positive")
    return Subset(dataset, range(min(maximum, len(dataset))))


def make_datasets(config, normalization, include_test=False):
    """Construct training/validation only unless the caller explicitly requests test."""
    name = config["dataset"]
    image_size = config.get("image_size", 224)
    train_transform = make_transforms(image_size, **normalization, training=name != "crowding")
    evaluation_transform = make_transforms(image_size, **normalization)
    seed = config.get("split_seed", 1729)
    test = None
    if name == "crowding":
        common = {"image_size": image_size, "density": config.get("density", "dense"),
                  "neighbors": config.get("neighbors", "mixed"),
                  "distant_similarity": config.get("distant_similarity", 0.5),
                  "transform": evaluation_transform}
        training = CrowdingDataset(config.get("train_samples", 512), seed=seed, **common)
        validation = CrowdingDataset(config.get("val_samples", 128), seed=seed + 100000000, **common)
        if include_test:
            test = CrowdingDataset(config.get("test_samples", 256), seed=seed + 200000000, **common)
        roles = {"kind": "independent procedural seeds", "train_seed": seed,
                 "validation_seed": seed + 100000000, "test_seed": seed + 200000000}
        num_classes = 2
    elif name in ("cifar100", "food101"):
        root = config.get("data_root", "data")
        download = config.get("download", False)
        if name == "cifar100":
            raw = datasets.CIFAR100(root, train=True, download=download)
            targets = raw.targets
            dataset_type = lambda transform: datasets.CIFAR100(root, train=True, transform=transform, download=False)
            test_type = lambda: datasets.CIFAR100(root, train=False, transform=evaluation_transform, download=download)
        else:
            raw = datasets.Food101(root, split="train", download=download)
            targets = raw._labels
            dataset_type = lambda transform: datasets.Food101(root, split="train", transform=transform, download=False)
            test_type = lambda: datasets.Food101(root, split="test", transform=evaluation_transform, download=download)
        train_indices, val_indices = stratified_indices(targets, config.get("validation_fraction", 0.1), seed)
        training = Subset(dataset_type(train_transform), train_indices)
        validation = Subset(dataset_type(evaluation_transform), val_indices)
        test = test_type() if include_test else None
        roles = {"kind": "official training-pool stratification", "seed": seed,
                 "train_indices": train_indices, "validation_indices": val_indices,
                 "test_role": "official held-out test; never used for selection"}
        num_classes = len(raw.classes)
    elif name == "imagefolder":
        root = Path(config["data_root"])
        training = datasets.ImageFolder(root / "train", transform=train_transform)
        validation = datasets.ImageFolder(root / "val", transform=evaluation_transform)
        if training.class_to_idx != validation.class_to_idx:
            raise ValueError("train and val class-to-index mappings differ")
        train_hashes = {path: file_sha256(path) for path, _ in training.samples}
        val_hashes = {path: file_sha256(path) for path, _ in validation.samples}
        if set(train_hashes.values()) & set(val_hashes.values()):
            raise ValueError("identical image content appears in train and val")
        if include_test:
            test = datasets.ImageFolder(root / "test", transform=evaluation_transform)
            if test.class_to_idx != training.class_to_idx:
                raise ValueError("test class-to-index mapping differs")
            test_hashes = {file_sha256(path) for path, _ in test.samples}
            if test_hashes & (set(train_hashes.values()) | set(val_hashes.values())):
                raise ValueError("identical image content appears in final test and training/validation")
        roles = {"kind": "explicit train/val/test directories", "classes": training.class_to_idx,
                 "train_files": [{"path": str(Path(path).relative_to(root)), "sha256": train_hashes[path]} for path, _ in training.samples],
                 "validation_files": [{"path": str(Path(path).relative_to(root)), "sha256": val_hashes[path]} for path, _ in validation.samples]}
        num_classes = len(training.classes)
    else:
        raise ValueError("dataset must be crowding, cifar100, food101 or imagefolder")
    payload = json.dumps(roles, sort_keys=True).encode()
    roles["sha256"] = hashlib.sha256(payload).hexdigest()
    training = _subset_limit(training, config.get("max_train_samples"))
    validation = _subset_limit(validation, config.get("max_val_samples"))
    if test is not None:
        test = _subset_limit(test, config.get("max_test_samples"))
    roles["effective_sizes"] = {"train": len(training), "validation": len(validation),
                                "test": len(test) if test is not None else None}
    return training, validation, test, num_classes, roles
