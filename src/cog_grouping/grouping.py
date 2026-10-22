"""Finite-window directional Laplace grouping with explicit evidence accounting.

This proposed 2D operator borrows local similarity-dependent spread and a
saddle-based unique/shared distinction. It does not reproduce the paper's
continuous 1D integration, group cutoff, eccentricity, or behavioral readout.
"""

from dataclasses import dataclass
import math
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GroupingConfig:
    """Specify spatial extent in feature-grid cells, not retinal visual angles."""

    radius: int = 2
    min_width: float = 0.2
    max_width: float = 1.25
    similarity_temperature: float = 0.7
    samples_per_cell: int = 2
    adaptive_width: bool = True
    eps: float = 1e-6

    def __post_init__(self):
        """Reject settings that make the directional integral ill-defined."""
        if not isinstance(self.radius, int) or self.radius < 1:
            raise ValueError("radius must be a positive integer")
        if not isinstance(self.samples_per_cell, int) or self.samples_per_cell < 2:
            raise ValueError("samples_per_cell must be an integer >= 2")
        for name in ("min_width", "max_width", "similarity_temperature", "eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.min_width > self.max_width:
            raise ValueError("min_width cannot exceed max_width")


@dataclass
class GroupingResult:
    """Return inspectable B,C,H,W maps and B,4,H,W directional widths."""

    unique: Tensor
    shared: Tensor
    context: Tensor
    private_fraction: Tensor
    widths: Tensor


def shifted(x: Tensor, dy: int, dx: int) -> Tuple[Tensor, Tensor]:
    """Read x[y+dy,x+dx], returning zeros and a validity mask outside the grid."""
    height, width = x.shape[-2:]
    y = torch.arange(height, device=x.device) + dy
    col = torch.arange(width, device=x.device) + dx
    valid = ((y >= 0) & (y < height))[:, None] & ((col >= 0) & (col < width))[None, :]
    out = x.index_select(-2, y.clamp(0, height - 1)).index_select(-1, col.clamp(0, width - 1))
    return out * valid.to(out.dtype), valid[None, None]


class DirectionalGrouping(nn.Module):
    """Split nonnegative evidence using horizontal and vertical activity profiles.

    The private fraction is the minimum over four sides of central-basin excess
    above the nearest saddle, divided by total local group area on that side.
    An axis with no other active source returns fraction one. All integration
    and similarity calculations run in float32 (float64 is retained for tests).
    """

    directions = ((0, -1), (0, 1), (-1, 0), (1, 0))

    def __init__(self, config: GroupingConfig = GroupingConfig()):
        """Create a parameter-free geometry operator with explicit configuration."""
        super().__init__()
        self.config = config

    def neighbor_widths(self, descriptors: Tensor) -> Tensor:
        """Compute each source's directional spread from its immediate neighbor."""
        cfg = self.config
        z = F.normalize(descriptors, dim=1, eps=cfg.eps)
        widths = []
        for dy, dx in self.directions:
            neighbor, valid = shifted(z, dy, dx)
            distance_sq = (z - neighbor).square().sum(1, keepdim=True)
            similarity = torch.exp(-distance_sq / (2 * cfg.similarity_temperature ** 2))
            if not cfg.adaptive_width:
                similarity = torch.ones_like(similarity)
            width = cfg.min_width + (cfg.max_width - cfg.min_width) * similarity
            widths.append(torch.where(valid, width, torch.full_like(width, cfg.min_width)))
        return torch.cat(widths, dim=1)

    def _axis(self, evidence: Tensor, widths: Tensor, axis: int):
        """Integrate finite-window profiles and select the nearest side minima."""
        cfg = self.config
        radius = cfg.radius
        count = 2 * radius * cfg.samples_per_cell + 1
        t = torch.linspace(-radius, radius, count, device=evidence.device, dtype=evidence.dtype)
        source_offsets = torch.arange(-radius, radius + 1, device=evidence.device, dtype=evidence.dtype)
        amplitudes, negative_widths, positive_widths, source_validity = [], [], [], []
        for offset in range(-radius, radius + 1):
            dy, dx = (0, offset) if axis == 0 else (offset, 0)
            value, valid = shifted(evidence, dy, dx)
            left, _ = shifted(widths[:, axis * 2:axis * 2 + 1], dy, dx)
            right, _ = shifted(widths[:, axis * 2 + 1:axis * 2 + 2], dy, dx)
            amplitudes.append(value)
            source_validity.append(valid)
            negative_widths.append(torch.where(valid, left, torch.full_like(left, cfg.min_width)))
            positive_widths.append(torch.where(valid, right, torch.full_like(right, cfg.min_width)))
        amplitude = torch.stack(amplitudes, dim=-1)
        negative = torch.stack(negative_widths, dim=-1)
        positive = torch.stack(positive_widths, dim=-1)
        displacement = t[None, :] - source_offsets[:, None]
        spread = torch.where(displacement[None, None, None, None] < 0, negative[..., None], positive[..., None])
        kernel = torch.exp(-displacement.abs()[None, None, None, None] / spread)
        kernel = kernel * torch.stack(source_validity, dim=-1)[..., None]
        landscape = (amplitude[..., None] * kernel).sum(-2)
        center_index = radius * cfg.samples_per_cell
        length = evidence.shape[-1] if axis == 0 else evidence.shape[-2]
        coordinate = torch.arange(length, device=evidence.device, dtype=evidence.dtype)
        coordinate = coordinate[None, None, None, :, None] if axis == 0 else coordinate[None, None, :, None, None]
        sample_valid = (coordinate + t >= -0.5) & (coordinate + t <= length - 0.5)
        trough = torch.zeros_like(landscape, dtype=torch.bool)
        left, mid, right = landscape[..., :-2], landscape[..., 1:-1], landscape[..., 2:]
        local = (mid <= left) & (mid <= right) & ((mid < left) | (mid < right))
        local_valid = sample_valid[..., :-2] & sample_valid[..., 1:-1] & sample_valid[..., 2:]
        trough[..., 1:-1] = local & local_valid
        side_fractions = []
        center_value = landscape[..., center_index]
        measure = torch.ones_like(t)
        measure[0] = measure[-1] = 0.5
        measure[center_index] = 0.5
        measure = sample_valid * measure / cfg.samples_per_cell
        for side in (-1, 1):
            indices = torch.arange(center_index + side, count if side == 1 else -1, side, device=evidence.device)
            minima = trough.index_select(-1, indices)
            has_minimum = minima.any(-1)
            first = minima.to(torch.int64).argmax(-1)
            candidates = landscape.index_select(-1, indices)
            threshold = candidates.gather(-1, first[..., None]).squeeze(-1)
            threshold = torch.where(has_minimum, threshold, center_value)
            side_sources = amplitude[..., radius + 1:] if side == 1 else amplitude[..., :radius]
            distances = t.index_select(0, indices).abs()
            basin_extent = distances[first]
            basin_extent = torch.where(has_minimum, basin_extent, torch.zeros_like(basin_extent))
            side_mask = (t * side >= 0).to(evidence.dtype)
            basin_mask = t.abs() <= basin_extent[..., None]
            unique_area = ((landscape - threshold[..., None]).clamp_min(0) * measure * side_mask * basin_mask).sum(-1)
            total_area = (landscape * measure * side_mask).sum(-1)
            fraction = unique_area / total_area.clamp_min(cfg.eps)
            fraction = torch.where(side_sources.sum(-1) <= cfg.eps, torch.ones_like(fraction), fraction)
            side_fractions.append(fraction)
        fraction = torch.minimum(side_fractions[0], side_fractions[1]).clamp(0, 1)
        fraction = torch.where(evidence > cfg.eps, fraction, torch.zeros_like(fraction))
        center_kernel = kernel[..., center_index]
        context = (amplitude * center_kernel).sum(-1) / center_kernel.sum(-1).clamp_min(cfg.eps)
        return fraction, context

    def forward(self, evidence: Tensor, descriptors: Tensor) -> GroupingResult:
        """Partition nonnegative amplitudes; never mix pixels across image edges."""
        if evidence.ndim != 4 or descriptors.ndim != 4:
            raise ValueError("evidence and descriptors must be B,C,H,W tensors")
        if evidence.shape[0] != descriptors.shape[0] or evidence.shape[-2:] != descriptors.shape[-2:]:
            raise ValueError("evidence and descriptor batch/grid shapes must match")
        if min(evidence.shape) < 1 or min(descriptors.shape) < 1:
            raise ValueError("empty feature maps are not supported")
        if evidence.device != descriptors.device:
            raise ValueError("evidence and descriptors must share a device")
        if not evidence.is_floating_point() or not descriptors.is_floating_point():
            raise TypeError("grouping inputs must be floating-point tensors")
        with torch.autocast(device_type=evidence.device.type, enabled=False):
            dtype = torch.float64 if evidence.dtype == torch.float64 else torch.float32
            amplitude = evidence.to(dtype)
            if not torch.isfinite(amplitude).all() or (amplitude < 0).any():
                raise ValueError("evidence must be finite and nonnegative")
            if not torch.isfinite(descriptors).all():
                raise ValueError("descriptors must be finite")
            widths = self.neighbor_widths(descriptors.to(dtype))
            horizontal, context_h = self._axis(amplitude, widths, 0)
            vertical, context_v = self._axis(amplitude, widths, 1)
            fraction = torch.minimum(horizontal, vertical)
            unique = amplitude * fraction
            shared = amplitude - unique
            context = (context_h + context_v) * 0.5
        return GroupingResult(unique, shared, context, fraction, widths)
