"""Single post-backbone adapters and matched mechanism controls."""

from dataclasses import replace
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .grouping import DirectionalGrouping, GroupingConfig, GroupingResult, shifted


class EvidenceAdapter(nn.Module):
    """Keep the same learned projections across grouping, spatial and edge controls.

    All non-identity modes use C->D and 3D->C projections and start with a zero
    residual update. The cognitive difference is the deterministic middle operator.
    """

    modes = ("grouping", "no_unique", "fixed_split", "fixed_width", "boundary", "spatial", "mlp")

    def __init__(self, channels: int, bottleneck: int = 64, mode: str = "grouping",
                 grouping_config: GroupingConfig = GroupingConfig()):
        """Construct one residual adapter; no classifier or backbone is included."""
        super().__init__()
        if channels < 1 or bottleneck < 1 or mode not in self.modes:
            raise ValueError("invalid adapter dimension or mode")
        self.mode = mode
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        self.down = nn.Conv2d(channels, bottleneck, 1)
        self.up = nn.Conv2d(bottleneck * 3, channels, 1)
        config = replace(grouping_config, adaptive_width=False) if mode == "fixed_width" else grouping_config
        self.grouping = DirectionalGrouping(config)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def _boundary_control(self, amplitude: Tensor, descriptors: Tensor):
        """Use pairwise edge-aware aggregation without a landscape or saddle split."""
        cfg = self.grouping.config
        z = F.normalize(descriptors.float(), dim=1, eps=cfg.eps)
        numerator = amplitude.float()
        denominator = torch.ones_like(amplitude[:, :1], dtype=torch.float32)
        affinity_sum = torch.zeros_like(denominator)
        count = torch.zeros_like(denominator)
        for dy, dx in self.grouping.directions:
            neighbor_z, valid = shifted(z, dy, dx)
            neighbor_a, _ = shifted(amplitude.float(), dy, dx)
            weight = torch.exp(-(z - neighbor_z).square().sum(1, keepdim=True) /
                               (2 * cfg.similarity_temperature ** 2)) * valid
            numerator = numerator + weight * neighbor_a
            denominator = denominator + weight
            affinity_sum = affinity_sum + weight
            count = count + valid
        private = 1 - affinity_sum / count.clamp_min(1)
        return amplitude * private, amplitude * (1 - private), numerator / denominator

    def components(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Optional[GroupingResult]]:
        """Expose the component paths without retaining tensors on module state."""
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("adapter input must match B,C,H,W and configured channels")
        normalized = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        descriptors = self.down(normalized)
        amplitude = F.relu(descriptors)
        result = None
        if self.mode in ("grouping", "no_unique", "fixed_width", "fixed_split"):
            result = self.grouping(amplitude, descriptors)
            unique, shared, context = result.unique, result.shared, result.context
            if self.mode == "no_unique":
                unique = torch.zeros_like(unique)
            if self.mode == "fixed_split":
                unique, shared = amplitude * 0.5, amplitude * 0.5
        elif self.mode == "boundary":
            unique, shared, context = self._boundary_control(amplitude, descriptors)
        elif self.mode == "spatial":
            kernel = 2 * self.grouping.config.radius + 1
            context = F.avg_pool2d(amplitude, kernel, stride=1, padding=kernel // 2, count_include_pad=False)
            unique, shared = amplitude * 0.5, amplitude * 0.5
        else:
            unique, shared, context = amplitude * 0.5, amplitude * 0.5, amplitude
        return unique, shared, context, result

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual correction, preserving the feature-map shape and dtype."""
        unique, shared, context, _ = self.components(x)
        packed = torch.cat((unique, shared, context), dim=1).to(x.dtype)
        return x + self.up(packed)

    def diagnostics(self, x: Tensor) -> Dict[str, float]:
        """Produce detached scalar diagnostics for reporting, not supervision."""
        with torch.no_grad():
            unique, shared, _, result = self.components(x)
            total = unique + shared
            stats = {"unique_mass": float(unique.mean()), "shared_mass": float(shared.mean())}
            if result is not None:
                active = total > self.grouping.config.eps
                stats["active_private_fraction"] = float(result.private_fraction[active].mean()) if active.any() else 0.0
            return stats
