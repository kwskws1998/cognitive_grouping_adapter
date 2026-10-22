"""A single adapter after a frozen spatial encoder, followed by a linear head."""

from typing import Dict

import torch
from torch import Tensor, nn

from .adapters import EvidenceAdapter
from .grouping import GroupingConfig


class FrozenVisionClassifier(nn.Module):
    """Train only an optional post-backbone adapter and the classification head."""

    def __init__(self, backbone: nn.Module, num_classes: int, mode: str = "grouping",
                 bottleneck: int = 64, grouping_config: GroupingConfig = GroupingConfig()):
        """Physical placement is identical across the locally implemented controls."""
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.backbone = backbone
        self.mode = mode
        self.adapter = nn.Identity() if mode == "linear" else EvidenceAdapter(
            backbone.channels, bottleneck, mode, grouping_config)
        self.head = nn.Linear(backbone.channels, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """The no-grad encoder output is a normal tensor usable by trainable layers."""
        features = self.backbone(images).spatial
        adapted = self.adapter(features)
        return self.head(adapted.mean(dim=(-2, -1)))

    def trainable_parameter_counts(self) -> Dict[str, int]:
        """Report backbone, adapter and head budgets separately."""
        return {"backbone_total": sum(p.numel() for p in self.backbone.parameters()),
                "backbone_trainable": sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
                "adapter_trainable": sum(p.numel() for p in self.adapter.parameters() if p.requires_grad),
                "head_trainable": sum(p.numel() for p in self.head.parameters() if p.requires_grad)}

    def trainable_state_dict(self):
        """Store only the newly trained components; encoder provenance is separate."""
        return {"adapter": self.adapter.state_dict(), "head": self.head.state_dict()}

    def load_trainable_state_dict(self, state):
        """Use strict loading to prevent silently changing the operator or readout."""
        self.adapter.load_state_dict(state["adapter"], strict=True)
        self.head.load_state_dict(state["head"], strict=True)

    def frozen_fingerprint(self):
        """Fingerprint all frozen encoder parameters and normalization buffers."""
        from .provenance import state_fingerprint
        return state_fingerprint(self.backbone)


def build_classifier(backbone, num_classes, config, project_root):
    """Keep local post-backbone controls separate from the full official ParaX model."""
    if config.get("mode") == "official_parax":
        from .official_baseline import OfficialParaXClassifier
        options = config.get("parax", {})
        model = OfficialParaXClassifier(backbone, num_classes, options.get("rank", config.get("bottleneck", 192)),
                                        project_root, options)
    else:
        model = FrozenVisionClassifier(backbone, num_classes, config.get("mode", "grouping"),
                                       config.get("bottleneck", 64), GroupingConfig(**config.get("grouping", {})))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.get("seed", 42) + 1000003)
        if isinstance(model.head, nn.Sequential):
            model.head[-1].reset_parameters()
            nn.init.trunc_normal_(model.head[-1].weight, std=0.01)
        else:
            model.head.reset_parameters()
    return model
