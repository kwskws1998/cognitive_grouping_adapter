"""Frozen torchvision ResNets and timm ViTs with explicit weight provenance."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor, nn


@dataclass
class BackboneOutput:
    """Retain patch layout until after the adapter, with no classifier tokens."""

    spatial: Tensor


class FrozenBackbone(nn.Module):
    """Keep dropout, normalization statistics, parameters, and autograd frozen."""

    def train(self, mode: bool = True):
        """A parent classifier's train() must never change the frozen encoder."""
        super().train(False)
        return self

    def freeze(self):
        """Disable parameter gradients and force evaluation mode recursively."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
        return self


class ResNetBackbone(FrozenBackbone):
    """Extract the final convolutional map; default torchvision weights are explicit."""

    def __init__(self, name: str, pretrained: bool, weights_path: Optional[str] = None):
        """Never silently fall back to random weights after a download failure."""
        super().__init__()
        from torchvision import models
        if name not in ("resnet18", "resnet50"):
            raise ValueError("supported ResNets: resnet18, resnet50")
        weights_type = models.ResNet18_Weights if name == "resnet18" else models.ResNet50_Weights
        weights = weights_type.DEFAULT if pretrained and not weights_path else None
        model = getattr(models, name)(weights=weights)
        self.weights_info = {"architecture": name, "pretrained": bool(pretrained),
                             "weights": str(weights) if weights else "random"}
        if weights_path:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=True)
            self.weights_info["weights"] = str(Path(weights_path).resolve())
            self.weights_info["pretrained"] = True
        self.channels = model.fc.in_features
        self.encoder = nn.Sequential(*list(model.children())[:-2])
        self.data_config = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
        self.freeze()

    @torch.no_grad()
    def forward(self, images: Tensor) -> BackboneOutput:
        """Extract tensors under no_grad; downstream adapters remain differentiable."""
        return BackboneOutput(self.encoder(images))


class TimmViTBackbone(FrozenBackbone):
    """Extract the normalized patch grid of a ViT without training any encoder layer."""

    def __init__(self, name: str, pretrained: bool, weights_path: Optional[str] = None):
        """Accept official MAE encoder weights with a checked key allowlist."""
        super().__init__()
        import timm
        from timm.data import resolve_model_data_config
        if not name.startswith("vit_"):
            raise ValueError("only timm ViT architectures are supported here")
        self.encoder = timm.create_model(name, pretrained=pretrained and not weights_path, num_classes=0)
        self.channels = self.encoder.num_features
        self.weights_info = {"architecture": name, "pretrained": bool(pretrained),
                             "weights": name if pretrained else "random"}
        if weights_path:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            if "model" in state:
                state = state["model"]
            allowed_prefixes = ("decoder_", "mask_token", "head.", "fc_norm.")
            state = {key: value for key, value in state.items() if not key.startswith(allowed_prefixes)}
            incompatible = self.encoder.load_state_dict(state, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError("MAE encoder keys do not match: " + str(incompatible))
            self.weights_info["weights"] = str(Path(weights_path).resolve())
            self.weights_info["pretrained"] = True
        cfg = resolve_model_data_config(self.encoder)
        self.data_config = {"mean": cfg["mean"], "std": cfg["std"]}
        self.freeze()

    @torch.no_grad()
    def forward(self, images: Tensor) -> BackboneOutput:
        """Preserve rectangular grids and remove exactly the model's prefix tokens."""
        tokens = self.encoder.forward_features(images)
        if not isinstance(tokens, Tensor) or tokens.ndim != 3:
            raise ValueError("timm backbone must return B,N,C tokens")
        tokens = tokens[:, self.encoder.num_prefix_tokens:]
        patch_h, patch_w = self.encoder.patch_embed.patch_size
        height, width = images.shape[-2] // patch_h, images.shape[-1] // patch_w
        if tokens.shape[1] != height * width:
            raise ValueError("patch token count does not match the image grid")
        return BackboneOutput(tokens.transpose(1, 2).reshape(images.shape[0], self.channels, height, width))


def make_backbone(name: str, pretrained: bool = True, weights_path: Optional[str] = None):
    """Use the same backbone factory across all post-backbone adapter controls."""
    if name.startswith("resnet"):
        return ResNetBackbone(name, pretrained, weights_path)
    return TimmViTBackbone(name, pretrained, weights_path)
