"""Check a real pretrained encoder, gradient flow, and exact frozen-weight retention."""

import argparse

from _common import PROJECT_ROOT
import torch
from torch.nn import functional as F

from cog_grouping.backbones import make_backbone
from cog_grouping.grouping import GroupingConfig
from cog_grouping.model import FrozenVisionClassifier
from cog_grouping.provenance import atomic_json, environment, seed_everything, state_fingerprint


def main():
    """Synthetic labels validate code paths only; no recognition gain is measured."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--weights-path")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "runs" / "pretrained_smoke.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    seed_everything(42)
    backbone = make_backbone(args.backbone, not args.random, args.weights_path)
    before = state_fingerprint(backbone)
    model = FrozenVisionClassifier(backbone, 3, bottleneck=args.bottleneck, grouping_config=GroupingConfig(radius=2)).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    up_before = model.adapter.up.weight.detach().clone()
    down_before = model.adapter.down.weight.detach().clone()
    losses = []
    for _ in range(3):
        model.train()
        image = torch.randn(2, 3, args.image_size, args.image_size, device=args.device)
        target = torch.tensor([0, 1], device=args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(image), target)
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError("smoke loss is nonfinite")
        optimizer.step()
        losses.append(float(loss.detach()))
    checks = {"backbone_unchanged": state_fingerprint(backbone) == before,
              "backbone_gradients_absent": all(p.grad is None for p in backbone.parameters()),
              "backbone_eval_mode": not backbone.training,
              "adapter_up_changed": not torch.equal(up_before, model.adapter.up.weight),
              "adapter_down_changed": not torch.equal(down_before, model.adapter.down.weight)}
    if not all(checks.values()):
        raise RuntimeError("smoke check failed: " + str(checks))
    result = {"checks": checks, "weights": backbone.weights_info, "backbone_sha256": before,
              "parameters": model.trainable_parameter_counts(), "synthetic_losses": losses,
              "environment": environment(), "efficacy_established": False,
              "validation_kind": "pretrained code-path smoke" if not args.random else "random-backbone smoke"}
    atomic_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
