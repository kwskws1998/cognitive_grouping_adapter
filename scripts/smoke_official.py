"""Validate the full official ParaX architecture with the same pretrained MAE encoder."""

import argparse

from _common import PROJECT_ROOT, read_config
import torch
from torch.nn import functional as F

from cog_grouping.backbones import make_backbone
from cog_grouping.official_baseline import OfficialParaXClassifier
from cog_grouping.model import build_classifier
from cog_grouping.provenance import atomic_json, environment, seed_everything


def main():
    """Verify all frozen weights remain unchanged after an actual adapter update."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-path", default=str(PROJECT_ROOT / "checkpoints/mae_pretrain_vit_base.pth"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "runs/official_parax_smoke.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config")
    args = parser.parse_args()
    torch.set_num_threads(2)
    seed_everything(42)
    backbone = make_backbone("vit_base_patch16_224.mae", True, args.weights_path)
    config = read_config(args.config) if args.config else {"mode": "official_parax", "bottleneck": 192}
    config["mode"] = "official_parax"
    model = build_classifier(backbone, 3, config, PROJECT_ROOT).to(args.device)
    del backbone
    frozen_before = model.frozen_fingerprint()
    learnable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    adapter_before = model.encoder.peft_params["projB_params"].detach().clone()
    optimizer = torch.optim.AdamW(learnable, lr=1e-3)
    model.train()
    images = torch.randn(2, 3, 224, 224, device=args.device)
    labels = torch.tensor([0, 1], device=args.device)
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(images), labels)
    loss.backward()
    finite = all(torch.isfinite(p.grad).all().item() for p in learnable if p.grad is not None)
    if not finite:
        raise FloatingPointError("official baseline has nonfinite gradients")
    optimizer.step()
    checks = {"full_12_block_layout_retained": len(model.encoder.blocks) == 12,
              "frozen_core_unchanged": model.frozen_fingerprint() == frozen_before,
              "expert_parameters_changed": not torch.equal(adapter_before, model.encoder.peft_params["projB_params"]),
              "finite_gradients": finite}
    if not all(checks.values()):
        raise RuntimeError(str(checks))
    result = {"checks": checks, "parameters": model.trainable_parameter_counts(), "loss": float(loss.detach()),
              "vendor_revision": model.vendor_revision, "environment": environment(),
              "parax_options": config.get("parax", {}), "readout": model.readout,
              "protocol": config.get("benchmark_protocol", "legacy harmonized architecture smoke"),
              "published_score_reproduced": False, "classification_efficacy_established": False}
    atomic_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
