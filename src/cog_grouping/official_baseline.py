"""Full distributed official ParaX architecture in a harmonized pilot harness.

The vendor files are unmodified Apache-2.0 sources. The checkpoint translation
and patch-mean readout below are ours and are explicitly not the paper's full
training/readout protocol. The baseline retains every official ParaX block.
"""

import importlib
import json
from pathlib import Path
import sys
import types

import torch
from torch import nn

from .provenance import file_sha256, tensor_mapping_fingerprint

PARAX_REVISION = "a1c4aa049b1d859db287535f3597b113ed01a2a4"


def load_vendor(project_root):
    """Verify immutable-source hashes before importing the reviewed official module."""
    root = Path(project_root) / "third_party" / "ParaX"
    manifest_path = root / "SOURCE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("run python scripts/fetch_references.py --parax first")
    manifest = json.loads(manifest_path.read_text())
    if manifest["revision"] != PARAX_REVISION:
        raise ValueError("unexpected official ParaX revision")
    for name in ("models/parax.py", "models/vit_parax.py"):
        if file_sha256(root / name) != manifest["files"][name]:
            raise ValueError("official baseline source has been modified: " + name)
    namespace = "_cog_grouping_official_parax"
    if namespace in sys.modules:
        existing = Path(sys.modules[namespace].__path__[0]).resolve()
        if existing != (root / "models").resolve():
            raise RuntimeError("a different vendor source directory is already loaded")
    else:
        package = types.ModuleType(namespace)
        package.__path__ = [str(root / "models")]
        sys.modules[namespace] = package
    return importlib.import_module(namespace + ".vit_parax")


def translate_timm_encoder(state):
    """Map packed QKV and nested MLP names to official independent projection names."""
    output = {}
    for name, tensor in state.items():
        if name.startswith(("head.", "fc_norm.")):
            continue
        if ".attn.qkv." in name:
            if tensor.shape[0] % 3:
                raise ValueError("invalid packed QKV shape")
            for part, value in zip(("q_proj", "k_proj", "v_proj"), tensor.chunk(3, dim=0)):
                output[name.replace("qkv", part)] = value
        else:
            output[name.replace(".mlp.fc1.", ".fc1.").replace(".mlp.fc2.", ".fc2.")] = tensor
    return output


class OfficialParaXClassifier(nn.Module):
    """Train official ParaX modules across all blocks and a matched linear readout."""

    def __init__(self, backbone, num_classes, rank, project_root):
        """Load every frozen pretrained weight with strict missing-key allowlists."""
        super().__init__()
        if not hasattr(backbone, "encoder") or not hasattr(backbone.encoder, "blocks"):
            raise ValueError("the official baseline bridge requires a timm ViT backbone")
        source = backbone.encoder
        if source.num_features != 768 or len(source.blocks) != 12:
            raise ValueError("official full baseline currently supports ViT-B/16 only")
        if tuple(source.patch_embed.patch_size) != (16, 16):
            raise ValueError("official baseline requires 16x16 patches")
        vendor = load_vendor(project_root)
        self.encoder = vendor.vit_base_patch16_parax(num_classes=0, global_pool=False, parax_rank=rank,
                                                    parax_enable_conv=False, drop_path_rate=0.0)
        translated = translate_timm_encoder(source.state_dict())
        incompatible = self.encoder.load_state_dict(translated, strict=False)
        missing = [name for name in incompatible.missing_keys if ".adapter." not in name and not name.startswith("peft_params.")]
        if missing or incompatible.unexpected_keys:
            raise ValueError("unsafe pretrained conversion: " + str((missing, incompatible.unexpected_keys)))
        for name, parameter in self.encoder.named_parameters():
            parameter.requires_grad_(".adapter." in name or name.startswith("peft_params."))
        self.head = nn.Linear(768, num_classes)
        self.mode = "official_parax"
        self.vendor_revision = PARAX_REVISION

    def train(self, mode=True):
        """Keep frozen stochastic layers in eval while allowing adapters to train."""
        super().train(mode)
        self.encoder.eval()
        for block in self.encoder.blocks:
            block.adapter.train(mode)
        return self

    def forward(self, images):
        """Retain all official distributed adapters, then match the patch-mean head."""
        encoder = self.encoder
        tokens = encoder.patch_embed(images)
        tokens = torch.cat((encoder.cls_token.expand(images.shape[0], -1, -1), tokens), dim=1)
        tokens = encoder.pos_drop(tokens + encoder.pos_embed)
        for block in encoder.blocks:
            tokens = block(tokens, params=encoder.peft_params)
        return self.head(encoder.norm(tokens)[:, 1:].mean(1))

    def trainable_parameter_counts(self):
        """Count the actual loaded baseline, rather than copying a paper's budget."""
        frozen = sum(p.numel() for p in self.encoder.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        return {"backbone_total": frozen, "backbone_trainable": 0, "adapter_trainable": trainable,
                "head_trainable": sum(p.numel() for p in self.head.parameters())}

    def frozen_fingerprint(self):
        """Include only immutable core parameters and buffers in the baseline audit."""
        frozen_names = {name for name, value in self.encoder.named_parameters() if not value.requires_grad}
        state = {name: value for name, value in self.encoder.state_dict().items() if name in frozen_names}
        state.update(dict(self.encoder.named_buffers()))
        return tensor_mapping_fingerprint(state)

    def trainable_state_dict(self):
        """Save only the official adapter tensors and the matched task head."""
        names = {name for name, parameter in self.encoder.named_parameters() if parameter.requires_grad}
        state = {name: value for name, value in self.encoder.state_dict().items() if name in names}
        return {"encoder_adapters": state, "head": self.head.state_dict()}

    def load_trainable_state_dict(self, state):
        """Reject adapter-state mismatches without allowing pretrained core replacement."""
        expected = set(self.trainable_state_dict()["encoder_adapters"])
        if set(state["encoder_adapters"]) != expected:
            raise ValueError("official adapter checkpoint keys do not match")
        self.encoder.load_state_dict(state["encoder_adapters"], strict=False)
        self.head.load_state_dict(state["head"], strict=True)
