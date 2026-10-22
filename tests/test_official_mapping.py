"""Check checkpoint conversion without claiming baseline benchmark reproduction."""

from functools import partial
from pathlib import Path
import types
import unittest

import torch
from torch import nn

from cog_grouping.official_baseline import load_vendor, translate_timm_encoder


class CheckpointMappingTests(unittest.TestCase):
    """Translation must preserve frozen attention and MLP computation."""

    def test_packed_qkv_is_split_in_correct_order(self):
        """Query, key and value chunks cannot be swapped silently."""
        value = torch.arange(18).reshape(6, 3)
        result = translate_timm_encoder({"blocks.0.attn.qkv.weight": value,
                                         "blocks.0.mlp.fc1.weight": torch.ones(4, 3)})
        torch.testing.assert_close(result["blocks.0.attn.q_proj.weight"], value[:2])
        torch.testing.assert_close(result["blocks.0.attn.k_proj.weight"], value[2:4])
        torch.testing.assert_close(result["blocks.0.attn.v_proj.weight"], value[4:])
        self.assertIn("blocks.0.fc1.weight", result)

    def test_neutralized_vendor_core_matches_timm(self):
        """An independently computed tiny ViT checks the full pretrained-key mapping."""
        root = Path(__file__).resolve().parents[1]
        if not (root / "third_party/ParaX/SOURCE_MANIFEST.json").exists():
            self.skipTest("pinned official sources have not been fetched")
        try:
            from timm.models.vision_transformer import VisionTransformer
            vendor = load_vendor(root)
        except ImportError as error:
            self.skipTest("optional baseline dependencies unavailable: " + str(error))
        torch.set_num_threads(1)
        torch.manual_seed(90)
        kwargs = {"img_size": 32, "patch_size": 8, "num_classes": 0, "embed_dim": 24,
                  "depth": 2, "num_heads": 3, "mlp_ratio": 4, "qkv_bias": True,
                  "norm_layer": partial(nn.LayerNorm, eps=1e-6)}
        original = VisionTransformer(**kwargs).eval()
        config = types.SimpleNamespace(ffn_adapt=False, vpt_on=False)
        converted = vendor.VisionTransformer(**kwargs, tuning_config=config, parax_rank=4).eval()
        missing = converted.load_state_dict(translate_timm_encoder(original.state_dict()), strict=False)
        self.assertFalse(missing.unexpected_keys)
        self.assertTrue(all(".adapter." in key or key.startswith("peft_params.") for key in missing.missing_keys))
        with torch.no_grad():
            converted.peft_params["projB_params"].zero_()
            image = torch.randn(2, 3, 32, 32)
            first = original.forward_features(image)
            tokens = converted.patch_embed(image)
            tokens = torch.cat((converted.cls_token.expand(2, -1, -1), tokens), dim=1)
            tokens = converted.pos_drop(tokens + converted.pos_embed)
            for block in converted.blocks:
                tokens = block(tokens, params=converted.peft_params)
            second = converted.norm(tokens)
        torch.testing.assert_close(first, second, atol=3e-6, rtol=3e-5)
