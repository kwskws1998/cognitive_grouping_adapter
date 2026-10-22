# Verified ParaX benchmark contract

## Paper evidence

Source: [ParaX v2, section 4.4](https://arxiv.org/html/2602.06862v2#S4.SS4).

Exact excerpt: **“all models are trained for 100 epochs with an initial learning rate of 0.1”**.

The section names **“CIFAR-100”**, **“SVHN”**, **“Food-101”**, and **“ImageNet-R”**. It specifies warmup 20, batch 1024 and an ImageNet-R split of 24,000 training and 6,000 evaluation images. These are reported settings, not results from this repository.

## Code evidence

Pinned revision: `a1c4aa049b1d859db287535f3597b113ed01a2a4` at [LMMMEng/ParaX](https://github.com/LMMMEng/ParaX/tree/a1c4aa049b1d859db287535f3597b113ed01a2a4). `PARAX_LAUNCHER_EVIDENCE.json` preserves the exact classification launchers and Git blob IDs. Vendor Python files remain unmodified and have SHA-256 manifests.

| Component | Official source | New implementation |
|---|---|---|
| Data | CIFAR-100, SVHN, Food-101 official pools; supplied ImageNet-R folders | Same data identities; validation only from training |
| Train transform | Single-draw/clamped crop 224, scale 0.08–1, aspect 3/4–4/3, random flip | Sampling checked against unchanged vendor crop |
| Eval transform | Bicubic short-side resize 256, center crop 224 | Same sequence |
| Normalization | ImageNet mean/std for MAE | Same |
| Optimizer | SGD, momentum 0.9, weight decay 0 | Same |
| Schedule | Fractional-epoch cosine, 100 epochs, warmup 20 | Same formula at accumulation boundaries |
| LR | `lr = blr * effective_batch / 256`, launcher `blr=0.1` | Explicit matched LR 0.1; reference grid 0.1/0.4 |
| ParaX | Rank 128 default, convolution enabled, kernels (3,5,7), router 16 | All paths restored; all official blocks retained |
| Readout | CLS, BatchNorm1d(affine=False, eps=1e-6), linear head | Retained in reference; common patch mean in matched recipe |
| Food-101 drop path | 0.05 in launcher | Retained in reference |
| Precision | Engine disables autocast | Reference FP32; all matched methods BF16 |
| Evaluation | Test evaluated every epoch and maximum logged | Validation selection, then full-pool fixed-epoch confirmation and final test |

The source's actual LR 0.4 at batch 1024 differs from the paper's stated 0.1. Both are explicit. The old bridge's disabled convolutions and rank-192 configuration remain legacy settings and are not the new benchmark default.

## Comparability and limits

Matched methods share preprocessing, data roles, precision, optimizer, sample budget and readout. Local nonlinear controls also share learned projection shapes. ParaX has different placement and a larger separately reported budget. Each method receives the same development LR/seed grid before selecting its own LR.

Reference is a single-GPU strength check. Physical batch 8 accumulated 128 times preserves nominal batch 1024, but BatchNorm sees eight samples rather than the original per-GPU 256. The final short accumulation group is correctly normalized by its actual sample count. This is not exact multi-GPU published-score reproduction. Reference and matched recipes remain separate tables.

Development holds out a fixed stratified 10% of the training pool. Test datasets are not constructed during training/preparation/selection. Frozen plans contain candidate manifest and source hashes. Confirmation uses the full official training pool and final epoch without validation selection. Different method-specific LRs can share a confirmation summary only under the same frozen-plan hash; source, split, precision and recipe differences remain incompatible.

ImageNet-R requires externally supplied reference folders and a split source record. Training paths/content hashes are retained; final evaluation checks test count, class mappings and train/test content overlap. The supplied split has no publisher-verified digest.

Unit fixtures check computation/protocol, real-weight smokes check encoder retention and updates, and tiny real-image runs check train/save/evaluate integration. Full-data multi-seed experiments remain necessary for performance claims.
