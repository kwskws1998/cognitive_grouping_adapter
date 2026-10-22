# Conda and real-image benchmark validation

Verified locally on 2026-10-22. This report concerns the new benchmark implementation, not a measured method improvement.

## Actual environment

A new dedicated conda environment was created at `/private/tmp/cognitive_grouping_conda`, without creating another Python environment inside it. Observed Python 3.11.17, PyTorch 2.8.0, torchvision 0.23.0, timm 1.0.22, NumPy 2.4.6, Pillow 11.3.0; macOS ARM64 CPU. `conda_environment.json` records the interpreter and prefix. CUDA was unavailable on this machine.

## Executed checks

| Check | Observed result | Artifact |
|---|---|---|
| Numerical, data, optimizer, checkpoint, selection and CLI suite | 53 passed; 0 failed; 0 skipped | conda_benchmark_tests.json |
| Syntax/comment policy and recursive configs | 35 Python files; 14 configs passed | source_audit.json |
| Actual CIFAR-100 archive | Official MD5 matched; SHA-256 recorded; 100 classes | conda_cifar100_preparation.json |
| CIFAR development partition | 45,000 train / 5,000 validation; final-test object never constructed | conda_cifar100_preparation.json |
| Pretrained MAE + grouping smoke | Both learned projections updated; no encoder gradients; frozen hash retained | conda_mae_smoke.json |
| Official ParaX reference smoke | All 12 blocks, rank 128, convolution 3/5/7, CLS + BN; finite gradients and updated experts; frozen core retained | conda_parax_reference_smoke.json |
| Real-image grouping training/evaluation | 16 training / 8 validation images, two epochs; checkpoint reloaded; frozen encoder unchanged | conda_realdata_check.json |
| Real-image full ParaX training/evaluation | Same real-image subset and epochs; checkpoint reloaded; frozen core unchanged | conda_realdata_check.json |

The original CIFAR server was slow. The completed archive came from the Paddle dataset mirror and passed torchvision's official MD5 `eb9058c3a382ffc7106e4002c42a8d85`. Observed archive SHA-256: `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`.

## Retained negative/small-sample outcomes

The real-image integration subset produced selected validation accuracy 0/8 for grouping and 1/8 for ParaX. Both are too small and too briefly trained for efficacy conclusions. They are retained to avoid presenting an integration smoke as a positive experiment. No final-test evaluation was performed for these retained runs.

The official-source reference smoke counted 3,098,340 trainable adapter parameters at rank 128 with convolution, plus a 2,307-parameter three-class head. These are observed code counts; no published table budget is substituted.

## Protocol checks

Tests compare crop sampling to unchanged vendor code, independently check the fractional learning-rate formula, verify SGD momentum and tail accumulation against real larger batches, check real dataset constructors never open test during development/confirmation, and exercise full-training-pool resume with fixed-final checkpoint selection.

Selection tests enforce complete LR/seed grids, equal budgets between methods, validation-only method-specific choices, frozen-plan replay and rejection after final-test access. Confirmation summaries allow different selected LRs only when the frozen-plan hashes agree. Incomplete training cannot open final test.

## Remaining empirical work

No CUDA memory measurement or full 100-epoch benchmark has been run locally. SVHN/Food-101 download and full-data training, supplied ImageNet-R split verification, multi-seed confirmation, and reference strength on the target RTX 4090 remain to be executed. The reference recipe uses a smaller physical batch than the original launcher; gradient accumulation does not preserve BatchNorm batch statistics. Neither benchmark superiority nor published-score reproduction is established.

Current real-image validation source fingerprint: `33030a7f1afd8f066da07e691bc82b02d7c500fea4aa02727cec1d7deaec8c74`.
