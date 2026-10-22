# Cognitive Grouping Adapter

A spatial evidence adapter after a frozen pretrained image encoder, with matched local controls and full official ParaX. Primary experiments now use real image benchmarks; synthetic crowding is an auxiliary diagnostic.

## 1. Conda setup

Run from the project root (`/workspace/cognitive_grouping_adapter` on the GPU server).

```text
conda env create -f environment.yml
conda activate cognitive_grouping
python scripts/setup_env.py --torch-index-url https://download.pytorch.org/whl/cu128
python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

Create the environment once and activate it on later sessions. Setup validates the active conda interpreter before installing PyTorch 2.8.0, torchvision 0.23.0, timm 1.0.22 and baseline dependencies. For CPU/macOS, omit the torch index argument. Resolved packages are recorded in `environment.resolved.txt`. CuBLAS workspace configuration is set before importing PyTorch, and unsupported deterministic operations raise errors.

## 2. Weights and tests

```text
python scripts/fetch_weights.py
python scripts/fetch_references.py --parax
python scripts/check.py --report reports/current_tests.json
python scripts/audit.py
```

Official code is pinned and hashed. Launcher text is archived in JSON; no shell launcher is created or executed. Exact paper excerpts and source differences are documented in `docs/BENCHMARK_PROTOCOL.md`.

## 3. Real benchmark data

```text
python scripts/prepare_data.py --config configs/benchmark/cifar100.json --cifar100-mirror-url https://dataset.bj.bcebos.com/cifar/cifar-100-python.tar.gz
python scripts/prepare_data.py --config configs/benchmark/svhn.json
python scripts/prepare_data.py --config configs/benchmark/food101.json
```

The CIFAR mirror is optional; torchvision enforces the official archive MD5. Preparation validates training-pool/class counts and records a stratified validation split from the training pool. It never constructs final-test datasets, even when downloaded archives contain test files.

ImageNet-R uses `configs/benchmark/imagenet-r.json`. Supply `data/imagenet-r/train` and `test` with the reference 24,000/6,000 images and 200 matching classes, plus `split_provenance.json` containing a `source` URL or reference identifier. No new random split is generated. Paths/hashes are recorded and final evaluation checks content overlap; the supplied split identity has no publisher-verified digest.

## 4. Real-image integration and memory

```text
python scripts/validate_benchmark.py --output runs/realdata_check_v1 --device cuda --precision bf16
python scripts/smoke_official.py --config configs/benchmark/reference/cifar100.json --device cuda --output runs/parax_reference_smoke.json
python scripts/profile_cuda.py --config configs/benchmark/cifar100.json --output runs/grouping_memory_v1.json
python scripts/profile_cuda.py --config configs/benchmark/cifar100.json --mode official_parax --output runs/parax_memory_v1.json
python scripts/profile_cuda.py --config configs/benchmark/reference/cifar100.json --output runs/parax_reference_memory_v1.json
```

The real-image check trains grouping and full ParaX on 16 actual training images for two epochs and evaluates eight validation images. It verifies finite updates, frozen weights and saved evaluation, not benchmark performance. Profiling includes optimizer-state warmup. Physical batch 8 with accumulation 128 gives nominal effective batch 1024. Measure target-GPU memory before full training. Reference BatchNorm still uses the physical batch; accumulation does not reproduce per-device batch-256 statistics.

## 5. Development comparisons

Start with CIFAR-100 and seed 42:

```text
python scripts/run_benchmarks.py --datasets cifar100 --seeds 42
python scripts/summarize_benchmarks.py --root runs/benchmark_v1/matched/development
```

This sequentially runs `linear`, `mlp`, `spatial`, `boundary`, `grouping`, `fixed_split`, `no_unique`, `fixed_width`, and full official ParaX. Matched conditions share preprocessing, data, head initialization, SGD, 100 epochs, 20 warmup epochs, absolute LR 0.1, effective batch 1024 and BF16. Nonlinear local controls have identical projection shapes and budgets. ParaX retains all distributed blocks, rank 128 and convolution kernels 3/5/7; its larger budget is reported separately.

Give every method an equal validation-only LR search, retaining completed pilot conditions:

```text
python scripts/run_benchmarks.py --datasets cifar100 --seeds 42 --learning-rates 0.03 0.1 0.3 --resume
```

Use `--datasets cifar100 svhn food101` for all automatic datasets, and add `imagenet-r` when prepared. `--dry-run` prints resolved conditions. `--resume` checks source/config identity, skips complete runs and resumes interrupted ones. `--epochs-this-invocation 1` stops each condition after one epoch without changing its 100-epoch schedule.

## 6. Reference ParaX strength check

```text
python scripts/run_benchmarks.py --recipe reference --datasets cifar100 --learning-rates 0.1 0.4 --seeds 42
```

Reference uses the official CLS + BatchNorm readout, FP32, convolution paths, SGD and stochastic-layer behavior. The source launcher scales `blr=0.1` to actual LR 0.4 at effective batch 1024; the paper states initial LR 0.1. Food-101 retains source drop-path 0.05. Physical batch, accumulation, validation partition and final-test policy differ from the original multi-GPU execution. Keep this single-GPU reference check separate from matched comparisons and published table scores.

## 7. Freeze, confirm, then evaluate final test

After development candidates finish:

```text
python scripts/freeze_benchmarks.py --root runs/benchmark_v1/matched/development --output runs/benchmark_v1/matched/frozen_plan.json
python scripts/run_benchmarks.py --phase confirmation --frozen-plan runs/benchmark_v1/matched/frozen_plan.json --seeds 42 43 44
python scripts/summarize_benchmarks.py --root runs/benchmark_v1/matched/confirmation --allow-test
```

Each method selects its own LR by mean validation accuracy, breaking ties by NLL. Freezing rejects unequal search budgets, incomplete candidates, changed source and prior final-test access. Confirmation trains on the complete official training pool for the fixed epoch count; its checkpoint is the final epoch. Test evaluation is explicit and training cannot resume afterward. Freeze/reference confirmation can be performed separately on the `reference/development` root.

Reports include top-1/top-5 accuracy, NLL, seed mean/std, parameters, optimizer steps, training time, measured GPU memory and source/encoder hashes. No published scores are mixed with observed results.

## Historical diagnostics

The existing crowding scripts remain for mechanism diagnostics. Sparse validation accuracy at ceiling cannot establish grouping benefits. Legacy 15-epoch crowding and 30-epoch CIFAR configurations remain historical pilots. New runs use `runs/benchmark_v1`; evaluate older checkpoints using their matching source snapshot.

Current evidence is in `reports/CONDA_BENCHMARK_VALIDATION.md`; original macOS evidence remains in `reports/VALIDATION.md`.
