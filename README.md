# Cognitive Grouping Adapter

Research implementation of candidate 1: **preserve item-specific evidence while representing perceptual groups**, attached as one module to a frozen pretrained image encoder.

This repository contains a working PyTorch operator, pretrained ResNet/MAE-ViT integration, matched controls, a pinned official ParaX baseline bridge, training/evaluation scripts, and numerical/protocol tests. It does not establish a recognition improvement or CVPR-level novelty. CUDA memory must be measured on the target GPU.

## 1. Set up

Run commands from this directory. Python 3.11 is recommended for a new Linux environment. The source also supports the Python 3.9 environment used for local verification.

```text
python scripts/setup_env.py --torch-index-url https://download.pytorch.org/whl/cu128
.venv/bin/python scripts/check.py
```

The setup creates a local `.venv`, installs PyTorch 2.8.0, torchvision 0.23.0, timm 1.0.22 and baseline dependencies, and records resolved versions. No shell scripts are generated. For CPU/macOS, omit `--torch-index-url`. An existing environment is never overwritten.

## 2. Inspect the mechanism before training

```text
.venv/bin/python scripts/probe.py
```

Inspect `runs/mechanism_probe/probe.json` and `private_fraction.png`. This is a constructed-feature diagnostic, not a human-data fit or a classification benchmark. The exact computation and its differences from the source model are in `docs/METHOD.md`.

## 3. Verify a pretrained integration

```text
.venv/bin/python scripts/smoke.py --backbone resnet18
.venv/bin/python scripts/fetch_weights.py
.venv/bin/python scripts/smoke.py --backbone vit_base_patch16_224.mae --weights-path checkpoints/mae_pretrain_vit_base.pth --image-size 224 --output runs/pretrained_mae_smoke.json
```

The smoke uses synthetic labels. It checks finite optimization, adapter updates, the absence of encoder gradients, evaluation-mode normalization, and exact frozen-weight retention. It does not measure recognition quality. Pretrained loading failures are raised; random-backbone runs require an explicit smoke override.

## 4. Measure the 24GB configuration

```text
.venv/bin/python scripts/profile_cuda.py --config configs/cifar100_vit_24gb.json --output runs/grouping_memory.json
.venv/bin/python scripts/fetch_references.py --parax
.venv/bin/python scripts/profile_cuda.py --config configs/cifar100_parax_24gb.json --output runs/parax_memory.json
```

The classifier starts at 224x224 with microbatch 8. The profile measures warm optimizer-state memory as well as forward/backward. A CUDA-unavailable report is explicitly not a memory measurement. Our single post-backbone adapter runs the entire encoder under `no_grad`; full distributed ParaX requires gradients through frozen operations between its trainable blocks and can use more activation memory.

## 5. Run a matched pilot

```text
.venv/bin/python scripts/run_controls.py --config configs/cifar100_vit_24gb.json --output runs/cifar100_controls --seeds 42
.venv/bin/python scripts/train.py --config configs/cifar100_parax_24gb.json --output runs/cifar100_parax_seed42
```

The local modes are `linear`, `mlp`, `spatial`, `boundary`, `grouping`, `fixed_split`, `no_unique`, and `fixed_width`. Every non-linear local control has exactly the same learned projection shapes and parameter budget. ParaX retains its full 12-block official architecture, with a larger separately reported adapter budget. See `docs/BASELINES.md` before interpreting any comparison.

The CIFAR-100 pilot uses a stratified validation subset of the official training pool. Training never constructs the official test dataset. This 30-epoch AdamW pilot is not a reproduction of the ParaX paper's 100-epoch protocol. Use the same recipe for all compared methods, and tune each method using only validation data before making a performance claim.

For the specific cognitive prediction, train an orientation head with the sparse diagnostic configuration and evaluate unseen dense arrangements:

```text
.venv/bin/python scripts/train.py --config configs/crowding_vit_24gb.json --output runs/crowding_grouping_seed42
.venv/bin/python scripts/evaluate_crowding.py --run runs/crowding_grouping_seed42 --output runs/crowding_interventions.json --device cuda --precision bf16
```

This intervention report defaults to development seeds. Freeze all choices before adding `--allow-final-diagnostic` for the separate final diagnostic seed pool. It records paired trial predictions rather than only a pooled accuracy.

## 6. Resume and final evaluation

```text
.venv/bin/python scripts/train.py --config configs/cifar100_vit_24gb.json --output runs/cifar100_grouping_seed42 --epochs-this-invocation 1
.venv/bin/python scripts/train.py --config configs/cifar100_vit_24gb.json --output runs/cifar100_grouping_seed42 --resume
.venv/bin/python scripts/evaluate.py --run runs/cifar100_grouping_seed42 --split validation --device cuda --precision bf16
.venv/bin/python scripts/evaluate.py --run runs/cifar100_grouping_seed42 --split test --allow-test --device cuda --precision bf16 --predictions
```

`last.pt` is resumable; `best.pt` is chosen using validation accuracy only. Configurations, source hashes, encoder hashes, data roles, RNG states and parameter counts are recorded. Epoch-boundary resume is tested against uninterrupted training. Existing runs are not overwritten, and training cannot resume after explicit final-test access.

## Implementation stages

1. Numerical operator and controlled feature probes.
2. Single-module integration with a frozen pretrained encoder.
3. Correct training/checkpoint/data-role handling and matched controls.
4. Full official ParaX architecture in the harmonized harness.
5. Target-GPU memory profiling and real-image multi-seed experiments.

Stages 1-4 concern implementation and code-path validation. Stage 5 requires actual GPU and real-data evidence. `reports/VALIDATION.md` records what was really run.

Paper evidence and exact excerpts are in `docs/SOURCES.md`. The source-derived model, proposed 2D extension, and empirical claims are kept separate.

The ordered experiment gates and stopping criteria are in `docs/EXPERIMENTS.md`. To compare completed matched runs:

```text
.venv/bin/python scripts/summarize.py --runs runs/cifar100_controls/grouping_seed42 runs/cifar100_controls/boundary_seed42 --output runs/matched_validation.json
```

The delivered macOS environment uses system site packages for the already installed PyTorch runtime; it is not an isolated Linux/CUDA reproduction. A new Linux environment should use the setup command above.
