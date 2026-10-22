# Baseline contract

## Locally matched controls

All non-linear local modes use the same LayerNorm, `C -> D` projection, three `D`-channel inputs to the up-projection, and `3D -> C` projection. They start at the same identity function and differ in their deterministic middle computation.

The training harness resets every task head using a separate, identical seed. It also resets the training RNG after constructing the model, so a different amount of initialization randomness cannot change the initial head or augmentation stream.

| Mode | Middle computation | Purpose |
|---|---|---|
| linear | no adapter | encoder-only lower bound |
| mlp | duplicated pointwise channels | capacity without spatial context |
| spatial | uniform spatial mean, fixed split | context without boundaries |
| boundary | pairwise similarity aggregation, contrast-based split | edge-aware alternative without activity landscapes |
| fixed_split | full context, constant 0.5 split | remove context-dependent decomposition |
| no_unique | suppress unique path | remove the proposed evidence route |
| fixed_width | constant spread for valid neighbors | remove neighbor-conditioned width |
| grouping | full operator | candidate 1 |

The no-unique intervention is an ablation, not a parameter-efficient competitive method. Fixed-split retains the original amplitude information and is necessary alongside it.

## Official ParaX architecture

Official source: https://github.com/LMMMEng/ParaX

Pinned commit: `a1c4aa049b1d859db287535f3597b113ed01a2a4`.

`scripts/fetch_references.py --parax` downloads immutable Python sources and the Apache-2.0 license, records SHA-256 per file, and does not execute them. The bridge checks hashes before import. Vendor code is not restyled or edited.

The bridge retains all 12 official ParaX blocks and shared parameter experts. It does not move ParaX to a single output location. The primary benchmark now uses rank 128 and enables convolution kernels 3/5/7, matching the checked-in official classification launchers. The old rank-192, convolution-disabled settings remain available only through legacy pilot configurations. All budgets are counted from the constructed model.

MAE/timm checkpoints pack Q/K/V together and nest MLP layers. The official class uses separate Q/K/V names and flattened MLP names. The bridge explicitly translates them and rejects unexpected/missing frozen keys. It must not initialize missing attention weights randomly and train them under the name of an adapter. An independent tiny-model test verifies the translated frozen core against timm when ParaX updates are neutralized.

## Matched and reference benchmark recipes

The matched recipe uses the same normalized patch-mean linear readout, training-pool validation split, SGD and official-source preprocessing for all methods. The separate reference recipe retains the official CLS + BatchNorm head, FP32, SGD, convolution paths and source stochastic-layer behavior. Exact settings and differences are recorded in BENCHMARK_PROTOCOL.md. Published scores must not be placed beside observed runs as if matched.

Tune each method with the same validation-only LR/seed search grid. Freezing rejects unequal budgets and prior final-test access. The reference recipe is a single-GPU strength check: gradient accumulation preserves effective batch but does not reproduce the official per-device BatchNorm statistics. Exact multi-GPU score reproduction remains a separate experiment.

The single adapter and distributed ParaX differ in placement, parameter count and activation-memory needs. Report all three rather than attributing a difference only to cognition.
