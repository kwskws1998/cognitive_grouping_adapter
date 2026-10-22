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

The bridge retains all 12 official ParaX blocks and the shared parameter experts. It does not move ParaX to a single output location. The classification bridge disables the optional spatial-convolution path and uses rank 192, reporting the actual trainable parameter count.

MAE/timm checkpoints pack Q/K/V together and nest MLP layers. The official class uses separate Q/K/V names and flattened MLP names. The bridge explicitly translates them and rejects unexpected/missing frozen keys. It must not initialize missing attention weights randomly and train them under the name of an adapter. An independent tiny-model test verifies the translated frozen core against timm when ParaX updates are neutralized.

## Harmonized pilot, not published-score reproduction

For isolation, the bridge uses the same normalized patch-mean linear readout as our local methods, the same training-pool validation split, the same optimizer and the same augmentation. The published official trainer has its own readout, optimizer, preprocessing and evaluation recipe. We do not claim published scores were reproduced, and paper-table numbers must not be placed beside our pilot numbers as if matched.

Preserve a separate official-protocol reproduction when reporting strong-benchmark results. Before a paper claim, tune the harmonized ParaX baseline with a comparable validation search budget; add the original readout/SGD recipe as a strength check. Our current pilot defaults alone do not establish superiority over the strongest tuned ParaX condition.

The single adapter and distributed ParaX differ in placement, parameter count and activation-memory needs. Report all three rather than attributing a difference only to cognition.
