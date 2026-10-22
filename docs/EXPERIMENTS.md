# Ordered experiment gates

## 1. Verify the computation

Run `scripts/check.py`, `scripts/audit.py`, and `scripts/probe.py`. Review the analytic integral, finite-difference gradient, boundary, equivariance, identity, accumulation, and checkpoint tests. The feature probe should be interpreted as an operator observation. It cannot establish human alignment or recognition quality.

## 2. Verify actual pretrained weights

Run the ResNet and MAE smoke commands in README. Run `scripts/smoke_official.py` for the full official ParaX architecture. Confirm frozen hashes and trainable parameter counts before spending GPU time. The delivered MAE checkpoint has a download receipt and local SHA-256; no publisher checksum is claimed.

## 3. Measure target-GPU memory

Profile both ViT configurations before training. The proposed default is microbatch 8 with accumulation 8, effective batch 64 at 224 pixels. The profiler records allocated and reserved memory after optimizer-state warmup. If either condition exceeds the available GPU memory, reduce that condition's microbatch while preserving effective batch size, then profile again. Record the change rather than labeling the unmeasured configuration as 24GB verified.

## 4. Run the mechanism pilot

Train each local control on the same sparse orientation stimuli, then use `evaluate_crowding.py` on development stimuli. Change nearest-neighbor similarity while holding labels, jitter, contrast, and background noise fixed; separately change distant flanker proportions. Inspect paired predictions and isolated-target accuracy. A chance-level orientation classifier does not provide interpretable evidence about selective clutter sensitivity.

The proposed signature is better target recognition with reduced dependence on nearest-neighbor similarity, while retaining useful context. The source's nearest-neighbor premise is quoted exactly in SOURCES.md. Predicting a useful adapter from that premise is our extension. These central synthetic stimuli do not reproduce peripheral human crowding or eccentricity effects.

## 5. Run the real-image pilot

Use the CIFAR-100/MAE configurations and the same validation split, augmentation, optimizer, head initialization, and sample budget. Compare all local controls and full ParaX. The current 30-epoch defaults are a screening recipe, not a tuned leaderboard comparison. Record accuracy, NLL, trainable parameters, optimizer steps, wall time, and measured peak GPU memory. Use `summarize.py` to reject incompatible configurations before inspecting a ranking.

If grouping does not improve over the pairwise boundary and fixed-split controls, do not describe the result as evidence for individual/shared decomposition. If it wins only over a linear probe, the mechanism is not isolated. If gain disappears when placement or parameter budget is controlled, report that limitation.

## 6. Freeze and confirm

Tune each competitive method with a comparable validation-only search budget, including a strength check using ParaX's original readout/optimizer protocol. Freeze the method and hyperparameters; then repeat at least three seeds and evaluate final test once. The code requires explicit final-test access and records it. Add a second real-image dataset and a stronger pretrained CNN/ViT before making a broad claim.

Neither a positive synthetic probe nor a single CIFAR-100 run establishes CVPR-level novelty. An operator-level prior-art audit and stable real-data gains remain separate requirements.
