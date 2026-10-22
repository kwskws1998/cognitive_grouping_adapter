# Implementation validation

Verified on 2026-10-22 in the local macOS ARM64 environment. This is implementation evidence, not a trained-method result.

| Check | Observed result | Artifact |
|---|---|---|
| Unit, numerical and CLI tests | 32 passed, 0 failed, 0 skipped | unit_tests.json |
| Syntax, comment policy and configs | 28 Python files, 6 configs; passed | source_audit.json |
| Real pretrained ResNet-18 | Three optimizer steps; frozen hash unchanged; both adapter projections updated | pretrained_resnet18.json |
| Real pretrained MAE ViT-B | Three optimizer steps; frozen hash unchanged; both adapter projections updated | pretrained_mae.json |
| Full official ParaX | All 12 blocks retained; frozen core unchanged; expert parameters updated; finite gradients | official_parax.json |
| CPU train/evaluate orchestration | Two epochs on 16 synthetic training and 8 validation images; selected validation accuracy 0.50 | ../runs/cpu_validation/ |
| Synthetic intervention command | Nine paired development conditions, 8 images each; all accuracy 0.50 | cpu_interventions.json |
| CUDA profile | Unavailable; no CUDA device | cuda_memory.json |

The CPU classifier uses an explicitly random encoder and tiny synthetic data. Its chance accuracy is retained as a negative result; it demonstrates orchestration only. The pretrained smokes use synthetic images/labels and establish valid updates, not recognition efficacy.

## Actual adapter budgets

With bottleneck 64, the proposed adapter has 132,672 trainable parameters after ResNet-18, and 198,976 after MAE ViT-B. The three-class smoke heads have 1,539 and 2,307 parameters respectively. Full ParaX at rank 192 has 4,200,672 adapter parameters in the pinned official code plus the 2,307-parameter smoke head. A CIFAR-100 head has more parameters; the harness records it for each run. These are observed code counts, not copied paper-table budgets.

## Mechanism observation

The controlled-feature probe gives central private fractions of 1.000000 for an isolated source, 0.013747 for near/far similar descriptors, 0.100969 for different nearest neighbors with similar distant descriptors, and 0.021288 for similar nearest/different distant descriptors. Far context still has a nonzero effect in this operator. This is not a fit to human psychophysics, and the cognitive interpretation remains a hypothesis.

## Protocol checks

Tests verify the analytic three-item integral, finite-difference gradients, no wraparound, edge normalization, rotation equivariance, amplitude scaling, initial identity across local controls, finite gradients, frozen normalization buffers, sample-correct gradient accumulation, safe checkpoint round trips, exact epoch-boundary resume, and a strict timm-to-official checkpoint mapping. CLI coverage includes development interventions, matching/rejecting comparison summaries, explicit final-test access, and prevention of training after access. Synthetic test access occurs inside temporary test fixtures; the retained CPU validation run has not accessed its final test pool.

## Runtime and source

Observed Python 3.9.6; PyTorch 2.8.0; torchvision 0.23.0; timm 1.0.22; NumPy 2.0.2; Pillow 11.3.0. CUDA is unavailable. The delivered macOS virtual environment uses system site packages for the existing PyTorch runtime.

Source fingerprint: `0ba8a46490bdc7bd05222657551e7f8601446368e74273c9cbfef3b0b23acaea`. Source hashes exclude documentation, dependencies and run artifacts. The official baseline has its own pinned revision and file hashes. MAE checkpoint SHA-256: `aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d` (observed download hash).

## Work that still requires experiments

No target-GPU peak memory or 24GB fit has been measured. No real-image training, multi-seed comparison, human-data fit, recognition superiority, published-score reproduction, or novelty certification is claimed. An optional CIFAR-100 download was stopped because the server estimated roughly 40 minutes; no partial dataset is delivered. The pretrained ResNet and MAE integrations were tested successfully with actual weights. Follow docs/EXPERIMENTS.md for the remaining experiment gates.
