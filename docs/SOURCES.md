# Source evidence and provenance

## Cognitive source

Van der Burg, Cass & Olivers (2024), *A CODE model bridging crowding in sparse and dense displays*, Vision Research 215, 108345. DOI: https://doi.org/10.1016/j.visres.2023.108345

Exact excerpts from the abstract, printed page 1: **“grouping by proximity and similarity”**, **“nearest neighbour rule”**, and **“fail to segment”**. These are the source's mechanisms and segmentation premise; an accuracy benefit from preserving two adapter channels is our hypothesis.

Full paper: https://pure.uva.nl/ws/files/256296258/A_CODE_model_bridging_crowding_in_sparse_and_dense_displays.pdf

Published code: https://osf.io/tjcbu/ ; model download: https://osf.io/download/casmu/

The paper combines proximity, local feature similarity and segmentation strength. Its equations 1-3 and Figure 2 motivate the operator. The author's `CODE_model.py` was inspected locally. In that code the above-threshold numerator uses the summed activity landscape, which resolves an ambiguity in a literal target-profile reading of the equation. Our implementation is not a numerical reproduction of that program; see METHOD.md for departures.

The OSF code is only fetched into `work/original_code_reference` for private source review. This project does not assert a redistribution license for that code. The baseline source is separately licensed.

## Competitive architecture

Lou, Yu & Yu (2026), *Parameters as Experts: Adapting Vision Models with Dynamic Parameter Routing*.

Paper: https://arxiv.org/abs/2602.06862v2

Exact excerpts from the abstract: **“shared expert centers”** and **“each expert is a trainable parameter matrix”**.

Official source and license: https://github.com/LMMMEng/ParaX ; Apache-2.0.

The official architecture is integrated through pinned, hashed source. Its layout and the local harness differences are specified in BASELINES.md. No score from the paper is presented as an observed result of this repository.

## Pretraining and libraries

MAE source: https://github.com/facebookresearch/mae

MAE ViT-B pretraining checkpoint: https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth

timm source: https://github.com/huggingface/pytorch-image-models

torchvision weights: https://docs.pytorch.org/vision/stable/models.html

Checkpoint hashes in this project are observed local file/state hashes. A publisher-provided checksum is not implied unless explicitly supplied during download.
