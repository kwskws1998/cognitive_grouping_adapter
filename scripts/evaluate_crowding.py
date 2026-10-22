"""Evaluate paired sparse/dense and neighbor-similarity interventions."""

import argparse
from pathlib import Path

from _common import PROJECT_ROOT
import torch
from torch.utils.data import DataLoader

from cog_grouping.backbones import make_backbone
from cog_grouping.data import CrowdingDataset, make_transforms
from cog_grouping.engine import evaluate
from cog_grouping.model import build_classifier
from cog_grouping.provenance import atomic_json, seed_everything, source_fingerprint, state_fingerprint


def main():
    """Use development stimuli by default and mark final diagnostic access explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--allow-final-diagnostic", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite an intervention report")
    saved = torch.load(Path(args.run) / "best.pt", map_location="cpu", weights_only=True)
    manifest = saved["manifest"]
    config = manifest["config"]
    if config["dataset"] != "crowding" or manifest["num_classes"] != 2:
        raise ValueError("this probe requires an orientation-classification diagnostic checkpoint")
    if manifest["source"] != source_fingerprint(PROJECT_ROOT):
        raise ValueError("source differs from training")
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(config.get("threads", 4))
    backbone = make_backbone(config["backbone"], config.get("pretrained", True), config.get("weights_path"))
    if state_fingerprint(backbone) != manifest["backbone_sha256"]:
        raise ValueError("encoder differs from training")
    model = build_classifier(backbone, 2, config, PROJECT_ROOT)
    model.load_trainable_state_dict(saved["model"])
    device = torch.device(args.device)
    model.to(device)
    transform = make_transforms(config.get("image_size", 224), **backbone.data_config)
    role = "final diagnostic" if args.allow_final_diagnostic else "development diagnostic"
    offset = 200000000 if args.allow_final_diagnostic else 100000000
    if args.allow_final_diagnostic:
        atomic_json(Path(args.run) / "test_accessed.json", {"role": role, "selected_epoch": saved["epoch"]})
    observations = []
    conditions = [("alone", "similar", 0.5)] + [(density, neighbors, proportion)
                  for density in ("sparse", "dense") for neighbors in ("similar", "dissimilar")
                  for proportion in ((0., 0.5, 1.) if density == "dense" else (0.5,))]
    for density, neighbors, proportion in conditions:
        dataset = CrowdingDataset(size=args.samples, image_size=config.get("image_size", 224),
                                  seed=config.get("split_seed", 1729) + offset, density=density,
                                  neighbors=neighbors, distant_similarity=proportion, transform=transform)
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config.get("workers", 0))
        result = evaluate(model, loader, device, args.precision, return_predictions=True)
        result.update({"density": density, "nearest_neighbors": neighbors, "distant_similarity": proportion})
        observations.append(result)
        print({key: value for key, value in result.items() if key != "predictions"}, flush=True)
    atomic_json(output, {"role": role, "selected_epoch": saved["epoch"], "observations": observations,
                         "human_fit_established": False, "natural_image_efficacy_established": False})


if __name__ == "__main__":
    main()
