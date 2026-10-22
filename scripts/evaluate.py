"""Explicit held-out evaluation of a validation-selected or fixed-final checkpoint."""

import argparse
import json
from pathlib import Path

from _common import PROJECT_ROOT
import torch
from torch.utils.data import DataLoader

from cog_grouping.backbones import make_backbone
from cog_grouping.data import make_datasets
from cog_grouping.engine import evaluate
from cog_grouping.grouping import GroupingConfig
from cog_grouping.model import build_classifier
from cog_grouping.provenance import atomic_json, file_sha256, seed_everything, source_fingerprint, state_fingerprint


def main():
    """Never treat final test as another checkpoint-selection loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="fp32", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--predictions", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("final test requires --allow-test after method selection is complete")
    run = Path(args.run)
    status = json.loads((run / "status.json").read_text())
    if args.split == "test" and not status["complete"]:
        raise ValueError("final test requires a completed training run")
    result_path = run / (args.split + "_evaluation.json")
    if result_path.exists():
        raise FileExistsError("evaluation already recorded: " + str(result_path))
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    manifest, config = saved["manifest"], saved["manifest"]["config"]
    if args.split == "validation" and config.get("phase") == "confirmation":
        raise ValueError("confirmation uses the full training pool and has no validation split")
    if manifest["source"] != source_fingerprint(PROJECT_ROOT):
        raise ValueError("source changed since training; use a separately versioned evaluation")
    torch.set_num_threads(config.get("threads", 4))
    seed_everything(config.get("seed", 42))
    backbone = make_backbone(config["backbone"], config.get("pretrained", True), config.get("weights_path"))
    if state_fingerprint(backbone) != manifest["backbone_sha256"]:
        raise ValueError("pretrained encoder does not match the training run")
    if args.split == "test":
        atomic_json(run / "test_accessed.json", {"selected_epoch": saved["epoch"], "status": "opening final-test data"})
    _, validation, test, classes, split = make_datasets(config, backbone.data_config, include_test=args.split == "test")
    if split["sha256"] != manifest["split_sha256"]:
        raise ValueError("data split does not match the training run")
    model = build_classifier(backbone, classes, config, PROJECT_ROOT)
    model.load_trainable_state_dict(saved["model"])
    device = torch.device(args.device)
    model.to(device)
    dataset = test if args.split == "test" else validation
    if args.split == "test":
        atomic_json(run / "test_accessed.json", {"selected_epoch": saved["epoch"], "split_sha256": split["sha256"]})
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config.get("workers", 0))
    result = evaluate(model, loader, device, args.precision, return_predictions=args.predictions)
    result.update({"split": args.split, "selected_epoch": saved["epoch"], "backbone_sha256": manifest["backbone_sha256"]})
    result.update({"checkpoint_sha256": file_sha256(run / "best.pt"), "source_sha256": manifest["source"]["sha256"]})
    atomic_json(result_path, result)
    print({key: value for key, value in result.items() if key != "predictions"})


if __name__ == "__main__":
    main()
