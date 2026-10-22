"""Summarize matched local runs without mixing paper scores or incompatible splits."""

import argparse
import json
import statistics
from pathlib import Path

from _common import PROJECT_ROOT
from cog_grouping.provenance import atomic_json


def main():
    """Reject incomplete or incompatible comparisons before ranking any method."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    reference = None
    rows = []
    for directory in args.runs:
        run = Path(directory)
        manifest = json.loads((run / "manifest.json").read_text())
        status = json.loads((run / "status.json").read_text())
        if not status["complete"]:
            raise ValueError("incomplete run: " + directory)
        config = manifest["config"]
        method_keys = {"mode", "seed", "bottleneck", "grouping", "weights_path"}
        if config.get("phase") == "confirmation" and config.get("frozen_plan_sha256"):
            method_keys.add("learning_rate")
        protocol = {key: value for key, value in config.items() if key not in method_keys}
        signature = {"encoder": manifest["backbone_sha256"], "split": manifest["split_sha256"],
                     "normalization": manifest["normalization"], "source": manifest["source"]["sha256"], "protocol": protocol}
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("incompatible encoder/split/source/training protocol: " + directory)
        if args.final_test:
            result = json.loads((run / "test_evaluation.json").read_text())
        else:
            if config.get("phase") == "confirmation":
                raise ValueError("confirmation has no validation ranking; evaluate final test after freezing settings")
            history = json.loads((run / "history.json").read_text())
            result = max(history, key=lambda entry: (entry["validation"]["accuracy"], -entry["validation"]["nll"]))["validation"]
        row = {"run": directory, "mode": config.get("mode", "grouping"), "seed": config.get("seed", 42),
               "accuracy": result["accuracy"], "nll": result["nll"], "samples": result["samples"],
               "adapter_parameters": manifest["parameters"]["adapter_trainable"],
               "head_parameters": manifest["parameters"]["head_trainable"]}
        row["learning_rate"] = config.get("learning_rate", 1e-3)
        row["top5_accuracy"] = result.get("top5_accuracy")
        row["method_configuration"] = {key: config.get(key) for key in ("bottleneck", "grouping")}
        if config.get("mode") == "official_parax":
            row["method_configuration"]["parax"] = config.get("parax")
        history = json.loads((run / "history.json").read_text())
        row["optimizer_steps"] = sum(epoch["train"]["optimizer_steps"] for epoch in history)
        row["training_seconds"] = sum(epoch["train"]["seconds"] for epoch in history)
        measured = [epoch["train"]["peak_cuda_allocated_bytes"] for epoch in history
                    if epoch["train"]["peak_cuda_allocated_bytes"] is not None]
        row["peak_cuda_allocated_bytes"] = max(measured) if measured else None
        rows.append(row)
    rows.sort(key=lambda row: (-row["accuracy"], row["nll"]))
    groups = {}
    for row in rows:
        key = json.dumps([row["mode"], row["method_configuration"]], sort_keys=True)
        groups.setdefault(key, []).append(row)
    aggregates = []
    for group in groups.values():
        if len({row["seed"] for row in group}) != len(group):
            raise ValueError("duplicate seeds for the same method configuration")
        values = [row["accuracy"] for row in group]
        aggregates.append({"mode": group[0]["mode"], "method_configuration": group[0]["method_configuration"],
                           "seeds": sorted(row["seed"] for row in group), "n_seeds": len(group),
                           "mean_accuracy": statistics.mean(values),
                           "std_accuracy": statistics.stdev(values) if len(values) > 1 else None,
                           "mean_nll": statistics.mean(row["nll"] for row in group)})
    aggregates.sort(key=lambda row: (-row["mean_accuracy"], row["mean_nll"]))
    report = {"role": "final test" if args.final_test else "development validation", "comparison_signature": reference,
              "rows": rows, "seed_statistics": aggregates,
              "published_scores_included": False, "method_superiority_established": False}
    atomic_json(args.output, report)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
