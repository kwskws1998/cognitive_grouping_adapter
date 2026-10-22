"""Summarize matched local runs without mixing paper scores or incompatible splits."""

import argparse
import json
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
            history = json.loads((run / "history.json").read_text())
            result = max(history, key=lambda entry: entry["validation"]["accuracy"])["validation"]
        row = {"run": directory, "mode": config.get("mode", "grouping"), "seed": config.get("seed", 42),
               "accuracy": result["accuracy"], "nll": result["nll"], "samples": result["samples"],
               "adapter_parameters": manifest["parameters"]["adapter_trainable"],
               "head_parameters": manifest["parameters"]["head_trainable"]}
        row["method_configuration"] = {key: config.get(key) for key in ("bottleneck", "grouping")}
        rows.append(row)
    rows.sort(key=lambda row: (-row["accuracy"], row["nll"]))
    report = {"role": "final test" if args.final_test else "development validation", "comparison_signature": reference,
              "rows": rows, "published_scores_included": False, "method_superiority_established": False}
    atomic_json(args.output, report)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
