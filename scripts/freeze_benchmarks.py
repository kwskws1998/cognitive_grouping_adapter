"""Freeze method-specific hyperparameters using completed development validation only."""

import argparse
import json
from pathlib import Path
import statistics

from _common import PROJECT_ROOT
from cog_grouping.provenance import atomic_json, file_sha256, source_fingerprint


def freeze(root, output):
    """Require matched search budgets, no prior test access, and unchanged source."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite frozen choices")
    source = source_fingerprint(PROJECT_ROOT)
    datasets, recipes = {}, set()
    for path in sorted(Path(root).rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        config, run = manifest["config"], path.parent
        if not config.get("benchmark_protocol"):
            continue
        if config.get("phase", "development") != "development" or (run / "test_accessed.json").exists():
            raise ValueError("selection requires development runs with no final-test access")
        if manifest["source"] != source:
            raise ValueError("development source differs from current source")
        status = json.loads((run / "status.json").read_text())
        if not status["complete"]:
            raise ValueError("incomplete development run: " + str(run))
        recipes.add("reference" if "reference" in config["benchmark_protocol"] else "matched")
        best = max(json.loads((run / "history.json").read_text()),
                   key=lambda row: (row["validation"]["accuracy"], -row["validation"]["nll"]))
        candidate = {"run": str(run.resolve()), "config": config, "validation": best["validation"],
                     "selected_epoch": best["epoch"], "manifest_sha256": file_sha256(path)}
        datasets.setdefault(config["dataset"], {}).setdefault(config["mode"], []).append(candidate)
    if len(recipes) != 1:
        raise ValueError("freeze one recipe at a time and provide at least one completed run")
    choices, all_candidates = [], []
    for dataset, methods in datasets.items():
        budget = None
        protocol = None
        for mode, candidates in methods.items():
            grid = {(row["config"]["learning_rate"], row["config"]["seed"]) for row in candidates}
            if len(grid) != len(candidates):
                raise ValueError("duplicate development conditions")
            rates = {lr for lr, seed in grid}
            seeds = {seed for lr, seed in grid}
            if grid != {(lr, seed) for lr in rates for seed in seeds}:
                raise ValueError("every learning rate requires the same development seed set")
            if budget is not None and grid != budget:
                raise ValueError("methods have unequal learning-rate/seed search budgets: " + dataset)
            budget = grid
            groups = {}
            for row in candidates:
                signature = {key: value for key, value in row["config"].items()
                             if key not in ("mode", "learning_rate", "seed", "bottleneck", "grouping")}
                if protocol is not None and signature != protocol:
                    raise ValueError("incompatible development training protocols")
                protocol = signature
                groups.setdefault(row["config"]["learning_rate"], []).append(row)
            selected_lr = max(groups, key=lambda lr: (
                statistics.mean(row["validation"]["accuracy"] for row in groups[lr]),
                -statistics.mean(row["validation"]["nll"] for row in groups[lr])))
            choices.append({"dataset": dataset, "mode": mode, "config": groups[selected_lr][0]["config"],
                            "selected_learning_rate": selected_lr,
                            "selection_runs": [row["run"] for row in groups[selected_lr]]})
            all_candidates.extend(candidates)
    result = {"schema": 1, "recipe": recipes.pop(), "source": source, "selection_role": "development validation only",
              "choices": choices, "candidates": all_candidates, "final_test_accessed": False}
    atomic_json(output, result)
    return result


def main():
    """Save concrete immutable choices for full-training-pool confirmation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = freeze(args.root, args.output)
    print(json.dumps([{key: choice[key] for key in ("dataset", "mode", "selected_learning_rate")}
                      for choice in result["choices"]], indent=2))


if __name__ == "__main__":
    main()
