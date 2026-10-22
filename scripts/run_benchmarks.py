"""Run real-image comparisons in conda, keeping development and confirmation separate."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from _common import PROJECT_ROOT, read_config
from setup_env import active_conda_python
from cog_grouping.provenance import atomic_json, file_sha256, source_fingerprint

LOCAL_MODES = ["linear", "mlp", "spatial", "boundary", "grouping", "fixed_split", "no_unique", "fixed_width"]


def build_plan(datasets, modes, seeds, recipe, phase, learning_rates, output):
    """Resolve every compared condition before launching any training process."""
    if len(set(seeds)) != len(seeds) or len(set(modes)) != len(modes) or len(set(datasets)) != len(datasets):
        raise ValueError("duplicate datasets, modes or seeds are not allowed")
    if recipe == "reference" and modes != ["official_parax"]:
        raise ValueError("the reference recipe is reserved for official_parax")
    plan = []
    for dataset in datasets:
        folder = PROJECT_ROOT / "configs" / "benchmark"
        if recipe == "reference":
            folder = folder / "reference"
        base = read_config(folder / (dataset + ".json"))
        for lr in learning_rates or [base["learning_rate"]]:
            if lr <= 0:
                raise ValueError("learning rates must be positive")
            for seed in seeds:
                for mode in modes:
                    config = json.loads(json.dumps(base))
                    config.update({"mode": mode, "seed": seed, "learning_rate": lr, "phase": phase})
                    if phase == "confirmation":
                        config["validation_fraction"] = 0.0
                    run = Path(output).resolve() / recipe / phase / dataset / ("lr" + format(lr, ".9g")) / (mode + "_seed" + str(seed))
                    plan.append({"config": config, "run": run})
    if len({entry["run"] for entry in plan}) != len(plan):
        raise ValueError("conditions map to duplicate run directories")
    return plan


def main():
    """Stop on failure and resume only runs with matching source and configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("cifar100", "svhn", "food101", "imagenet-r"))
    parser.add_argument("--modes", nargs="+", choices=LOCAL_MODES + ["official_parax"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--recipe", choices=("matched", "reference"), default="matched")
    parser.add_argument("--phase", choices=("development", "confirmation"), default="development")
    parser.add_argument("--learning-rates", nargs="+", type=float)
    parser.add_argument("--output", default="runs/benchmark_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs-this-invocation", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--frozen-plan")
    args = parser.parse_args()
    if args.epochs_this_invocation is not None and args.epochs_this_invocation < 1:
        parser.error("--epochs-this-invocation must be positive")
    if args.phase == "confirmation" and not args.frozen_plan:
        parser.error("confirmation requires --frozen-plan from validation-only selection")
    if args.frozen_plan:
        if args.phase != "confirmation" or args.learning_rates:
            parser.error("a frozen plan requires confirmation and fixes learning rates")
        frozen = json.loads(Path(args.frozen_plan).read_text())
        if frozen["source"] != source_fingerprint(PROJECT_ROOT):
            raise ValueError("source changed since validation choices were frozen")
        digest = file_sha256(args.frozen_plan)
        plan = []
        for choice in frozen["choices"]:
            config = choice["config"]
            if args.datasets and config["dataset"] not in args.datasets:
                continue
            if args.modes and config["mode"] not in args.modes:
                continue
            for seed in args.seeds:
                selected = json.loads(json.dumps(config))
                selected.update({"seed": seed, "phase": "confirmation", "validation_fraction": 0.0,
                                 "frozen_plan_sha256": digest})
                run = Path(args.output).resolve() / frozen["recipe"] / "confirmation" / config["dataset"] / \
                      ("lr" + format(config["learning_rate"], ".9g")) / (config["mode"] + "_seed" + str(seed))
                plan.append({"config": selected, "run": run})
        if not plan or len({entry["run"] for entry in plan}) != len(plan):
            raise ValueError("frozen plan has no selected conditions or duplicate seeds")
    else:
        modes = args.modes or (["official_parax"] if args.recipe == "reference" else LOCAL_MODES + ["official_parax"])
        plan = build_plan(args.datasets or ["cifar100"], modes, args.seeds, args.recipe, args.phase, args.learning_rates, args.output)
    if args.dry_run:
        for entry in plan:
            print(json.dumps({"run": str(entry["run"]), "config": entry["config"]}))
        return
    active_conda_python()
    source = source_fingerprint(PROJECT_ROOT)
    for entry in plan:
        run, config = entry["run"], entry["config"]
        config_path = run.parent / (run.name + ".config.json")
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("saved launch configuration differs: " + str(config_path))
        if run.exists() and any(run.iterdir()):
            if not args.resume:
                raise FileExistsError("run already exists; use --resume: " + str(run))
            manifest = json.loads((run / "manifest.json").read_text())
            if manifest["source"] != source or manifest["config"] != config:
                raise ValueError("run source or configuration changed: " + str(run))
            status_path = run / "status.json"
            if status_path.exists() and json.loads(status_path.read_text())["complete"]:
                print("Completed run retained: " + str(run), flush=True)
                continue
            resume = True
        else:
            resume = False
        atomic_json(config_path, config)
        command = [sys.executable, str(PROJECT_ROOT / "scripts/train.py"), "--config", str(config_path), "--output", str(run)]
        if resume:
            command.append("--resume")
        if args.epochs_this_invocation is not None:
            command += ["--epochs-this-invocation", str(args.epochs_this_invocation)]
        print("Starting " + str(run), flush=True)
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
