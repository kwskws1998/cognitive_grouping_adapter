"""Exercise real pretrained training/evaluation on explicit tiny real-image subsets."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from _common import PROJECT_ROOT, read_config
from setup_env import active_conda_python
from cog_grouping.provenance import atomic_json, environment


def main():
    """Check integration and frozen weights without claiming benchmark performance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark/cifar100.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--modes", nargs="+", default=["grouping", "official_parax"])
    args = parser.parse_args()
    active_conda_python()
    base = read_config(args.config)
    base.update({"phase": "development", "validation_fraction": 0.1, "epochs": 2, "warmup_epochs": 0,
                 "max_train_samples": 16, "max_val_samples": 8, "batch_size": 2, "accumulation_steps": 2,
                 "workers": 0, "threads": 2, "device": args.device, "precision": args.precision,
                 "benchmark_protocol": "real_image_code_path_smoke_v1"})
    if not base.get("pretrained", True) or base["dataset"] == "crowding":
        raise ValueError("real pretrained encoders and real benchmark images are required")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh validation output directory")
    rows = []
    for mode in args.modes:
        config = dict(base, mode=mode)
        config_path, run = output / (mode + ".config.json"), output / mode
        atomic_json(config_path, config)
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/train.py"), "--config", str(config_path),
                        "--output", str(run)], cwd=str(PROJECT_ROOT), check=True)
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/evaluate.py"), "--run", str(run),
                        "--device", args.device, "--precision", args.precision], cwd=str(PROJECT_ROOT), check=True)
        status = json.loads((run / "status.json").read_text())
        history = json.loads((run / "history.json").read_text())
        if not status["complete"] or not status["backbone_unchanged"] or (run / "test_accessed.json").exists():
            raise RuntimeError("real-image integration protocol check failed")
        if any(row["train"]["samples"] != 16 or row["validation"]["samples"] != 8 for row in history):
            raise RuntimeError("real-image smoke used unexpected sample counts")
        rows.append({"mode": mode, "run": str(run), "status": status,
                     "validation": json.loads((run / "validation_evaluation.json").read_text())})
    atomic_json(output / "validation_report.json", {"dataset": base["dataset"], "real_images": True,
                "pretrained": True, "environment": environment(), "rows": rows,
                "classification_efficacy_established": False})


if __name__ == "__main__":
    main()
