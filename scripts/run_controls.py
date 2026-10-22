"""Run identical local adapter/control protocols sequentially, using Python only."""

import argparse
from pathlib import Path
import subprocess
import sys

from _common import PROJECT_ROOT


def main():
    """Require distinct run directories and stop on the first failed condition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", nargs="+", default=["linear", "mlp", "spatial", "boundary", "grouping", "fixed_split", "no_unique", "fixed_width"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--allow-random-backbone", action="store_true")
    args = parser.parse_args()
    for seed in args.seeds:
        for mode in args.modes:
            output = Path(args.output).resolve() / (mode + "_seed" + str(seed))
            command = [sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"), "--config", str(Path(args.config).resolve()),
                       "--mode", mode, "--seed", str(seed), "--output", str(output)]
            if args.allow_random_backbone:
                command.append("--allow-random-backbone")
            subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    main()
