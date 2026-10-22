"""Evaluate or summarize completed benchmarks without mixing datasets or recipes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from _common import PROJECT_ROOT


def main():
    """Require explicit final-test access and refuse incomplete comparison groups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    groups = {}
    for path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        if not manifest["config"].get("benchmark_protocol"):
            continue
        run = path.parent
        status = json.loads((run / "status.json").read_text())
        if not status["complete"]:
            raise ValueError("incomplete run: " + str(run))
        parent = run.parent.parent if manifest["config"].get("phase") == "confirmation" else run.parent
        groups.setdefault(parent, []).append((run, manifest))
    if not groups:
        raise ValueError("no completed benchmark runs found")
    for parent, entries in groups.items():
        if args.allow_test:
            for run, manifest in entries:
                if not (run / "test_evaluation.json").exists():
                    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/evaluate.py"), "--run", str(run),
                                    "--split", "test", "--allow-test", "--predictions", "--device", args.device,
                                    "--precision", manifest["config"]["precision"]], cwd=str(PROJECT_ROOT), check=True)
        command = [sys.executable, str(PROJECT_ROOT / "scripts/summarize.py"), "--runs"]
        command += [str(run) for run, _ in entries]
        command += ["--output", str(parent / ("final_test_summary.json" if args.allow_test else "validation_summary.json"))]
        if args.allow_test:
            command.append("--final-test")
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
