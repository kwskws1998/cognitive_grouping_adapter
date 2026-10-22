"""Install project dependencies into the active conda environment."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def active_conda_python():
    """Reject inactive conda environments and interpreters from a different prefix."""
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        raise RuntimeError("Activate a conda environment first: conda activate cognitive_grouping")
    destination = Path(prefix).resolve()
    if not (destination / "conda-meta").is_dir():
        raise RuntimeError("CONDA_PREFIX does not identify a conda environment")
    if Path(sys.prefix).resolve() != destination:
        raise RuntimeError("The running Python does not belong to the active conda environment")
    return sys.executable


def main(argv=None):
    """Install pinned packages using only the active conda interpreter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-index-url", help="e.g. https://download.pytorch.org/whl/cu128")
    args = parser.parse_args(argv)
    try:
        python = active_conda_python()
    except RuntimeError as error:
        parser.error(str(error))
    root = Path(__file__).resolve().parents[1]
    subprocess.run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools>=68", "wheel"], check=True)
    torch_command = [python, "-m", "pip", "install", "torch==2.8.0", "torchvision==0.23.0"]
    if args.torch_index_url:
        torch_command += ["--index-url", args.torch_index_url]
    subprocess.run(torch_command, check=True)
    subprocess.run([python, "-m", "pip", "install", "-e", str(root) + "[baseline]"], check=True)
    snapshot = subprocess.check_output([python, "-m", "pip", "freeze"], text=True)
    (root / "environment.resolved.txt").write_text(snapshot, encoding="utf-8")
    print("Environment ready: " + str(python))


if __name__ == "__main__":
    main()
