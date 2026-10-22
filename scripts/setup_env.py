"""Create an isolated Python environment; no shell scripts are generated."""

import argparse
from pathlib import Path
import subprocess
import sys
import venv


def main():
    """Install explicit PyTorch/timm versions without changing the caller's environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-system-site-packages", action="store_true")
    parser.add_argument("--torch-index-url", help="e.g. https://download.pytorch.org/whl/cu128")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / ".venv"
    if destination.exists():
        raise FileExistsError("environment already exists: " + str(destination))
    venv.EnvBuilder(with_pip=True, system_site_packages=args.use_system_site_packages).create(destination)
    python = destination / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools>=68", "wheel"], check=True)
    torch_command = [str(python), "-m", "pip", "install", "torch==2.8.0", "torchvision==0.23.0"]
    if args.torch_index_url:
        torch_command += ["--index-url", args.torch_index_url]
    subprocess.run(torch_command, check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-e", str(root) + "[baseline]"], check=True)
    snapshot = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (root / "environment.resolved.txt").write_text(snapshot, encoding="utf-8")
    print("Environment ready: " + str(python))


if __name__ == "__main__":
    main()
