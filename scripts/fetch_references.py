"""Fetch pinned official ParaX Python sources and optional original CODE code.

Downloads are never executed by this script. Every fetched file is hashed. The
ParaX classification baseline must retain its official distributed placement;
it must not be relabeled as one of our post-backbone controls.
"""

import argparse
import json
from pathlib import Path
import urllib.request

from _common import PROJECT_ROOT
from cog_grouping.provenance import atomic_json, file_sha256

PARAX_REVISION = "a1c4aa049b1d859db287535f3597b113ed01a2a4"


def get(url):
    """Read a public resource with a bounded timeout and an identifiable user agent."""
    request = urllib.request.Request(url, headers={"User-Agent": "cognitive-grouping-research/0.1"})
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read()


def fetch_parax():
    """Preserve official Python sources and license at an immutable commit."""
    destination = PROJECT_ROOT / "third_party" / "ParaX"
    manifest_path = destination / "SOURCE_MANIFEST.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["revision"] != PARAX_REVISION:
            raise ValueError("existing ParaX revision differs")
        for path, sha in saved["files"].items():
            if file_sha256(destination / path) != sha:
                raise ValueError("modified reference file: " + path)
        print("Verified existing ParaX sources", flush=True)
        return
    tree = json.loads(get("https://api.github.com/repos/LMMMEng/ParaX/git/trees/" + PARAX_REVISION + "?recursive=1"))
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree was truncated")
    files = {}
    for entry in tree["tree"]:
        relative = entry["path"]
        accepted = relative in ("LICENSE", "README.md") or (
            relative.endswith(".py") and relative.startswith(("models/", "image_classification/")))
        if entry["type"] != "blob" or not accepted:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = get("https://raw.githubusercontent.com/LMMMEng/ParaX/" + PARAX_REVISION + "/" + relative)
        target.write_bytes(content)
        files[relative] = file_sha256(target)
        print("Fetched " + relative, flush=True)
    if "models/parax.py" not in files or "image_classification/main_image_parax.py" not in files or "LICENSE" not in files:
        raise RuntimeError("official ParaX source set is incomplete")
    atomic_json(manifest_path, {"repository": "https://github.com/LMMMEng/ParaX", "revision": PARAX_REVISION,
                                "files": files, "executed": False})


def fetch_code_reference():
    """Archive original model files for private inspection, preserving provenance."""
    url = "https://api.osf.io/v2/nodes/tjcbu/files/osfstorage/656d932443d06718c3b19e2c/"
    metadata = json.loads(get(url))
    destination = PROJECT_ROOT / "work" / "original_code_reference"
    files = {}
    for entry in metadata["data"]:
        name = entry["attributes"]["name"]
        if entry["attributes"]["kind"] != "file" or not name.endswith(".py"):
            continue
        if Path(name).name != name:
            raise ValueError("unexpected path in OSF model folder")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(get(entry["links"]["download"]))
        files[name] = {"sha256": file_sha256(target), "download": entry["links"]["download"]}
        print("Fetched original reference " + name, flush=True)
    atomic_json(destination / "SOURCE_MANIFEST.json", {"osf_project": "https://osf.io/tjcbu/", "files": files,
                                                       "executed": False, "license_checked": False})


def main():
    """Fetch only explicitly requested sources, without running external code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parax", action="store_true")
    parser.add_argument("--code-reference", action="store_true")
    args = parser.parse_args()
    if not args.parax and not args.code_reference:
        parser.error("choose --parax and/or --code-reference")
    if args.parax:
        fetch_parax()
    if args.code_reference:
        fetch_code_reference()


if __name__ == "__main__":
    main()
