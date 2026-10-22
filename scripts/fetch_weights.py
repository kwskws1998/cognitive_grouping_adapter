"""Fetch official MAE pretraining weights and record their observed SHA-256."""

import argparse
import json
import os
import urllib.request

from _common import PROJECT_ROOT
from cog_grouping.provenance import atomic_json, file_sha256

MAE_URL = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"


def main():
    """Download atomically; a stored hash detects later local file modification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    destination = PROJECT_ROOT / "checkpoints" / "mae_pretrain_vit_base.pth"
    record = destination.with_suffix(".source.json")
    if destination.exists():
        digest = file_sha256(destination)
        if not record.exists() or json.loads(record.read_text())["sha256"] != digest:
            raise ValueError("existing MAE weights lack matching provenance")
        if args.expected_sha256 and digest != args.expected_sha256:
            raise ValueError("checkpoint differs from requested SHA-256")
        print("Verified existing MAE checkpoint: " + digest)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    try:
        request = urllib.request.Request(MAE_URL, headers={"User-Agent": "cognitive-grouping-research/0.1"})
        size = 0
        with urllib.request.urlopen(request, timeout=60) as response, open(temporary, "wb") as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
                size += len(block)
                if size % (50 * 1024 * 1024) == 0:
                    print("Downloaded %d MiB" % (size // (1024 * 1024)), flush=True)
        digest = file_sha256(temporary)
        if args.expected_sha256 and digest != args.expected_sha256:
            raise ValueError("downloaded checkpoint differs from requested SHA-256")
        os.replace(temporary, destination)
        atomic_json(record, {"url": MAE_URL, "sha256": digest, "bytes": size,
                             "hash_kind": "observed download hash; no publisher digest asserted"})
        print("Saved MAE checkpoint: " + digest)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
