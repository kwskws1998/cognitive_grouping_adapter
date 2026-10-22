"""Download and validate benchmark training pools without constructing test datasets."""

import argparse

from _common import PROJECT_ROOT, read_config
from cog_grouping.data import make_datasets
from cog_grouping.provenance import atomic_json


def main():
    """Record actual class/sample counts and the exact development split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--cifar100-mirror-url", help="optional archive mirror; torchvision still enforces the official MD5")
    args = parser.parse_args()
    config = read_config(args.config)
    if config["dataset"] not in ("cifar100", "svhn", "food101", "imagenet-r"):
        parser.error("use a real-image benchmark configuration")
    normalization = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
    if args.cifar100_mirror_url:
        if config["dataset"] != "cifar100":
            parser.error("the CIFAR-100 archive mirror only applies to CIFAR-100")
        from torchvision.datasets import CIFAR100
        CIFAR100.url = args.cifar100_mirror_url
    training, validation, test, classes, roles = make_datasets(config, normalization, include_test=False)
    if test is not None:
        raise RuntimeError("data preparation unexpectedly constructed final test")
    report = {"dataset": config["dataset"], "classes": classes, "train_samples": len(training),
              "validation_samples": len(validation) if validation is not None else None,
              "final_test_constructed": False, "data_roles": roles}
    if args.cifar100_mirror_url:
        from pathlib import Path
        from cog_grouping.provenance import file_sha256
        archive = Path(config.get("data_root", "data")) / CIFAR100.filename
        report["archive"] = {"requested_mirror_url": args.cifar100_mirror_url,
                             "official_md5_checked_by_torchvision": CIFAR100.tgz_md5,
                             "sha256": file_sha256(archive) if archive.exists() else None}
    output = args.output or PROJECT_ROOT / "runs" / "prepared" / (config["dataset"] + ".json")
    atomic_json(output, report)
    print({key: value for key, value in report.items() if key != "data_roles"})


if __name__ == "__main__":
    main()
