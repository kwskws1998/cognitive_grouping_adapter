"""Inspect controlled feature-level grouping without training or human-fit claims."""

import argparse
from dataclasses import asdict
from pathlib import Path

from _common import PROJECT_ROOT
import torch
from PIL import Image, ImageDraw

from cog_grouping.grouping import DirectionalGrouping, GroupingConfig
from cog_grouping.provenance import atomic_json


def fixture(neighbors, distant, isolated=False):
    """Keep evidence amplitudes fixed while changing the local descriptor relation."""
    amplitude = torch.zeros(1, 1, 9, 9)
    descriptors = torch.zeros(1, 2, 9, 9)
    descriptors[:, 0] = 1
    amplitude[..., 4, 4] = 1
    if not isolated:
        for distance in (1, 2):
            for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                y, x = 4 + dy * distance, 4 + dx * distance
                amplitude[..., y, x] = 1
                similar = neighbors == "similar" if distance == 1 else distant == "similar"
                descriptors[..., y, x] = torch.tensor([1., 0.]) if similar else torch.tensor([0., 1.])
    return amplitude, descriptors


def main():
    """Save observations, including negative outcomes; never turn fixtures into efficacy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "runs" / "mechanism_probe"))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = GroupingConfig(radius=2)
    operator = DirectionalGrouping(config)
    records = []
    figure = Image.new("RGB", (1000, 270), "white")
    draw = ImageDraw.Draw(figure)
    conditions = [("isolated", "similar", "similar", True),
                  ("near same / far same", "similar", "similar", False),
                  ("near other / far same", "dissimilar", "similar", False),
                  ("near same / far other", "similar", "dissimilar", False)]
    for column, (name, near, far, isolated) in enumerate(conditions):
        amplitude, descriptors = fixture(near, far, isolated)
        result = operator(amplitude, descriptors)
        center = float(result.private_fraction[0, 0, 4, 4])
        records.append({"condition": name, "center_private_fraction": center,
                        "center_widths": result.widths[0, :, 4, 4].tolist(),
                        "center_unique": float(result.unique[0, 0, 4, 4]),
                        "center_shared": float(result.shared[0, 0, 4, 4])})
        values = result.private_fraction[0, 0].numpy()
        tile = Image.fromarray((values * 255).round().astype("uint8")).resize((216, 216), resample=Image.Resampling.NEAREST).convert("RGB")
        figure.paste(tile, (column * 250 + 10, 35))
        draw.text((column * 250 + 10, 5), name, fill="black")
        draw.text((column * 250 + 10, 255), "center private = %.4f" % center, fill="black")
    figure.save(output / "private_fraction.png")
    result = {"operator": asdict(config), "conditions": records,
              "kind": "constructed-feature mechanism observation", "human_fit_established": False,
              "classification_efficacy_established": False}
    atomic_json(output / "probe.json", result)
    print(result)


if __name__ == "__main__":
    main()
