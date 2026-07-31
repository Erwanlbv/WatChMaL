#!/usr/bin/env python3
"""
Export a few events from a PyG graph dataset to the JSON the landing page renders.

The landing page ships an interactive 3-D event display. Its data is a static JSON
payload committed alongside the page - no dataset is read at page load, and the site
keeps its property of having no external or runtime dependencies.

    python setup/make_event_display_data.py --out landing/events.json

Coordinates
-----------
Node features are (charge, time, x, y, z) for the `qtxyz` datasets. Some datasets store
the coordinates min-max normalised to [0, 1]; that mapping is global and therefore
exactly invertible, which is what `--extent` does. The recovered geometry is a cylinder
of radius 3243 cm and half-height 3296 cm - a 65 m by 66 m tank.

Size
----
Positions are written as integers (1 cm resolution, far below the ~50 cm PMT spacing)
and charge to two decimals, as parallel flat arrays rather than per-hit objects. That is
roughly 25 bytes per hit, so a few thousand hits stay well inside 100 kB before the
gzip that GitHub Pages applies on the wire.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Global min-max extents, read off the one event in each shipped smoke dataset that kept
# raw coordinates. Used to invert the [0, 1] normalisation.
DEFAULT_EXTENT = ((-3242.77, 3242.77), (-3242.77, 3242.77), (-3296.47, 3296.47))

# (dataset directory, particle label, event index) - chosen by eye from unrolled views
# for a clearly developed Cherenkov ring at a hit count the browser can draw smoothly.
DEFAULT_EVENTS = [
    ("e-_200_qtxyz_pid_knn10", "e−", 11),
    ("mu-_200_qtxyz_pid_knn10", "μ−", 118),
]


def load_event(dataset_dir: Path, index: int, extent=DEFAULT_EXTENT):
    """Return (charge, xyz) for one event, with coordinates in centimetres."""
    data, slices, _ = torch.load(
        dataset_dir / "processed" / "data.pt",
        map_location="cpu", weights_only=False, mmap=True,
    )
    bounds = slices["x"]
    if not 0 <= index < len(bounds) - 1:
        raise SystemExit(f"event {index} out of range (dataset has {len(bounds)-1})")

    x = data["x"][bounds[index]:bounds[index + 1]].numpy().astype(np.float64)
    charge, xyz = x[:, 0], x[:, 2:5].copy()

    # Normalised datasets map every coordinate into [0, 1]; raw ones are in the thousands.
    if np.abs(xyz).max() <= 10.0:
        for k in range(3):
            lo, hi = extent[k]
            xyz[:, k] = xyz[:, k] * (hi - lo) + lo

    return charge, xyz


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets-root", type=Path,
                   default=Path.home() / "work/mc_prods/tutorial-dataset-watchmal"
                                          "/graph_datasets/smoke",
                   help="directory holding the dataset directories")
    p.add_argument("--out", type=Path, default=Path("landing/events.json"))
    args = p.parse_args()

    events = []
    for name, label, index in DEFAULT_EVENTS:
        charge, xyz = load_event(args.datasets_root / name, index)
        events.append({
            "label": label,
            "source": f"{name}#{index}",
            "hits": int(len(charge)),
            "charge_total": round(float(charge.sum()), 1),
            "x": [int(round(v)) for v in xyz[:, 0]],
            "y": [int(round(v)) for v in xyz[:, 1]],
            "z": [int(round(v)) for v in xyz[:, 2]],
            "q": [round(float(v), 2) for v in charge],
        })
        print(f"  {label:3s} {name}#{index}: {len(charge)} hits, "
              f"Q={charge.sum():.0f}, q in [{charge.min():.2f}, {charge.max():.1f}]")

    payload = {
        "detector": {"radius": 3243.0, "half_height": 3296.5, "units": "cm"},
        "events": events,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
