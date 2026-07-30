#!/usr/bin/env python3
"""
Build the `mpmt_image_positions` file the CNN mPMT dataset needs, from a detector
geometry file.

`watchmal/dataset/cnn_mpmt/cnn_mpmt_dataset.py` loads
`np.load(mpmt_positions_file)['mpmt_image_positions']` - an (n_mpmt, 2) int array giving
the (row, col) each mPMT occupies in the unrolled CNN image. That is different
information from a geometry file, which says where each PMT sits in the tank, and some
datasets ship without it.

The layout is not invented here: it was reverse-engineered from a known-good pair
(`wcte_full_geo_v1.2.npz` + `WCTE_mPMT_image_positions.npz`) and reproduces that file,
up to a rigid rotation of the barrel columns (see --phi-offset). The structure is:

    rows [0 .. cap_h)                 first endcap   - (v, w) grid, centred
    rows [cap_h .. cap_h + n_bands)   barrel         - row = position along the axis
                                                       (descending), col = azimuth
    rows [.. + cap_h)                 second endcap  - same grid as the first

    python setup/build_mpmt_image_positions.py GEOMETRY.npz OUT.npz
    python setup/build_mpmt_image_positions.py --validate    # check against WCTE
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PMTS_PER_MPMT = 19
REFERENCE_GEOMETRY = Path("/Users/erwan/work/geom/wcte_full_geo_v1.2.npz")
REFERENCE_MAPPING = Path("/Users/erwan/work/geom/WCTE_mPMT_image_positions.npz")


def mpmt_centres(geometry: Path) -> np.ndarray:
    """Mean position of each mPMT. `tube_no` is 1-indexed, hence the -1: without it the
    first and last groups are partial and every centre is an average over two modules."""
    data = np.load(geometry)
    tube, position = data["tube_no"], data["position"]
    group = (np.asarray(tube) - 1) // PMTS_PER_MPMT
    return np.stack([position[group == g].mean(axis=0) for g in range(group.max() + 1)])


def _detect_axis(centres: np.ndarray) -> int:
    """Which coordinate is the cylinder axis.

    Endcap modules share a single value of it (a flat disc at each end), so the axis is
    the coordinate with the MOST modules sitting at its two extremes. Counting is the
    reliable signal: scoring by in-plane spread instead picks the wrong axis on a tank
    whose radius and half-height are similar (mPMT_3m has ~60 spread on all three).
    """
    counts = []
    for axis in range(3):
        along = centres[:, axis]
        extremes = (np.isclose(along, along.max(), atol=1e-3)
                    | np.isclose(along, along.min(), atol=1e-3))
        counts.append(extremes.sum())
    return int(np.argmax(counts))


def build(centres: np.ndarray, phi_offset_cols: int = 0) -> np.ndarray:
    axis = _detect_axis(centres)
    along = centres[:, axis]
    # In-plane axes in CYCLIC order, not the natural order np.delete would give: for a
    # y-axis detector the convention is (z, x), and (x, z) transposes both endcaps.
    v = centres[:, (axis + 1) % 3]
    w = centres[:, (axis + 2) % 3]
    radius = np.hypot(v, w)

    # Barrel modules sit on the wall (radius at its maximum); the rest are endcaps.
    is_barrel = radius > 0.95 * radius.max()
    top = is_barrel | (along < along.max() - 1e-3)
    is_top_cap = ~is_barrel & (along > 0)
    is_bottom_cap = ~is_barrel & (along <= 0)

    # --- barrel: row = band along the axis (descending), col = azimuth ---
    bands = np.unique(np.round(along[is_barrel], 3))[::-1]
    band_of = {value: index for index, value in enumerate(bands)}
    phi = np.degrees(np.arctan2(w, v))
    n_cols = int(np.round(np.median([np.sum(np.isclose(np.round(along[is_barrel], 3), b))
                                     for b in bands])))
    spacing = 360.0 / n_cols

    # --- endcaps: (v, w) grid at the module pitch, centred in the image width ---
    def cap_grid(mask, row_dir):
        """row_dir=-1 for the far endcap: seen from inside the tank the two discs are
        mirror images, and fitting each cap of the WCTE reference separately confirms it
        (21/21 exact per cap, opposite row directions)."""
        if not mask.any():
            return {}, 0
        vv, ww = v[mask], w[mask] * row_dir
        steps = np.unique(np.round(vv, 3))
        pitch = np.min(np.diff(steps)) if steps.size > 1 else 1.0
        col = np.round((vv - vv.min()) / pitch).astype(int)
        row = np.round((ww.max() - ww) / pitch).astype(int)
        col += (n_cols - 1 - col.max()) // 2          # centre the block horizontally
        return {i: (int(r), int(c)) for i, r, c in zip(np.flatnonzero(mask), row, col)}, int(row.max() + 1)

    top_map, top_h = cap_grid(is_top_cap, +1)
    bottom_map, _ = cap_grid(is_bottom_cap, -1)

    positions = np.zeros((len(centres), 2), dtype=np.int64)
    for index in range(len(centres)):
        if is_barrel[index]:
            row = top_h + band_of[np.round(along[index], 3)]
            col = int(np.round((phi[index] + 180.0) / spacing)) % n_cols
            col = (col + phi_offset_cols) % n_cols
        elif index in top_map:
            row, col = top_map[index]
        else:
            row, col = bottom_map[index]
            row += top_h + len(bands)
        positions[index] = (row, col)
    return positions


def validate() -> None:
    """Rebuild the WCTE mapping and compare with the shipped file."""
    expected = np.load(REFERENCE_MAPPING)["mpmt_image_positions"]
    centres = mpmt_centres(REFERENCE_GEOMETRY)
    n_cols = expected[:, 1].max() + 1

    for offset in range(n_cols):
        got = build(centres, phi_offset_cols=offset)
        if np.array_equal(got, expected):
            print(f"PASS  reproduces {REFERENCE_MAPPING.name} exactly "
                  f"(barrel columns rotated by {offset})")
            print(f"      {len(centres)} mPMTs, image {expected[:,0].max()+1} x {n_cols}")
            print("      The rotation is a free convention: the barrel is padded")
            print("      circularly (conv_pad_mode: circular), so the azimuth origin")
            print("      cannot change what the network sees.")
            return
    got = build(centres)
    rows_ok = np.array_equal(got[:, 0], expected[:, 0])
    raise SystemExit(f"FAIL  no column rotation matches. rows identical: {rows_ok}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("geometry", type=Path, nargs="?")
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--phi-offset", type=int, default=0,
                        help="rigid rotation of the barrel columns (free convention)")
    parser.add_argument("--validate", action="store_true",
                        help="rebuild the reference WCTE mapping and compare")
    args = parser.parse_args()

    if args.validate:
        validate()
        return
    if args.geometry is None or args.out is None:
        parser.error("geometry and out are required unless --validate is given")

    centres = mpmt_centres(args.geometry)
    positions = build(centres, phi_offset_cols=args.phi_offset)
    np.savez(args.out, mpmt_image_positions=positions)

    height, width = positions[:, 0].max() + 1, positions[:, 1].max() + 1
    print(f"{len(centres)} mPMTs -> image {height} x {width}")
    occupied = np.zeros((height, width), dtype=int)
    for row, col in positions:
        occupied[row, col] += 1
    print(f"pixels occupied: {(occupied > 0).sum()} / {height * width}")
    if (occupied > 1).any():
        raise SystemExit(f"ERROR: {int((occupied > 1).sum())} pixel(s) hold more than one mPMT")
    print("\n".join("  " + "".join("#" if value else "." for value in row) for row in occupied))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
