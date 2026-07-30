#!/usr/bin/env python3
"""
Convert a per-event-group HK h5 file into the flat, offset-indexed layout the WatChMaL
CNN datasets read.

Source layout (one group per event):

    event_0/tube_ids (n_hits,)   pmt_charge (n_hits,)   pmt_time (n_hits,)
    event_0/energy () vertex_x () particle_dir_x () dwall () towall () ...
    event_1/...

Target layout (WatChMaL): per-event arrays of length N, plus FLAT hit arrays of length
sum(n_hits), joined by `event_hits_index` holding the start offset of each event:

    event_hits_index (N,)   hit_pmt / hit_time / hit_charge (total_hits,)
    labels (N,)  energies (N,1)  positions (N,1,3)  angles (N,2)  veto (N,) ...

    python setup/convert_hk_h5_to_watchmal.py IN.h5 OUT.h5
    python setup/convert_hk_h5_to_watchmal.py IN.h5 OUT.h5 --label-map 13=0 11=1

Field mapping, and what the source does NOT provide - see MISSING below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

# Datasets the WatChMaL CNN pipeline reads that the HK per-event format has no
# equivalent for. They are written, filled with a documented default, because the
# dataset classes and `analysis/` index them positionally.
MISSING = {
    "veto": "outer-detector veto flag - not in the source; written all-False",
    "veto2": "second veto flag - not in the source; written all-False",
    "fully_contained": "containment flag - not in the source; written all-True. "
                       "`dwall`/`towall` are kept so you can derive your own cut",
    "root_files": "provenance string per event - not in the source; written as the "
                  "input file name",
}


def direction_to_angles(dx, dy, dz):
    """WatChMaL stores direction as two angles; the HK format stores a unit vector.

    Convention here matches the tutorial file's ranges (both in radians, second in
    [-pi, pi]): polar angle from +y (the tank axis in these geometries) and azimuth in
    the x-z plane. Check this against your own geometry before trusting any
    direction-dependent result - it is the one field that cannot be copied verbatim.
    """
    polar = np.arccos(np.clip(dy, -1.0, 1.0))
    azimuth = np.arctan2(dz, dx)
    return np.stack([polar, azimuth], axis=1).astype(np.float32)


def convert(source: Path, target: Path, label_map: dict[int, int] | None = None,
            one_indexed_tubes: bool = True) -> None:
    with h5py.File(source, "r") as fin:
        events = sorted((k for k in fin.keys() if k.startswith("event_")),
                        key=lambda k: int(k.split("_")[1]))
        if not events:
            raise SystemExit(f"{source}: no event_* groups found")

        n = len(events)
        offsets = np.zeros(n, dtype=np.int64)
        pmt, charge, time = [], [], []
        scalars = {name: np.zeros(n, dtype=np.float32) for name in
                   ("energy", "dwall", "towall", "trigger_time",
                    "vertex_x", "vertex_y", "vertex_z",
                    "particle_dir_x", "particle_dir_y", "particle_dir_z",
                    "particle_start_x", "particle_start_y", "particle_start_z",
                    "particle_stop_x", "particle_stop_y", "particle_stop_z")}
        labels = np.zeros(n, dtype=np.int32)

        running = 0
        for index, key in enumerate(events):
            group = fin[key]
            offsets[index] = running
            tubes = group["tube_ids"][:].astype(np.int32)
            # WatChMaL indexes PMTs from 0 (hit_pmt // 19 selects the mPMT); the HK file
            # numbers tubes from 1. Getting this wrong shifts every hit by one PMT.
            if one_indexed_tubes:
                tubes = tubes - 1
            pmt.append(tubes)
            charge.append(group["pmt_charge"][:].astype(np.float32))
            time.append(group["pmt_time"][:].astype(np.float32))
            running += len(tubes)
            for name in scalars:
                scalars[name][index] = group[name][()]
            labels[index] = int(group["event_type"][()])

        hit_pmt = np.concatenate(pmt)
        hit_charge = np.concatenate(charge)
        hit_time = np.concatenate(time)

    if label_map:
        mapped = np.full(n, -1, dtype=np.int32)
        for source_label, target_label in label_map.items():
            mapped[labels == source_label] = target_label
        if (mapped < 0).any():
            unmapped = sorted(set(labels[mapped < 0].tolist()))
            raise SystemExit(f"--label-map does not cover event_type values {unmapped}")
        labels = mapped

    positions = np.stack([scalars["vertex_x"], scalars["vertex_y"], scalars["vertex_z"]],
                         axis=1).reshape(n, 1, 3).astype(np.float32)
    angles = direction_to_angles(scalars["particle_dir_x"], scalars["particle_dir_y"],
                                 scalars["particle_dir_z"])

    with h5py.File(target, "w") as fout:
        fout.create_dataset("event_hits_index", data=offsets)
        fout.create_dataset("hit_pmt", data=hit_pmt)
        fout.create_dataset("hit_charge", data=hit_charge)
        fout.create_dataset("hit_time", data=hit_time)
        fout.create_dataset("labels", data=labels)
        fout.create_dataset("energies", data=scalars["energy"].reshape(n, 1))
        fout.create_dataset("positions", data=positions)
        fout.create_dataset("angles", data=angles)
        fout.create_dataset("event_ids", data=np.arange(n, dtype=np.int32))
        # placeholders for the fields the source does not carry
        fout.create_dataset("veto", data=np.zeros(n, dtype=bool))
        fout.create_dataset("veto2", data=np.zeros(n, dtype=bool))
        fout.create_dataset("fully_contained", data=np.ones(n, dtype=bool))
        fout.create_dataset("root_files",
                            data=np.array([str(source)] * n, dtype=h5py.special_dtype(vlen=str)))
        # HK-specific extras, kept rather than dropped: dwall/towall are the usual basis
        # for a containment cut, and the start/stop points for a track-length one.
        for name in ("dwall", "towall", "trigger_time", "particle_start_x",
                     "particle_start_y", "particle_start_z", "particle_stop_x",
                     "particle_stop_y", "particle_stop_z", "particle_dir_x",
                     "particle_dir_y", "particle_dir_z"):
            fout.create_dataset(name, data=scalars[name])
        fout.attrs["converted_from"] = str(source)
        fout.attrs["placeholder_datasets"] = ", ".join(sorted(MISSING))

    print(f"{n} events, {len(hit_pmt):,} hits -> {target}")
    print(f"  hit_pmt {hit_pmt.min()}..{hit_pmt.max()}  "
          f"labels {sorted(set(labels.tolist()))}  "
          f"hits/event median {int(np.median(np.diff(np.append(offsets, len(hit_pmt)))))}")
    print("\nplaceholders written (source has no equivalent):")
    for name, why in sorted(MISSING.items()):
        print(f"  {name:17s} {why}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--label-map", nargs="*", default=None, metavar="SRC=DST",
                        help="remap event_type to class indices, e.g. 13=0 11=1")
    parser.add_argument("--zero-indexed-tubes", action="store_true",
                        help="source tube_ids already start at 0 (default: they start at 1)")
    args = parser.parse_args()

    label_map = None
    if args.label_map:
        label_map = dict(tuple(int(part) for part in item.split("=")) for item in args.label_map)

    convert(args.source, args.target, label_map, not args.zero_indexed_tubes)


if __name__ == "__main__":
    main()
