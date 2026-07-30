#!/usr/bin/env python3
"""
Concatenate flat WatChMaL h5 files into one, for multi-class CNN runs.

The graph family has `PyGConcatDataset` for this, but the h5/CNN datasets take a single
`h5file`, so an e-/mu- classification run needs the two single-particle files physically
merged. The only field that cannot simply be stacked is `event_hits_index`: it holds the
start offset of each event into the flat hit arrays, so every file after the first has to
be shifted by the running hit total.

    python setup/MERGE_CHECK_concat_watchmal_h5.py OUT.h5 IN1.h5 IN2.h5 [IN3.h5 ...]

Per-event arrays (first axis == n_events) are concatenated in file order, so the event
order is file1 then file2 - build the train/val/test split by permuting indices over the
whole result, not by slicing it. Hit arrays (first axis == n_hits) are concatenated and
`event_hits_index` is re-offset. `labels` is left untouched: map it to contiguous class
indices with the engine's `label_set` rather than rewriting the file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

HIT_ARRAYS = ("hit_pmt", "hit_time", "hit_charge", "hit_parent")


def concat(target: Path, sources: list[Path]) -> None:
    metas = []
    for s in sources:
        with h5py.File(s, "r") as f:
            n_events = f["event_hits_index"].shape[0]
            n_hits = f["hit_pmt"].shape[0]
            keys = sorted(f.keys())
            metas.append((s, n_events, n_hits, keys))
            print(f"  {s.name}: {n_events:,d} events, {n_hits:,d} hits, "
                  f"labels {np.unique(f['labels'][:]).tolist()}")

    keysets = [set(k) for _, _, _, k in metas]
    common = set.intersection(*keysets)
    dropped = set.union(*keysets) - common
    if dropped:
        print(f"  NOTE: dropping keys not present in every input: {sorted(dropped)}")

    total_events = sum(m[1] for m in metas)
    total_hits = sum(m[2] for m in metas)

    with h5py.File(target, "w") as fout:
        for key in sorted(common):
            if key == "event_hits_index":
                continue
            parts, hit_offset = [], 0
            for src, n_events, n_hits, _ in metas:
                with h5py.File(src, "r") as fin:
                    parts.append(fin[key][:])
            arr = np.concatenate(parts, axis=0)
            expected = total_hits if key in HIT_ARRAYS else total_events
            if arr.shape[0] != expected:
                raise SystemExit(f"{key}: concatenated length {arr.shape[0]} != expected {expected}")
            fout.create_dataset(key, data=arr)

        # event_hits_index: shift each file's offsets past the hits already written
        offsets, hit_offset = [], 0
        for src, n_events, n_hits, _ in metas:
            with h5py.File(src, "r") as fin:
                offsets.append(fin["event_hits_index"][:].astype(np.int64) + hit_offset)
            hit_offset += n_hits
        ehi = np.concatenate(offsets)
        fout.create_dataset("event_hits_index", data=ehi)

    # verify: every event's hit slice must be non-empty, in range and correctly ordered
    with h5py.File(target, "r") as f:
        ehi = f["event_hits_index"][:]
        n_hits = f["hit_pmt"].shape[0]
        bounds = np.append(ehi, n_hits)
        lengths = np.diff(bounds)
        labels, counts = np.unique(f["labels"][:], return_counts=True)
        assert ehi[0] == 0, "first event does not start at 0"
        assert np.all(np.diff(ehi) > 0), "event_hits_index is not strictly increasing"
        assert np.all(lengths > 0), "some event has zero hits"
        assert n_hits == total_hits, f"hit total {n_hits} != {total_hits}"
        print(f"\n{target}: {len(ehi):,d} events, {n_hits:,d} hits")
        print(f"  labels {dict(zip(labels.tolist(), counts.tolist()))}")
        print(f"  hits/event min {lengths.min()} median {int(np.median(lengths))} max {lengths.max()}")
        print("  event_hits_index: strictly increasing, starts at 0, no empty events  OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", type=Path)
    p.add_argument("sources", type=Path, nargs="+")
    a = p.parse_args()
    if len(a.sources) < 2:
        raise SystemExit("need at least two source files")
    concat(a.target, a.sources)
