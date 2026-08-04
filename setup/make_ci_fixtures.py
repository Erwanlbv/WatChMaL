#!/usr/bin/env python3
"""
Build the small, committed datasets that the Tier 3 end-to-end tests run on.

Tier 3 needs real data, not synthetic: the distributions matter. A synthetic fixture has
no zero-hit events, no realistic hit-count tail and no real charge/time spread - and a
zero-hit event is exactly what crashed `CNNmPMTDataset.process_data` the first time a
larger split was drawn (see claude_report/MERGE_CHECK_cnn_datasets_report.md §4.3).

So this carves small subsets out of the full local productions and writes them,
UNCOMPRESSED (see GZIP below), into `tests/data/`. Sizes are kept as small as coverage
allows:

  toy mPMT   200 e- + 200 mu-   ~1.7 MB   the fast every-PR substrate; the only fixture
                                          that exercises CNNmPMTDataset (19-PMT modules,
                                          channel permutations, the transform set)
  HK FD       50 e- +  50 mu-   ~3.1 MB   coverage only: the single-PMT CNNDataset path,
                                          the 274x144 image, and the GAT graph path
                                          (built from this same h5 by the test fixture)

Zero-hit events are excluded from both: they are physically real (muons below Cherenkov
threshold) but no image dataset can build an event display from them, and a CI job that
resamples its split should not start failing for that reason. The dataset classes should
handle them; until they do, this keeps the fixtures usable.

    python setup/make_ci_fixtures.py            write tests/data/
    python setup/make_ci_fixtures.py --check    verify the committed files match

Source productions are local paths (below); the script is here so the fixtures can be
regenerated and audited rather than being opaque committed blobs.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "tests" / "data"

SEED = 20260730

# (name, source h5, source geometry, source image-positions, labels (e-, mu-), n per class)
SOURCES = {
    "toy_mpmt": dict(
        h5="/Users/erwan/work/mc_prods/watchmal_tutorial/data/mPMT_3m_100k_e-_100k_mu-.h5",
        geometry="/Users/erwan/work/mc_prods/watchmal_tutorial/data/mPMT_3m_geometry.npz",
        positions="/Users/erwan/work/mc_prods/watchmal_tutorial/data/mPMT_3m_mpmt_image_positions.npz",
        labels=(1, 2), n_per_class=200,
    ),
    "hkfd": dict(
        h5="/Users/erwan/work/mc_prods/hk/hkfd_emu_rwcs_2k_watchmal.h5",
        geometry="/Users/erwan/work/geom/hyperk_20inch_pmts.npz",
        positions="/Users/erwan/work/geom/HK_HybridmPMT_WithOD_Realistic_mapping_squashed.npz",
        labels=(11, 13), n_per_class=50,
    ),
}

HIT_ARRAYS = ("hit_pmt", "hit_time", "hit_charge")
# NO compression. `H5CommonDataset.load_hits` memmaps the hit arrays using
# `dataset.id.get_offset()`, which is None for a chunked/compressed dataset - so a gzipped
# h5 cannot be read through the default `use_memmap=True` path at all; it dies with
# `TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'` from inside numpy.
# Shipping these uncompressed costs ~2 MB and buys coverage of the path production uses.
GZIP: dict = {}


def choose_events(src_h5: str, labels: tuple[int, int], n_per_class: int) -> np.ndarray:
    """Pick n_per_class events of each label, excluding zero-hit events."""
    with h5py.File(src_h5, "r") as f:
        lab = f["labels"][:]
        bounds = np.append(f["event_hits_index"][:], f["hit_pmt"].shape[0])
        n_hits = np.diff(bounds)
    rng = np.random.default_rng(SEED)
    picked = []
    for label in labels:
        pool = np.flatnonzero((lab == label) & (n_hits > 0))
        if len(pool) < n_per_class:
            raise SystemExit(f"{src_h5}: only {len(pool)} usable events with label {label}")
        picked.append(rng.choice(pool, n_per_class, replace=False))
    return np.sort(np.concatenate(picked))


def write_subset(src_h5: str, dst: Path, idx: np.ndarray) -> None:
    with h5py.File(src_h5, "r") as f, h5py.File(dst, "w") as g:
        bounds = np.append(f["event_hits_index"][:], f["hit_pmt"].shape[0])
        hits = {k: [] for k in HIT_ARRAYS}
        offsets, cursor = [], 0
        for i in idx:
            start, stop = bounds[i], bounds[i + 1]
            for k in HIT_ARRAYS:
                hits[k].append(f[k][start:stop])
            offsets.append(cursor)
            cursor += stop - start
        for k in HIT_ARRAYS:
            g.create_dataset(k, data=np.concatenate(hits[k]), **GZIP)
        g.create_dataset("event_hits_index", data=np.array(offsets, np.int64), **GZIP)
        # every per-event array the source has, subset in the same order
        for k in sorted(f.keys()):
            if k in HIT_ARRAYS or k == "event_hits_index":
                continue
            arr = f[k][:]
            if arr.shape[0] != lab_len(f):
                continue
            if arr.dtype == object:  # root_files: variable-length strings
                g.create_dataset(k, data=arr[idx].astype("S"), **GZIP)
            else:
                g.create_dataset(k, data=arr[idx], **GZIP)
        g.attrs["source"] = src_h5
        g.attrs["seed"] = SEED
        g.attrs["n_events"] = len(idx)


def lab_len(f) -> int:
    return f["event_hits_index"].shape[0]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def verify(dst: Path, src_h5: str, idx: np.ndarray) -> None:
    """Every hit of every event must match the source, byte for byte."""
    with h5py.File(src_h5, "r") as f, h5py.File(dst, "r") as g:
        sb = np.append(f["event_hits_index"][:], f["hit_pmt"].shape[0])
        db = np.append(g["event_hits_index"][:], g["hit_pmt"].shape[0])
        assert len(db) - 1 == len(idx), "event count mismatch"
        assert np.all(np.diff(db) > 0), "event_hits_index not strictly increasing"
        for j, i in enumerate(idx):
            for k in HIT_ARRAYS:
                if not np.array_equal(f[k][sb[i]:sb[i + 1]], g[k][db[j]:db[j + 1]]):
                    raise SystemExit(f"{dst.name}: {k} mismatch at event {i} -> {j}")
        assert np.array_equal(f["labels"][:][idx], g["labels"][:]), "labels mismatch"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify the committed fixtures instead of rewriting them")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for name, spec in SOURCES.items():
        if not Path(spec["h5"]).exists():
            print(f"SKIP {name}: source {spec['h5']} not on this machine")
            continue
        idx = choose_events(spec["h5"], spec["labels"], spec["n_per_class"])
        h5_out = OUT / f"{name}_ci.h5"
        if args.check:
            if not h5_out.exists():
                raise SystemExit(f"missing fixture {h5_out}")
            verify(h5_out, spec["h5"], idx)
            print(f"OK  {h5_out.relative_to(REPO_ROOT)}  sha256:{digest(h5_out)}")
            continue
        write_subset(spec["h5"], h5_out, idx)
        verify(h5_out, spec["h5"], idx)
        for key, suffix in (("geometry", "geometry"), ("positions", "image_positions")):
            src = np.load(spec[key])
            np.savez_compressed(OUT / f"{name}_{suffix}.npz", **{k: src[k] for k in src.files})
        with h5py.File(h5_out) as g:
            hits = np.diff(np.append(g["event_hits_index"][:], g["hit_pmt"].shape[0]))
            print(f"{h5_out.relative_to(REPO_ROOT)}: {len(idx)} events, {hits.sum():,d} hits "
                  f"(min {hits.min()}, median {int(np.median(hits))}), "
                  f"{h5_out.stat().st_size / 1e6:.2f} MB  sha256:{digest(h5_out)}")


if __name__ == "__main__":
    main()
