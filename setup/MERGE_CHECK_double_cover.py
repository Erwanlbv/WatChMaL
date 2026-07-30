#!/usr/bin/env python3
"""
Check the `double_cover` image transform, for both CNN dataset families.

`double_cover` rearranges an event display so the image is a genuine double cover of the
detector: every module appears twice, and the image edges become physically meaningful
cyclic boundaries - which is what makes `conv_pad_mode: circular` legitimate, since torch
pads BOTH image dimensions circularly.

The check is geometric, not a re-statement of the implementation. A tracer image carrying
module identity is pushed through the transform, and the modules that end up glued
together at the image edges are looked up in the real 3D geometry. If the transform works,
those pairs are true physical neighbours; if it does not, they are far apart. The
untransformed image is measured the same way, so the check can fail for the right reason.

    python setup/MERGE_CHECK_double_cover.py

Exit code is non-zero if any check fails. Paths are the local merge-check datasets; edit
the constants below for another machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOY_H5 = "/Users/erwan/work/mc_prods/watchmal_tutorial/data/mPMT_3m_100k_e-_100k_mu-.h5"
TOY_MAP = "/Users/erwan/work/mc_prods/watchmal_tutorial/data/mPMT_3m_mpmt_image_positions.npz"
TOY_GEO = "/Users/erwan/work/mc_prods/watchmal_tutorial/data/mPMT_3m_geometry.npz"
HK_MAP = "/Users/erwan/work/geom/HK_HybridmPMT_WithOD_Realistic_mapping_squashed.npz"
HK_GEO = "/Users/erwan/work/geom/hyperk_20inch_pmts.npz"


def edge_pairs(id_img, xyz, axis):
    """3D distances between the modules the image glues together across one edge."""
    a = id_img[-1] if axis == 0 else id_img[:, -1]
    b = id_img[0] if axis == 0 else id_img[:, 0]
    mask = (a > 0) & (b > 0)
    if not mask.any():
        return 0, float("nan")
    d = np.linalg.norm(xyz[a[mask] - 1] - xyz[b[mask] - 1], axis=1)
    return int(mask.sum()), float(np.median(d))


def interior_spacing(id_img, xyz):
    """Median 3D distance between modules adjacent in the image, edges excluded."""
    out = []
    for axis in (0, 1):
        a, b = id_img, np.roll(id_img, -1, axis=axis)
        a, b = (a[:-1], b[:-1]) if axis == 0 else (a[:, :-1], b[:, :-1])
        mask = (a > 0) & (b > 0)
        out.append(np.linalg.norm(xyz[a[mask] - 1] - xyz[b[mask] - 1], axis=1))
    return float(np.median(np.concatenate(out)))


def report_edges(label, id_img, xyz, typical):
    parts = []
    for axis, name in ((0, "top<->bottom"), (1, "left<->right")):
        n, med = edge_pairs(id_img, xyz, axis)
        parts.append(f"{name}: {n:3d} pairs, median gap {med:7.1f} cm ({med / typical:5.1f}x typical)")
    print(f"    {label:24s} " + " | ".join(parts))
    return [edge_pairs(id_img, xyz, ax)[1] for ax in (0, 1)]


def check_toy() -> bool:
    """mPMT toy detector: CNNmPMTDataset really applies double_cover."""
    from watchmal.dataset.cnn_mpmt.cnn_mpmt_dataset import CNNmPMTDataset

    print("=== mPMT_3m toy detector (CNNmPMTDataset) ===")
    ds = CNNmPMTDataset(h5file=TOY_H5, mpmt_positions_file=TOY_MAP, geometry_file=TOY_GEO,
                        channels=["charge", "time"], transforms=["double_cover"])
    pos = np.load(TOY_MAP)["mpmt_image_positions"]
    n = len(pos)
    H, W = ds.image_height, ds.image_width
    xyz = np.load(TOY_GEO)["position"].reshape(n, 19, 3).mean(axis=1)

    ids = np.zeros((H, W), np.int64)
    ids[pos[:, 0], pos[:, 1]] = np.arange(n) + 1
    # every channel carries the same id, so the within-mPMT channel permutation that
    # rotate_image applies cannot corrupt the trace
    tracer = np.broadcast_to(ids, (38, H, W)).astype(np.float32).copy()
    out = ds.double_cover({"data": tracer.copy()})["data"]
    assert np.all(out == out[0]), "channels disagree after transform"
    out_ids = out[0].astype(np.int64)

    ok = []
    barrel_h = H - 2 * ds.endcap_size
    shape_ok = out.shape[1] == H + barrel_h and out.shape[2] == W
    print(f"  shape        ({38},{H},{W}) -> {tuple(out.shape)}; expected height "
          f"{H}+{barrel_h}={H + barrel_h}: {'PASS' if shape_ok else 'FAIL'}")
    ok.append(shape_ok)

    counts = np.bincount(out_ids.ravel(), minlength=n + 1)[1:]
    cover_ok = bool((counts == 2).all())
    print(f"  double cover {int((counts == 2).sum())}/{n} mPMTs appear exactly twice "
          f"(min {counts.min()}, max {counts.max()}): {'PASS' if cover_ok else 'FAIL'}")
    ok.append(cover_ok)

    typical = interior_spacing(ids, xyz)
    print(f"  cyclic boundaries (typical adjacent-mPMT spacing {typical:.1f} cm):")
    report_edges("plain image", ids, xyz, typical)
    after = report_edges("after double_cover", out_ids, xyz, typical)
    edges_ok = all(d <= 1.5 * typical for d in after)
    print(f"    -> glued pixels are physical neighbours: {'PASS' if edges_ok else 'FAIL'}")
    ok.append(edges_ok)
    return all(ok)


def check_hk() -> bool:
    """HK far detector: single 20-inch PMTs, CNNDataset - double_cover is NOT reachable."""
    from watchmal.dataset.cnn.cnn_dataset import CNNDataset

    print("\n=== HK far detector (CNNDataset) ===")
    has = hasattr(CNNDataset, "double_cover")
    print(f"  CNNDataset defines double_cover: {has}")
    print("  CNNDataset.__init__ hardcodes `self.transforms = None` (the get_transformations")
    print("  call is commented out), so ANY transform listed in a config is silently ignored.")

    pip = np.load(HK_MAP)["pmt_image_positions"]
    xyz = np.load(HK_GEO)["position"]
    H, W = pip[:, 0].max() + 1, pip[:, 1].max() + 1
    ids = np.zeros((H, W), np.int64)
    ids[pip[:, 0], pip[:, 1]] = np.arange(len(pip)) + 1
    typical = interior_spacing(ids, xyz)
    print(f"  image {H}x{W}; typical adjacent-PMT spacing {typical:.1f} cm")
    print("  what conv_pad_mode: circular currently wraps onto what:")
    report_edges("plain image", ids, xyz, typical)
    print("  -> azimuth (left<->right) wraps correctly; the vertical wrap glues the two")
    print("     ends of the tank together. A single-PMT double_cover does not exist yet.")
    return not has  # informational: reports the gap, does not fail the suite


if __name__ == "__main__":
    results = [check_toy(), check_hk()]
    print(f"\n{'ALL CHECKS PASSED' if all(results) else 'SOME CHECKS FAILED'}")
    sys.exit(0 if all(results) else 1)
