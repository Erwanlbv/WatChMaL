"""
Tier 2 fixtures.

Most of Tier 2 needs no data at all. The two smoke-run tests do, and the datasets they
use are 200-event subsets carved off the cluster files with `setup/make_smoke_subset.py`
— far too large to commit, so they live outside the repo and the tests SKIP when they
are absent (which is the case on a CI runner). Point `WATCHMAL_SMOKE_DATA` at another
directory to use a different copy.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

DEFAULT_SMOKE_DATA = Path(
    "/Users/erwan/work/mc_prods/wat-repo-test-datasets/graph_datasets/smoke"
)

# Sub-directory names, and what each is for. The energy dataset is deliberately unused:
# regression is already covered by the vertex one, with 3 targets instead of 1.
PID_ELECTRON = "e-_200_qtxyz_pid_knn10"
PID_MUON = "mu-_200_qtxyz_pid_knn10"
VERTEX = "mu-_200_train_tqxyz_edges_xyz_label_vertex_knn5"


# The run-time graph path reads a flat WatChMaL HDF5 file and a geometry file rather than
# a stored PyG dataset. Both live outside the repo for the same reason: 1985 events are
# 20 MB of hits, and the geometry is detector-specific.
DEFAULT_H5_PID = Path("/Users/erwan/work/mc_prods/hk/hkfd_emu_rwcs_2k_watchmal.h5")
DEFAULT_GEOMETRY = Path("/Users/erwan/work/geom/hyperk_20inch_pmts.npz")


def smoke_data_root() -> Path:
    return Path(os.environ.get("WATCHMAL_SMOKE_DATA", DEFAULT_SMOKE_DATA))


def require_datasets(*names: str) -> list[Path]:
    """Return the requested dataset directories, or skip if any is missing."""
    root = smoke_data_root()
    paths = [root / name for name in names]
    missing = [p for p in paths if not (p / "processed" / "data.pt").is_file()]
    if missing:
        pytest.skip(
            "local smoke datasets not available "
            f"(missing: {[m.name for m in missing]} under {root}). "
            "Build them with setup/make_smoke_subset.py, or set WATCHMAL_SMOKE_DATA."
        )
    return paths


@pytest.fixture(scope="session")
def pid_datasets() -> list[Path]:
    """e- and mu- graphs for a two-class separation run."""
    return require_datasets(PID_ELECTRON, PID_MUON)


@pytest.fixture(scope="session")
def vertex_dataset() -> Path:
    """mu- graphs with a 3-component vertex target."""
    return require_datasets(VERTEX)[0]


@pytest.fixture
def make_split(tmp_path):
    """Write an index-list .npz, the way create_index.py does: one global permutation
    cut into train/val/test, so both classes land in every split when the underlying
    dataset is a concatenation of per-class files."""

    def _make(n_events: int, n_train: int, n_val: int, n_test: int, seed: int = 1234,
              name: str = "split.npz") -> Path:
        assert n_train + n_val + n_test <= n_events
        order = np.random.default_rng(seed).permutation(n_events)
        path = tmp_path / name
        np.savez(
            path,
            train_idxs=order[:n_train],
            val_idxs=order[n_train:n_train + n_val],
            test_idxs=order[n_train + n_val:n_train + n_val + n_test],
        )
        return path

    return _make


@pytest.fixture(scope="session")
def h5_pid_file() -> Path:
    """A flat WatChMaL HDF5 file holding e- and mu- events."""
    path = Path(os.environ.get("WATCHMAL_H5_PID", DEFAULT_H5_PID))
    if not path.is_file():
        pytest.skip(f"HDF5 PID file not available at {path}; set WATCHMAL_H5_PID.")
    return path


@pytest.fixture(scope="session")
def hk_geometry() -> Path:
    """Geometry providing the hit coordinates for the file above."""
    path = Path(os.environ.get("WATCHMAL_GEOMETRY", DEFAULT_GEOMETRY))
    if not path.is_file():
        pytest.skip(f"Geometry file not available at {path}; set WATCHMAL_GEOMETRY.")
    return path
