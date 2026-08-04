"""
Tier 3 fixtures — end-to-end runs on the committed datasets in `tests/data/`.

Unlike Tier 2's smoke runs, these never skip: the data ships with the repo (~2.5 MB,
built by `setup/make_ci_fixtures.py`), so a CI runner gets the same coverage a laptop
does. They are real detector events, not synthetic — the distributions matter, and a
synthetic fixture has no zero-hit events, no realistic hit-count tail and no real
charge/time spread.

Everything runs from a *copy of the shipped* `tutorial/config/watchmal` tree, repointed at
the fixtures. Never from the private `config/` workspace: what is tested has to be what
ships.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.discovery import REPO_ROOT

DATA = REPO_ROOT / "tests" / "data"
TUTORIAL_TREE = REPO_ROOT / "tutorial" / "config" / "watchmal"

# (h5, geometry, image positions, label pair, image channels) per detector fixture.
FIXTURES = {
    "toy_mpmt": dict(
        h5=DATA / "toy_mpmt_ci.h5",
        geometry=DATA / "toy_mpmt_geometry.npz",
        positions=DATA / "toy_mpmt_image_positions.npz",
        labels=[1, 2],
        # CNNmPMTDataset: 19 charge + 19 time + 3 mpmt_position
        num_input_channels=41,
        dataset_config="cnn_mpmt",
        positions_key="mpmt_positions_file",
    ),
    "hkfd": dict(
        h5=DATA / "hkfd_ci.h5",
        geometry=DATA / "hkfd_geometry.npz",
        positions=DATA / "hkfd_image_positions.npz",
        labels=[11, 13],
        # CNNDataset with use_times + use_charges
        num_input_channels=2,
        dataset_config="SK_cnn",
        positions_key="pmt_positions_file",
    ),
}


def pytest_generate_tests(metafunc):
    """Parametrise any test asking for `detector` over every committed fixture."""
    if "detector" in metafunc.fixturenames:
        metafunc.parametrize("detector", sorted(FIXTURES), ids=sorted(FIXTURES))


@pytest.fixture(scope="session")
def fixture_data():
    """The committed datasets, with a clear failure if they were not generated."""
    missing = [str(p) for spec in FIXTURES.values()
               for p in (spec["h5"], spec["geometry"], spec["positions"]) if not p.is_file()]
    if missing:
        pytest.fail("committed test data missing (regenerate with "
                    f"setup/make_ci_fixtures.py): {missing}")
    return FIXTURES


def n_events(h5file: Path) -> int:
    import h5py
    with h5py.File(h5file, "r") as f:
        return int(f["event_hits_index"].shape[0])


def patch_yaml(path: Path, edits: dict) -> None:
    """Apply `{"a.b.c": value}` edits to a YAML file in place, creating missing nodes."""
    doc = yaml.safe_load(path.read_text()) or {}
    for dotted, value in edits.items():
        node = doc
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


@pytest.fixture
def config_tree(tmp_path):
    """A copy of the shipped watchmal config tree, repointed at a committed fixture."""

    def _build(detector: str, split_path: Path, extra: dict[str, dict] | None = None) -> Path:
        spec = FIXTURES[detector]
        tree = tmp_path / "config"
        if not tree.exists():
            shutil.copytree(TUTORIAL_TREE, tree)
        # one data config per detector, named after it, selecting that detector's dataset
        data_cfg = tree / "data" / f"ci_{detector}.yaml"
        data_cfg.write_text(yaml.safe_dump({
            "defaults": [{"dataset": spec["dataset_config"]}],
            "split_path": str(split_path),
            "dataset": {
                "h5file": str(spec["h5"]),
                "geometry_file": str(spec["geometry"]),
                spec["positions_key"]: str(spec["positions"]),
            },
        }, sort_keys=False))
        # the committed h5 files are already 0-indexed (the converter applied the shift)
        if detector == "hkfd":
            patch_yaml(tree / "data" / "dataset" / "SK_cnn.yaml", {"one_indexed": False})
        for relative, edits in (extra or {}).items():
            patch_yaml(tree / relative, edits)
        return tree

    return _build


@pytest.fixture
def make_split(tmp_path):
    """Write an index-list npz over `n` events, as create_index.py does."""

    def _make(n: int, n_train: int, n_val: int, n_test: int, seed: int = 1234,
              name: str = "split.npz") -> Path:
        assert n_train + n_val + n_test <= n, "split larger than the dataset"
        order = np.random.default_rng(seed).permutation(n)
        path = tmp_path / name
        np.savez(path,
                 train_idxs=order[:n_train],
                 val_idxs=order[n_train:n_train + n_val],
                 test_idxs=order[n_train + n_val:n_train + n_val + n_test])
        return path

    return _make


def run_watchmal(config_path: Path, config_name: str, run_dir: Path, overrides: list[str],
                 searchpath: Path | None = None, timeout: int = 900) -> None:
    """Invoke main.py the way a user would; fail loudly with its output.

    `searchpath` defaults to `config_path`, which is right for the watchmal tree where the
    entry configs sit at the root. The caverns tree puts them in `main/` while the group
    defaults resolve against the tree root, so those two differ there.
    """
    cmd = [sys.executable, "main.py",
           "--config-path", str(config_path), "--config-name", config_name,
           f"hydra.searchpath=[{searchpath or config_path}]", f"hydra.run.dir={run_dir}",
           "gpu_list=[]", "seed=1234", *overrides]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                            timeout=timeout,
                            env={**os.environ, "NO_WATCHMAL_BANNER": "true"})
    if result.returncode != 0:
        pytest.fail(f"run failed ({config_name} {overrides}):\n"
                    f"--- stdout ---\n{result.stdout[-4000:]}\n"
                    f"--- stderr ---\n{result.stderr[-4000:]}")


# Overrides shared by every end-to-end run: one epoch, tiny batches, no worker processes.
#
# val_interval MUST be <= iterations-per-epoch (train events / batch_size). Validation is
# what writes the `_BEST` checkpoint, and `restore_best_state` then opens it by name - so
# if validation never fires the run dies with a bare
# `FileNotFoundError: ./outputs/..._BEST.pth` from three frames down, naming neither the
# cause nor the config key. 16 train events at batch 4 = 4 iterations, so 2 fires twice.
FAST = [
    "tasks.train.epochs=1",
    "tasks.train.val_interval=2",
    "tasks.train.num_val_batches=1",
    "tasks.train.data_loaders.train.batch_size=4",
    "tasks.train.data_loaders.train.num_workers=0",
    "tasks.train.data_loaders.train.pre_transforms=null",
    "tasks.train.data_loaders.validation.batch_size=4",
    "tasks.train.data_loaders.validation.num_workers=0",
    "tasks.evaluate.data_loaders.test.batch_size=4",
    "tasks.evaluate.data_loaders.test.num_workers=0",
]


# --------------------------------------------------------------------------------------
# Graph fixtures (T3.2, T3.5)
#
# The PyG graph datasets on the cluster are 30-85 MB for 200 events, and `edge_index` is
# ~80% of that - so they are not shippable. They do not need to be: the edges are derived
# (knn over the hit positions), so the graph fixture is BUILT from the same committed h5
# the CNN tests use. That also keeps the two families on one source of truth.
#
# Node features are written as (t, q) - two columns, NOT five. That is what the shipped
# configs require, and it is easy to get wrong:
#   * `Normalize.forward` loops `for ft_index in range(data.x.size(1))` and indexes
#     `feat_norm[:, ft_index]`, so feat_norm must be as wide as x. The shipped
#     `20inch_pmt_classification` gives 2 columns ([1900,1000] max / [550,0.01] min),
#     so a 5-column x raises IndexError deep inside the transform.
#   * `vanilla_gat_classifier` sets `in_channels: 2`.
#   * the dataset names in the shipped config say it outright: `train_tq_edges_xyz` -
#     node features t,q; xyz used only to build the edges.
# Feature 0 is time (range [550,1900]), feature 1 charge - consistent with
# `AddFeaturesInData.charge_index: 1`.
# --------------------------------------------------------------------------------------

def _knn_edge_index(pos, k: int):
    """k-nearest-neighbour graph, in pure torch.

    `torch_geometric.nn.knn_graph` needs `pyg-lib` or `torch_cluster` - compiled wheels
    that are version-fragile and are NOT part of the base requirements. CI should not have
    to install one just to build a fixture, so this uses cdist, which is always available.
    """
    import torch

    n = pos.shape[0]
    k = min(k, max(n - 1, 1))
    dist = torch.cdist(pos, pos)
    dist.fill_diagonal_(float("inf"))
    neighbours = dist.topk(k, largest=False).indices          # (n, k)
    source = neighbours.reshape(-1)
    target = torch.arange(n).repeat_interleave(k)
    return torch.stack([source, target], dim=0)


@pytest.fixture
def make_graph_dataset(tmp_path):
    """Build a PyG InMemoryDataset directory from the committed HK h5.

    Returns a callable (label, n_events, k) -> root path holding processed/data.pt, which
    is exactly what `PyGInMemory20inchDataset(pyg_data_folder_path=...)` loads.
    """
    import h5py
    import torch
    from torch_geometric.data import Data, InMemoryDataset

    spec = FIXTURES["hkfd"]
    geometry = np.load(spec["geometry"])["position"].astype(np.float32)

    def _build(label: int, n_wanted: int, k: int = 5, name: str | None = None) -> Path:
        root = tmp_path / (name or f"graph_{label}")
        (root / "processed").mkdir(parents=True, exist_ok=True)
        with h5py.File(spec["h5"], "r") as f:
            labels = f["labels"][:]
            bounds = np.append(f["event_hits_index"][:], f["hit_pmt"].shape[0])
            wanted = np.flatnonzero(labels == label)[:n_wanted]
            data_list = []
            for i in wanted:
                start, stop = bounds[i], bounds[i + 1]
                pmts = f["hit_pmt"][start:stop]
                pos = torch.tensor(geometry[pmts], dtype=torch.float32)
                t = torch.tensor(f["hit_time"][start:stop], dtype=torch.float32)
                q = torch.tensor(f["hit_charge"][start:stop], dtype=torch.float32)
                x = torch.stack([t, q], dim=1)   # (t, q); xyz only feeds the knn below
                data_list.append(Data(x=x, y=torch.tensor([int(labels[i])]),
                                      edge_index=_knn_edge_index(pos, k)))
        InMemoryDataset.save(data_list, str(root / "processed" / "data.pt"))
        return root

    return _build
