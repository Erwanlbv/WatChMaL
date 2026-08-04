"""
Tier 3 — T3.2 and T3.5: the graph (PyG / GAT) family, end to end.

The graph family takes a genuinely different route through the core than the CNN one:
the dataset is built parent-side in `main.py` and handed to the worker through the
`dataset=` constructor argument, and the engine splits `configure_dataset` from
`configure_data_loaders`. T3.1 never touches any of that.

T3.5 is the acceptance criterion for the whole evaluation path, and it is asserted over a
**concatenated** dataset with a test split spanning both halves — because that is where
the real bug lived: with a concatenation the saved index was the sub-dataset's *local*
one, so the dedup in `evaluate()` erased events. Silently: the metrics are accumulated
before the dedup, so they stayed correct while the artefacts `analysis/` reads were wrong.

Graph data is built from the same committed h5 the CNN tests use (see conftest); the real
PyG datasets are 30-85 MB per 200 events and are not shippable.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from tests.discovery import REPO_ROOT
from tests.tier3.conftest import run_watchmal

pytestmark = pytest.mark.graph

CAVERNS_TREE = REPO_ROOT / "tutorial" / "config" / "caverns"


@pytest.fixture
def graph_config_tree(tmp_path, make_graph_dataset):
    """Copy the shipped caverns tree and repoint it at two freshly built graph datasets.

    Two, not one, so the dataset is a genuine concatenation — which is what T3.5 needs.
    """
    import shutil

    def _build(n_per_class: int = 8, k: int = 5):
        tree = tmp_path / "caverns"
        shutil.copytree(CAVERNS_TREE, tree)
        roots = [str(make_graph_dataset(11, n_per_class, k, name="graph_e")),
                 str(make_graph_dataset(13, n_per_class, k, name="graph_mu"))]

        n_total = 2 * n_per_class
        order = np.random.default_rng(1234).permutation(n_total)
        split = tmp_path / "split.npz"
        # test split deliberately spans BOTH sub-datasets
        np.savez(split, train_idxs=order[:n_total - 6],
                 val_idxs=order[n_total - 6:n_total - 4], test_idxs=order[n_total - 4:])

        cfg_path = tree / "data" / "dataset" / "20inch_pmt_knn5_classification.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg["split_path"] = str(split)
        cfg["dataset_parameters"]["graph_folder_path"] = roots
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return tree, split, roots

    return _build


def test_t32_t35_graph_train_evaluate(graph_config_tree, tmp_path):
    """T3.2 — the graph family runs end to end; T3.5 — no test event is lost or mislabelled."""
    pytest.importorskip("torch_geometric")
    tree, split, roots = graph_config_tree()
    run_dir = tmp_path / "run"

    run_watchmal(tree / "main", "gat_classification", run_dir, [
        "tasks.train.epochs=1",
        "tasks.train.data_loaders.train.batch_size=2",
        "tasks.train.data_loaders.train.num_workers=0",
        "tasks.train.data_loaders.validation.batch_size=2",
        "tasks.train.data_loaders.validation.num_workers=0",
        "tasks.evaluate.data_loaders.test.batch_size=2",
        "tasks.evaluate.data_loaders.test.num_workers=0",
    ], searchpath=tree)

    outputs = run_dir / "outputs"
    written = sorted(p.name for p in outputs.iterdir())
    assert (outputs / "indices.npy").is_file(), f"no indices.npy; wrote {written}"
    assert (outputs / "log_val.csv").is_file(), f"no log_val.csv; wrote {written}"
    assert list(outputs.glob("log_train_*.csv")), f"no train CSV; wrote {written}"

    indices = np.load(outputs / "indices.npy")
    test_idxs = np.load(split)["test_idxs"]

    # ---- T3.5: the evaluation artefacts are complete and correctly attached ----
    assert len(indices) == len(test_idxs), (
        f"{len(indices)} rows saved for {len(test_idxs)} test events — events were lost "
        "or duplicated in the concat/dedup path")
    assert set(indices.tolist()) == set(test_idxs.tolist()), (
        f"saved indices {sorted(indices.tolist())} are not the test split "
        f"{sorted(test_idxs.tolist())} — the concat index is probably sub-dataset-local")
    assert len(set(indices.tolist())) == len(indices), "duplicate index in indices.npy"

    # every per-event output array must agree with indices on its first dimension
    for npy in outputs.glob("*.npy"):
        if npy.name == "indices.npy":
            continue
        arr = np.load(npy)
        assert arr.shape[0] == len(indices), (
            f"{npy.name} has {arr.shape[0]} rows against {len(indices)} indices")


@pytest.mark.transitional
def test_t32_graph_npy_names_are_pinned(graph_config_tree, tmp_path):
    """The graph engine's evaluate output names, pinned.

    The CI plan wanted `preds`/`targets` pinned so that moving the graph family onto the
    CNN naming (`softmax` / `predicted_<quantity>`) is a deliberate change. That move has
    since happened on this branch, so this asserts the CURRENT names and will fail loudly
    if they drift again.
    """
    pytest.importorskip("torch_geometric")
    tree, split, _ = graph_config_tree()
    run_dir = tmp_path / "run"
    run_watchmal(tree / "main", "gat_classification", run_dir, [
        "tasks.train.epochs=1",
        "tasks.train.data_loaders.train.batch_size=2",
        "tasks.train.data_loaders.train.num_workers=0",
        "tasks.train.data_loaders.validation.batch_size=2",
        "tasks.train.data_loaders.validation.num_workers=0",
        "tasks.evaluate.data_loaders.test.batch_size=2",
        "tasks.evaluate.data_loaders.test.num_workers=0",
    ], searchpath=tree)
    names = {p.name for p in (run_dir / "outputs").glob("*.npy")}
    assert "indices.npy" in names
    assert "softmax.npy" in names, (
        f"graph classification no longer writes softmax.npy (wrote {sorted(names)}); "
        "analysis/classification.py reads that name")


RETIRES_WITH = "P5 - one evaluate() writing one npy naming convention for every family"
