"""
Tier 2 — end-to-end runs on the local smoke datasets: e-/mu- separation, and vertex
regression.

Goal: everything else in Tier 1 and Tier 2 tests a piece. These two run the real thing —
`main.py` → `run.py` → engine → tracker → `outputs/` — on real detector graphs, and
assert the properties that only a finished run can show:

  * **the evaluation artefacts are complete.** `indices.npy` must hold exactly the test
    split. This is where a real bug lived: with a concatenated dataset the saved index
    was the sub-dataset's local one, so the dedup in `evaluate()` erased events —
    silently, because the metrics are accumulated *before* the dedup and stayed right.
    Only these files show it, and only over a concatenation.
  * **the run is reproducible.** With `deterministic: True`, two runs of the same config
    must agree bit for bit. Without that guarantee no future check can compare numbers,
    which is why it is asserted here rather than assumed. (Measured: without the flag
    two CPU runs already differ, because PyG's scatter reductions parallelise.)

The datasets are 200-event subsets carved off the cluster files; they are far too large
to commit, so these tests SKIP when the data is absent — which is the case on a CI
runner. They are the pre-cluster check you run by hand. The configs come from the
tracked `tutorial/` tree (copied to a temp dir and repointed), never from the private
`config/` workspace, so what is tested is what ships.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.discovery import REPO_ROOT

TUTORIAL_TREE = REPO_ROOT / "tutorial" / "config" / "caverns"


def _patch_yaml(path: Path, edits: dict) -> None:
    """Apply `{"a.b.c": value}` edits to a YAML file in place."""
    doc = yaml.safe_load(path.read_text())
    for dotted, value in edits.items():
        node = doc
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _config_tree(tmp_path: Path, edits: dict[str, dict]) -> Path:
    """Copy the shipped caverns config tree and repoint it at the local data."""
    tree = tmp_path / "config"
    shutil.copytree(TUTORIAL_TREE, tree)
    for relative, file_edits in edits.items():
        _patch_yaml(tree / relative, file_edits)
    return tree


def _run(tree: Path, config_name: str, run_dir: Path, seed: int) -> None:
    """Invoke main.py exactly as a user would, and fail loudly with its output."""
    cmd = [
        sys.executable, "main.py",
        "--config-path", str(tree / "main"), "--config-name", config_name,
        f"hydra.searchpath=[{tree}]", f"hydra.run.dir={run_dir}",
        "gpu_list=[]", "launch_wandb=False",
        "deterministic=True", f"seed={seed}",
        "tasks.train.epochs=1", "tasks.train.val_interval=10",
        "tasks.train.data_loaders.train.num_workers=0",
        "tasks.train.data_loaders.train.batch_size=10",
        "tasks.train.data_loaders.validation.num_workers=0",
        "tasks.evaluate.data_loaders.test.num_workers=0",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                            env={**dict(__import__("os").environ), "HK_BANNER": "0"})
    if result.returncode != 0:
        pytest.fail(f"run failed ({config_name}):\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")


def _outputs(run_dir: Path) -> Path:
    out = run_dir / "outputs"
    assert out.is_dir(), f"no outputs/ directory produced in {run_dir}"
    return out


def _assert_run_is_analysis_readable(run_dir: Path) -> None:
    out = _outputs(run_dir)
    for name in ("log_train_0.csv", "log_val.csv"):
        assert (out / name).is_file(), f"{name} missing — analysis/ cannot read this run"
    assert list(out.glob("*_BEST.pth")), "no best checkpoint was saved"

    from analysis.read import WatChMaLOutput

    train_epoch, train_loss, val_epoch, val_loss, _ = WatChMaLOutput(
        str(run_dir)
    ).read_training_log()
    assert len(np.asarray(train_loss).ravel()) > 0
    assert np.all(np.isfinite(np.asarray(train_loss).ravel())), "NaN/inf in the train loss"
    assert len(np.asarray(val_loss).ravel()) > 0


def _assert_test_set_is_complete(run_dir: Path, split_path: Path) -> None:
    """The property the concat-index bug broke."""
    out = _outputs(run_dir)
    saved = np.load(out / "indices.npy")
    expected = np.load(split_path)["test_idxs"]

    assert len(saved) == len(expected), (
        f"evaluate saved {len(saved)} rows for a {len(expected)}-event test split — "
        "test data was lost (a non-unique event index makes the dedup in evaluate() "
        "erase events, without changing any logged metric)"
    )
    assert set(saved.tolist()) == set(expected.tolist()), (
        "the saved indices are not the requested test events"
    )
    assert len(np.unique(saved)) == len(saved), "duplicate indices in indices.npy"
    for name in ("preds", "targets"):
        array = np.load(out / f"{name}.npy")
        assert array.shape[0] == len(expected), f"{name}.npy has {array.shape[0]} rows"


def _assert_identical(a: Path, b: Path) -> None:
    for name in ("log_train_0.csv", "log_val.csv"):
        assert (a / name).read_text() == (b / name).read_text(), (
            f"{name} differs between two deterministic runs of the same config"
        )
    for name in ("preds", "targets", "indices"):
        assert np.array_equal(np.load(a / f"{name}.npy"), np.load(b / f"{name}.npy")), (
            f"{name}.npy differs between two deterministic runs"
        )


# --------------------------------------------------------------------------- #
# e- / mu- separation (PID classification)
# --------------------------------------------------------------------------- #

@pytest.mark.graph
def test_pid_run_is_complete_and_reproducible(tmp_path, pid_datasets, make_split):
    """Two concatenated per-class datasets — the shape that made the index bug visible —
    run twice to pin reproducibility."""
    electron, muon = pid_datasets
    # 400 events: 0..199 e- (y=11), 200..399 mu- (y=13); the permuted split mixes them.
    split = make_split(n_events=400, n_train=60, n_val=20, n_test=20)

    tree = _config_tree(tmp_path, {
        "data/dataset/20inch_pmt_knn5_classification.yaml": {
            "split_path": str(split),
            "dataset_parameters.graph_folder_path": [str(electron), str(muon)],
        },
        # These files are q,t,x,y,z: charge is column 0 (the shipped value of 1 suits the
        # t,q,... files), and feat_norm shipped with 2 entries for a 5-feature dataset.
        "data/transforms/20inch_pmt_classification.yaml": {
            "transforms.AddFeaturesInData.charge_index": 0,
            "transforms.Normalize.feat_norm": [
                [1000, 1900, 3242.96, 3242.96, 3296.47],
                [0.01, 550, -3242.96, -3242.96, -3296.47],
            ],
        },
        "model/vanilla_gat_classifier.yaml": {"in_channels": 5},
    })

    first, second = tmp_path / "run1", tmp_path / "run2"
    _run(tree, "gat_classification", first, seed=4242)
    _assert_run_is_analysis_readable(first)
    _assert_test_set_is_complete(first, split)

    targets = np.load(first / "outputs" / "targets.npy")
    assert set(np.unique(targets).tolist()) <= {0, 1}, "MapLabels must map 13/11 to 0/1"
    assert len(np.unique(targets)) == 2, (
        "the test split should contain both classes — a single-class test set makes "
        "every classification metric meaningless"
    )
    preds = np.load(first / "outputs" / "preds.npy")
    assert preds.shape[1] == 2, f"expected 2 class scores per event, got {preds.shape}"

    _run(tree, "gat_classification", second, seed=4242)
    _assert_identical(first / "outputs", second / "outputs")


# --------------------------------------------------------------------------- #
# vertex regression
# --------------------------------------------------------------------------- #

@pytest.mark.graph
def test_vertex_regression_run_is_complete(tmp_path, vertex_dataset, make_split):
    """A different route through the same core: 3 continuous targets instead of 2
    classes, and a single (non-concatenated) dataset."""
    split = make_split(n_events=200, n_train=60, n_val=20, n_test=20)

    tree = _config_tree(tmp_path, {
        "data/dataset/20inch_pmt_knn5_vertex_regression.yaml": {
            "split_path": str(split),
            "dataset_parameters.graph_folder_path": [str(vertex_dataset)],
        },
    })

    run_dir = tmp_path / "run"
    _run(tree, "gat_vertex_regression", run_dir, seed=7)
    _assert_run_is_analysis_readable(run_dir)
    _assert_test_set_is_complete(run_dir, split)

    targets = np.load(run_dir / "outputs" / "targets.npy")
    preds = np.load(run_dir / "outputs" / "preds.npy")
    assert targets.shape[1] == 3, f"vertex targets are (x, y, z), got {targets.shape}"
    assert preds.shape == targets.shape
    assert np.all(np.isfinite(preds)), "non-finite predictions"


# --------------------------------------------------------------------------- #
# CLS-token readout
# --------------------------------------------------------------------------- #

@pytest.mark.graph
def test_cls_readout_run_is_complete(tmp_path, pid_datasets, make_split):
    """The same PID route through the classification-token readout.

    The mean-pool run above already covers the engine, the tracker and the output
    layout. What is exercised here is the readout that mean pooling does not reach: the
    virtual CLS nodes are appended inside the forward pass, so the batch the attention
    layers see is not the batch the data loader produced. The failure this catches is a
    CLS block that is laid out or sliced in the wrong order, which yields a well-formed
    run whose predictions belong to the wrong events.
    """
    electron, muon = pid_datasets
    split = make_split(n_events=400, n_train=60, n_val=20, n_test=20)

    tree = _config_tree(tmp_path, {
        "data/dataset/20inch_pmt_knn5_classification.yaml": {
            "split_path": str(split),
            "dataset_parameters.graph_folder_path": [str(electron), str(muon)],
        },
        # These files are q,t,x,y,z: charge is column 0, and the shipped feat_norm
        # carries 2 entries for a 5-feature dataset.
        "data/transforms/20inch_pmt_classification.yaml": {
            "transforms.AddFeaturesInData.charge_index": 0,
            "transforms.Normalize.feat_norm": [
                [1000, 1900, 3242.96, 3242.96, 3296.47],
                [0.01, 550, -3242.96, -3242.96, -3296.47],
            ],
        },
        "model/gat_cls_classifier.yaml": {"in_channels": 5, "hidden_channels": 32,
                                          "num_layers": 2},
    })

    run_dir = tmp_path / "run"
    _run(tree, "gat_cls_classification", run_dir, seed=4242)
    _assert_run_is_analysis_readable(run_dir)
    _assert_test_set_is_complete(run_dir, split)

    preds = np.load(run_dir / "outputs" / "preds.npy")
    targets = np.load(run_dir / "outputs" / "targets.npy")
    assert preds.shape[1] == 2, f"expected 2 class scores per event, got {preds.shape}"
    assert np.all(np.isfinite(preds)), "non-finite predictions"
    assert len(np.unique(targets)) == 2, "the test split should contain both classes"


# --------------------------------------------------------------------------- #
# Learning
# --------------------------------------------------------------------------- #

@pytest.mark.graph
@pytest.mark.parametrize("num_cls_tokens", [0, 1])
def test_model_overfits_a_small_sample(pid_datasets, num_cls_tokens):
    """The model can drive the training loss to zero on a sample it can memorise.

    Every other test in this file asserts that the pipeline runs and writes well-formed
    files. None of them would fail if the model were incapable of learning: a network
    whose input has been destroyed upstream still produces finite logits, a finite loss
    and a complete set of output arrays. Overfitting a handful of events is the cheapest
    property that separates a model that learns from one that only executes, and it is
    checked for both readouts because they share no code after the attention stack.

    Run directly on the model rather than through main.py: the engine is covered above,
    and 200 optimisation steps in-process cost seconds where 200 epochs of a subprocess
    run would cost minutes.
    """
    import torch
    from torch_geometric.data import Batch
    from watchmal.dataset.graph.pyg_in_memory_20inch_pmt import PyGInMemory20inchDataset
    from watchmal.model.gat import GraphAttentionNetwork

    torch.manual_seed(0)
    n_per_class = 4
    max_hits = 256          # see the note on cost below
    n_steps = 150

    graphs, labels = [], []
    for class_index, folder in enumerate(pid_datasets):
        dataset = PyGInMemory20inchDataset(
            pyg_data_folder_path=str(folder),
            pyg_data_file_names=["data.pt"],
            transforms=None,
        )
        for event in range(n_per_class):
            graph = dataset[event]
            # These events carry 137 to 7319 hits. Cost is linear in the edge count, and
            # the stored graphs are k=10 nearest-neighbour graphs, so a full event is
            # ~73 000 edges and 150 steps over eight of them is minutes of CPU. The
            # property under test — that the loss reaches zero on a memorisable sample —
            # does not depend on event size, so the graph is truncated to its first
            # max_hits nodes and the edges among them.
            keep = min(max_hits, graph.num_nodes)
            edge_index = graph.edge_index
            inside = (edge_index[0] < keep) & (edge_index[1] < keep)
            graph.edge_index = edge_index[:, inside]
            graph.x = graph.x[:keep]
            if getattr(graph, "pos", None) is not None:
                graph.pos = graph.pos[:keep]
            graph.num_nodes = keep
            graphs.append(graph)
            labels.append(class_index)

    batch = Batch.from_data_list(graphs)
    target = torch.tensor(labels, dtype=torch.long)

    # The shipped node features are q, t, x, y, z in detector units: charge in
    # photoelectrons, time in nanoseconds, coordinates in centimetres. Standardising
    # them per column is what the Normalize transform does in a real run; without it the
    # coordinate columns dominate the first linear layer by three orders of magnitude.
    features = batch.x
    batch.x = (features - features.mean(0)) / (features.std(0) + 1e-8)

    model = GraphAttentionNetwork(
        in_channels=5, hidden_channels=32, out_channels=2,
        num_layers=2, num_heads=4, num_cls_tokens=num_cls_tokens,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    first_loss, last_loss, accuracy = None, None, 0.0
    for step in range(n_steps):
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
        accuracy = (output.argmax(-1) == target).float().mean().item()

    assert last_loss < 0.05, (
        f"training loss stalled at {last_loss:.4f} after {n_steps} steps on "
        f"{len(graphs)} events (started at {first_loss:.4f}); the model is not learning "
        f"from its input"
    )
    assert accuracy == 1.0, f"accuracy {accuracy:.3f} on the memorised sample"


# --------------------------------------------------------------------------- #
# Graphs assembled at load time, edges built in the forward pass
# --------------------------------------------------------------------------- #

@pytest.mark.graph
def test_h5_graph_run_with_forward_pass_knn(tmp_path, h5_pid_file, hk_geometry, make_split):
    """The route that stores no graph at all.

    Everything above reads a PyG dataset whose edges were computed offline. Here the
    HDF5 file is the stored format, one Data is assembled per event as it is requested,
    and the k-nearest-neighbour graph is built inside the model's forward pass from
    data.pos. Three things can only fail on this route: the dataset factory has to accept
    dataset parameters that name no folder of stored tensors, the transforms have to run
    on an object the dataset just built rather than one sliced out of a collated file,
    and the model has to find coordinates where it expects them.
    """
    split = make_split(n_events=1985, n_train=60, n_val=20, n_test=20)

    tree = _config_tree(tmp_path, {
        "data/dataset/hk_h5_graph_classification.yaml": {
            "split_path": str(split),
            "dataset_parameters.h5_path": str(h5_pid_file),
            "dataset_parameters.geometry_file": str(hk_geometry),
        },
        "model/gat_cls_knn_classifier.yaml": {
            "hidden_channels": 32, "num_layers": 2, "knn_k": 6,
        },
    })

    run_dir = tmp_path / "run"
    _run(tree, "gat_cls_knn_h5_classification", run_dir, seed=4242)
    _assert_run_is_analysis_readable(run_dir)
    _assert_test_set_is_complete(run_dir, split)

    preds = np.load(run_dir / "outputs" / "preds.npy")
    targets = np.load(run_dir / "outputs" / "targets.npy")
    assert preds.shape[1] == 2, f"expected 2 class scores per event, got {preds.shape}"
    assert np.all(np.isfinite(preds)), (
        "non-finite predictions: the usual cause is a zero-charge hit reaching the "
        "logarithm in the Normalize transform, which the Threshold transform clamps"
    )
    assert len(np.unique(targets)) == 2, "the test split should contain both classes"


@pytest.mark.graph
def test_geometry_lookup_places_hits_correctly(h5_pid_file, hk_geometry):
    """Hit coordinates come from a geometry file keyed by PMT identifier, and the
    identifiers are not the row numbers.

    In hyperk_20inch_pmts.npz the tube_id column runs 19746, 19745, 19744, ..., so
    indexing position[] with a PMT identifier returns a valid point on the detector that
    belongs to a different PMT. Nothing raises, the hits still lie on the tank wall, and
    the summary statistics of the hit pattern barely move — the mean distance to the five
    nearest hits changes by under 1 %. What does move is the physics: a Cherenkov cone
    lights a region downstream of the interaction vertex along the particle's direction,
    so the charge-weighted centroid of the hits should lie near that direction, and under
    a scrambled assignment it does not.
    """
    import h5py
    from watchmal.dataset.graph.h5_graph_dataset import H5GraphDataset

    with h5py.File(h5_pid_file, "r") as h5:
        vertices = h5["positions"][:].squeeze(1)
        directions = np.stack(
            [h5[f"particle_dir_{axis}"][:] for axis in ("x", "y", "z")], axis=1
        )

    dataset = H5GraphDataset(str(h5_pid_file), str(hk_geometry))

    def median_opening_angle(source) -> float:
        angles = []
        for event in range(60):
            graph = source[event]
            charge = graph.x[:, 0].numpy()
            if charge.sum() <= 0:
                continue
            centroid = (graph.pos.numpy() * charge[:, None]).sum(0) / charge.sum()
            offset = centroid - vertices[event]
            offset = offset / (np.linalg.norm(offset) + 1e-9)
            direction = directions[event] / (np.linalg.norm(directions[event]) + 1e-9)
            angles.append(np.degrees(np.arccos(np.clip(offset @ direction, -1.0, 1.0))))
        return float(np.median(angles))

    with_lookup = median_opening_angle(dataset)

    scrambled = H5GraphDataset(str(h5_pid_file), str(hk_geometry))
    scrambled.pmt_to_geometry_row = None
    without_lookup = median_opening_angle(scrambled)

    assert with_lookup < 45.0, (
        f"median angle between the charge centroid and the true direction is "
        f"{with_lookup:.1f} deg; the hits are not where the geometry says they are"
    )
    assert without_lookup > 70.0, (
        f"indexing the geometry directly gave {without_lookup:.1f} deg, which is not the "
        f"random value this test expects. Either the geometry file is now ordered by "
        f"identifier, in which case the lookup table is redundant, or this check no "
        f"longer measures what it claims"
    )
