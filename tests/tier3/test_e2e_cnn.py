"""
Tier 3 — T3.1, T3.3, T3.6: the image (CNN) family, end to end on committed data.

T3.1 is the single most valuable test in the suite: it is the only one that wires
`main.py -> run.py -> build_model -> engine -> tracker -> dump_path` together, so it
catches integration breakage that every unit test passes through. It runs for *both*
detector fixtures, because they take genuinely different paths:

  toy_mpmt  CNNmPMTDataset  - 19-PMT modules, 41 channels, the transform pipeline
  hkfd      CNNDataset      - single PMTs, 2 channels, no transforms

Assertions are deliberately about **artefacts and shapes**, never about accuracy. Three of
the defects found while building this branch produced *plausible* output rather than an
error - a chance-level AUC from misaligned index order, a wrong-but-finite energy metric
from a (B,1)/(B,) broadcast - so a threshold assertion would have passed straight through
them while a shape/alignment assertion would not.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.tier3.conftest import FAST, n_events, run_watchmal


def test_t31_cnn_train_evaluate(detector, fixture_data, config_tree, make_split, tmp_path):
    """T3.1 — a full train -> validate -> _BEST -> restore -> evaluate chain writes every
    artefact `analysis/` needs, and `analysis/` can actually read them."""
    spec = fixture_data[detector]
    total = n_events(spec["h5"])
    split = make_split(total, n_train=16, n_val=8, n_test=12)
    tree = config_tree(detector, split)
    run_dir = tmp_path / "run"

    run_watchmal(tree, "tutorial_classification", run_dir, [
        f"data=ci_{detector}",
        "model=light_resnet18",
        f"model.num_input_channels={spec['num_input_channels']}",
        "model.num_output_channels=2",
        f"engine.label_set=[{spec['labels'][0]},{spec['labels'][1]}]",
        *FAST,
    ])

    outputs = run_dir / "outputs"
    for name in ("log_train_0.csv", "log_val.csv", "indices.npy", "softmax.npy", "labels.npy"):
        assert (outputs / name).is_file(), f"{name} missing from {sorted(p.name for p in outputs.iterdir())}"
    assert list(outputs.glob("*_BEST.pth")), "no _BEST checkpoint written"

    softmax = np.load(outputs / "softmax.npy")
    indices = np.load(outputs / "indices.npy")
    labels = np.load(outputs / "labels.npy")
    test_idxs = np.load(split)["test_idxs"]

    # every test event present exactly once, and the three arrays line up
    assert len(indices) == len(test_idxs), f"{len(indices)} rows for {len(test_idxs)} test events"
    assert set(indices.tolist()) == set(test_idxs.tolist()), "indices are not the test split"
    assert len(set(indices.tolist())) == len(indices), "duplicate index in indices.npy"
    assert softmax.shape == (len(test_idxs), 2), f"softmax shape {softmax.shape}"
    assert len(labels) == len(test_idxs)
    assert np.allclose(softmax.sum(axis=1), 1.0), "softmax rows do not sum to 1"
    # label_set maps the raw values onto 0..N-1
    assert set(np.unique(labels).tolist()) <= {0, 1}, f"labels not mapped: {np.unique(labels)}"

    # the CSV contract analysis/ depends on
    header = (outputs / "log_train_0.csv").read_text().splitlines()[0].split(",")
    assert header[:2] == ["iteration", "epoch"], header
    assert "loss" in header

    # and the real analysis class must load it
    from analysis.classification import WatChMaLClassification
    run = WatChMaLClassification(str(run_dir), run_label=detector,
                                 true_labels=labels[indices.argsort()])
    assert run.softmaxes.shape == softmax.shape
    assert run.plot_training_progression() is not None


def test_t33_checkpoint_round_trip(fixture_data, config_tree, make_split, tmp_path):
    """T3.3 — a saved checkpoint restores to identical parameters.

    Finding #4 in the CI plan: two `save_state`/`restore_state` implementations with
    different key sets survived P3, which is a cross-family data-loss hazard. This proves
    the image engine's own round-trip is lossless and pins the key set.
    """
    import torch

    detector = "toy_mpmt"  # cheapest fixture; the format is engine-level, not detector-level
    spec = fixture_data[detector]
    split = make_split(n_events(spec["h5"]), 16, 8, 12)
    tree = config_tree(detector, split)
    run_dir = tmp_path / "run"

    run_watchmal(tree, "tutorial_classification", run_dir, [
        f"data=ci_{detector}", "model=light_resnet18",
        f"model.num_input_channels={spec['num_input_channels']}",
        "model.num_output_channels=2",
        f"engine.label_set=[{spec['labels'][0]},{spec['labels'][1]}]",
        *FAST,
    ])

    ckpts = list((run_dir / "outputs").glob("*_BEST.pth"))
    assert len(ckpts) == 1, f"expected one _BEST checkpoint, got {ckpts}"
    state = torch.load(ckpts[0], map_location="cpu", weights_only=False)

    # The image engine's key set, pinned exactly. It is a strict SUBSET of the base
    # engine's, which also writes `epoch`, `seed` and `scheduler` - the CI plan's finding
    # #4, still true after P3. Pinned rather than "fixed" so that unifying the two is a
    # deliberate, reviewed change; see test_t33_checkpoint_formats_diverge below.
    assert set(state) == {"global_step", "optimizer", "state_dict"}, (
        f"image checkpoint key set changed: {sorted(state)}")

    # restoring into a fresh model reproduces the weights exactly
    from watchmal.model.light_resnet import light_resnet18
    model = light_resnet18(num_input_channels=spec["num_input_channels"],
                           num_output_channels=2, base_width=6, conv_pad_mode="circular")
    stripped = {k.replace("module.", "", 1): v for k, v in state["state_dict"].items()}
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    assert not unexpected, f"checkpoint carries unexpected keys: {unexpected[:5]}"
    assert not missing, f"checkpoint is missing model keys: {missing[:5]}"
    for name, param in model.state_dict().items():
        assert torch.equal(param, stripped[name]), f"parameter {name} changed on restore"


def test_t36_loader_edges_small_split(fixture_data, config_tree, make_split, tmp_path):
    """T3.6 — the defensive loader branches that only fire on small or odd splits.

    A batch_size larger than the split is exactly what a smoke run and a user's first
    dataset produce, and `num_workers == 0` is what a CPU run and a debugger want - torch
    rejects `multiprocessing_context` there, which used to make num_workers=0 impossible
    for the image family.
    """
    detector = "toy_mpmt"
    spec = fixture_data[detector]
    split = make_split(n_events(spec["h5"]), n_train=3, n_val=2, n_test=2)
    tree = config_tree(detector, split)
    run_dir = tmp_path / "run"

    # batch_size deliberately larger than every split, and no worker processes
    run_watchmal(tree, "tutorial_classification", run_dir, [
        f"data=ci_{detector}", "model=light_resnet18",
        f"model.num_input_channels={spec['num_input_channels']}",
        "model.num_output_channels=2",
        f"engine.label_set=[{spec['labels'][0]},{spec['labels'][1]}]",
        "tasks.train.epochs=1", "tasks.train.val_interval=1", "tasks.train.num_val_batches=1",
        "tasks.train.data_loaders.train.batch_size=64",
        "tasks.train.data_loaders.train.num_workers=0",
        "tasks.train.data_loaders.train.pre_transforms=null",
        "tasks.train.data_loaders.validation.batch_size=64",
        "tasks.train.data_loaders.validation.num_workers=0",
        "tasks.train.data_loaders.validation.drop_last=false",
        "tasks.evaluate.data_loaders.test.batch_size=64",
        "tasks.evaluate.data_loaders.test.num_workers=0",
    ])

    indices = np.load(run_dir / "outputs" / "indices.npy")
    assert len(indices) == 2, f"a 2-event test split produced {len(indices)} rows"
    assert set(indices.tolist()) == set(np.load(split)["test_idxs"].tolist())


RETIRES_WITH = "P5 - one save_state/restore_state, one checkpoint format for every engine"


@pytest.mark.transitional
def test_t33_checkpoint_formats_diverge(fixture_data, config_tree, make_split, tmp_path):
    """T3.3 (cross-format half) — document the two checkpoint formats and which direction
    is safe. This does not assert that the divergence is *good*; it pins it so P5 can
    remove it knowingly, and so the one unsafe direction is written down.

      image engine writes  {global_step, optimizer, state_dict}   (+ scaler under AMP)
      base engine  writes  {global_step, epoch, seed, optimizer, scheduler, state_dict}

    image -> base is SAFE: base's restore_state reads every optional key through
    `.get(...)`, so the smaller set degrades gracefully (epoch/seed fall back, and a
    missing seed is logged as "predates seed storage").

    base -> image is UNSAFE UNDER AMP: image's restore_state does
    `self.state_data['scaler']` unconditionally whenever `self.scaler is not None`, and
    the base engine never writes a `scaler` key - so restoring a graph/MR checkpoint into
    an AMP-enabled CNN run raises KeyError rather than warning.
    """
    import inspect
    import torch

    from watchmal.engine.base_engine import BaseEngine
    from watchmal.engine.images.reconstruction import ImageReconstructionEngine

    base_src = inspect.getsource(BaseEngine.save_state)
    image_src = inspect.getsource(ImageReconstructionEngine.save_state)
    base_keys = {k for k in ("global_step", "epoch", "seed", "optimizer", "scheduler",
                             "state_dict") if f'"{k}"' in base_src}
    image_keys = {k for k in ("global_step", "optimizer", "state_dict", "scaler")
                  if f"'{k}'" in image_src}

    assert image_keys - {"scaler"} < base_keys, (
        "the image checkpoint is no longer a strict subset of the base one; "
        f"image={sorted(image_keys)} base={sorted(base_keys)}")
    assert "scaler" in image_keys and "scaler" not in base_keys, (
        "the AMP `scaler` asymmetry changed - re-check restore_state on both engines")

    # and the unsafe direction really is unguarded: an unconditional subscript
    restore_src = inspect.getsource(ImageReconstructionEngine.restore_state)
    assert "self.state_data['scaler']" in restore_src, (
        "image restore_state no longer subscripts 'scaler' directly - if it now uses "
        ".get(), base -> image is safe under AMP and this test should be retired")
