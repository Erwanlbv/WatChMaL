"""
Tier 2 — the 'three_momenta' target of `H5CommonDataset`, read from an HDF5 file.

`H5CommonDataset.load_target('three_momenta')` multiplies the unit vector of the stored
'angles' by |p| = sqrt(E^2 - m^2), where E is the stored 'energies' array and m is looked
up from the raw 'labels' array by |PDG code| (tests/tier2/test_momentum_mass_lookup.py).
The engines call `set_target` before the dataset is initialised, so the targets are loaded
on the first item access, inside the DataLoader workers when there are any.

These tests write a flat HDF5 file holding only the arrays `H5Dataset` reads for this
target: 'event_hits_index', 'hit_pmt', 'hit_time' and 'hit_charge' for initialisation, and
'angles', 'energies' and 'labels' for the target. They need numpy, h5py, torch and pytest,
all installed by requirements-ci.txt.
"""

from __future__ import annotations

import logging

import h5py
import numpy as np
import pytest

from watchmal.dataset.h5_dataset import H5Dataset
from watchmal.utils import math as wm_math
from watchmal.utils.math import direction_from_angles

# PDG Review of Particle Physics 2024, MeV.
EXPECTED_MASSES = {11: 0.51099895, 13: 105.6583755, 22: 0.0, 111: 134.9768}

PDG_LABELS = [11, -13, 22, 111]
ENERGIES = [200.0, 300.0, 400.0, 500.0]
ANGLES = [[0.3, 0.1], [1.2, -2.0], [2.5, 0.7], [1.6, 3.0]]


@pytest.fixture
def make_dataset(tmp_path):
    """Write a flat HDF5 file with two hits per event and return an `H5Dataset` over it, uninitialised."""
    opened = []

    def _make(labels, energies, angles):
        n_events = len(labels)
        n_hits = 2 * n_events
        path = tmp_path / f"flat_{len(opened)}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("event_hits_index", data=np.arange(0, n_hits, 2, dtype=np.int64))
            f.create_dataset("hit_pmt", data=np.arange(n_hits, dtype=np.int32))
            f.create_dataset("hit_time", data=np.linspace(0.0, 10.0, n_hits, dtype=np.float32))
            f.create_dataset("hit_charge", data=np.ones(n_hits, dtype=np.float32))
            f.create_dataset("labels", data=np.asarray(labels, dtype=np.int32))
            f.create_dataset("energies", data=np.asarray(energies, dtype=np.float32).reshape(n_events, 1))
            f.create_dataset("angles", data=np.asarray(angles, dtype=np.float32))
        dataset = H5Dataset(str(path))
        opened.append(dataset)
        return dataset

    yield _make
    for dataset in opened:
        if dataset.h5_file is not None:
            dataset.h5_file.close()


def _convention_records(caplog):
    return [r for r in caplog.records
            if r.name == wm_math.__name__ and r.levelno == logging.WARNING and "looked up by PDG code" in r.getMessage()]


def test_pdg_labels_give_the_direction_times_the_momentum_magnitude(make_dataset):
    dataset = make_dataset(PDG_LABELS, ENERGIES, ANGLES)
    dataset.set_target("three_momenta")

    items = [dataset[i]["three_momenta"] for i in range(len(PDG_LABELS))]

    energies = np.array(ENERGIES)
    masses = np.array([EXPECTED_MASSES[abs(code)] for code in PDG_LABELS])
    expected = direction_from_angles(np.array(ANGLES)) * np.sqrt(energies**2 - masses**2)[:, None]
    assert all(item.shape == (3,) and item.dtype == np.float32 for item in items)
    np.testing.assert_allclose(np.stack(items), expected, rtol=1e-5, atol=1e-4)


def test_mapped_class_indices_raise_on_the_first_item_access(make_dataset):
    dataset = make_dataset([0, 1], ENERGIES[:2], ANGLES[:2])
    dataset.set_target("three_momenta")

    with pytest.raises(ValueError, match=r"Particle labels \[0, 1\] have no entry in the mass table") as error:
        dataset[0]
    assert "rewritten with PDG codes" in str(error.value)


@pytest.mark.parametrize("target_key", ["three_momenta", ["directions", "three_momenta"]], ids=["str", "list"])
def test_set_target_logs_the_convention_once_and_item_access_does_not_repeat_it(make_dataset, caplog, monkeypatch,
                                                                                target_key):
    # A logging configuration applied by an earlier test may have disabled the logger.
    monkeypatch.setattr(logging.getLogger(wm_math.__name__), "disabled", False)
    wm_math._reset_mass_convention_warning()
    caplog.set_level(logging.WARNING, logger=wm_math.__name__)
    dataset = make_dataset(PDG_LABELS, ENERGIES, ANGLES)

    dataset.set_target(target_key)

    assert not dataset.initialized
    assert len(_convention_records(caplog)) == 1

    for i in range(len(PDG_LABELS)):
        dataset[i]

    assert dataset.initialized
    assert len(_convention_records(caplog)) == 1
