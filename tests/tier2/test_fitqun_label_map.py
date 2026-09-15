"""
Tier 2 — fiTQun fit selection by particle label in `analysis/regression.py`.

`FitQun1ParticleFit` takes, for each event, the fiTQun single-particle fit whose entry in
`particle_label_map` equals the event's `true_labels` value; an event whose label has no
entry is left out of every fit's selection. Without a check, a sample in which no label has
an entry selects no fit and leaves the position, direction and momentum predictions at zero,
without an error unless `true_momenta` is given and a label is also absent from the mass
table. In a sample in which only some labels have an entry, the boolean-mask assignment that
assembles the predictions fails with a NumPy ValueError that names array shapes, not labels:
in the constructor for each truth array given, otherwise on the first access to a
prediction. A sample of several matched species fails in the same way: the class supports
one species per sample.

Both conventions reach this class. The default map holds class indices (gamma 0, electron
1, muon 2, pi0 3), while the mass lookup behind `true_momenta` requires PDG codes
(tests/tier2/test_momentum_mass_lookup.py). A label outside the map therefore raises
ValueError in the constructor. The fiTQun output is a stub carrying only the attributes the
class reads, so no ROOT file is needed.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from analysis.regression import FitQun1ParticleFit

N = 50
PDG_LABEL_MAP = {"gamma": 22, "electron": 11, "muon": 13, "pi0": 111}

# PDG Review of Particle Physics 2024, MeV.
ELECTRON_MASS_MEV = 0.51099895
MUON_MASS_MEV = 105.6583755


class _FiTQunStub:
    """The attributes of `analysis.read.FiTQunOutput` that `FitQun1ParticleFit` reads. Each fit
    has its own momentum and random positions and directions, so the selected fit is recognisable."""

    def __init__(self, n):
        rng = np.random.default_rng(3)
        self.chain = np.zeros(n)
        for fit, momentum in (("electron", 500.0), ("muon", 600.0), ("pi0", 700.0)):
            setattr(self, f"{fit}_momentum", np.full(n, momentum))
            setattr(self, f"{fit}_position", rng.normal(0.0, 30.0, (n, 3)))
            direction = rng.normal(size=(n, 3))
            setattr(self, f"{fit}_direction", direction / np.linalg.norm(direction, axis=1, keepdims=True))


def _fit(stub, **kwargs):
    true_directions = np.tile([0.0, 0.0, 1.0], (N, 1))
    return FitQun1ParticleFit(stub, "stub", true_positions=np.zeros((N, 3)), true_directions=true_directions, **kwargs)


@pytest.mark.parametrize("labels", [np.full(N, 2, dtype=np.int32), 2], ids=["array", "python-int"])
def test_default_map_selects_the_fit_named_by_a_class_index(labels):
    stub = _FiTQunStub(N)

    run = _fit(stub, true_labels=labels)

    assert list(run.particle_indices) == ["muon"]
    np.testing.assert_array_equal(run.position_prediction, stub.muon_position)
    np.testing.assert_array_equal(run.direction_prediction, stub.muon_direction)
    np.testing.assert_array_equal(run.momentum_prediction, stub.muon_momentum)


@pytest.mark.parametrize(
    ("labels", "label_map", "named"),
    [
        (np.full(N, 13, dtype=np.int32), None, "[13]"),
        (np.full(N, 2, dtype=np.int32), PDG_LABEL_MAP, "[2]"),
        (np.full(N, -13, dtype=np.int32), PDG_LABEL_MAP, "[-13]"),
        (np.array([1, 2, 5, 5, -1] * (N // 5), dtype=np.int32), None, "[-1, 5]"),
    ],
    ids=["pdg-labels-default-map", "class-index-labels-pdg-map", "antiparticle-code", "partly-unmatched"],
)
def test_a_label_outside_the_map_raises_in_the_constructor(labels, label_map, named):
    with pytest.raises(ValueError, match=re.escape(f"true_labels {named} have no entry in particle_label_map")):
        _fit(_FiTQunStub(N), true_labels=labels, particle_label_map=label_map)


@pytest.mark.parametrize(
    ("code", "fit", "mass"),
    [(13, "muon", MUON_MASS_MEV), (-11, "electron", ELECTRON_MASS_MEV)],
    ids=["mu-", "e+"],
)
def test_absolute_pdg_codes_with_a_pdg_map_select_the_fit_and_give_the_mass(code, fit, mass):
    stub = _FiTQunStub(N)
    true_momenta = np.linspace(50.0, 1500.0, N)

    run = _fit(stub, true_momenta=true_momenta, true_labels=np.abs(np.full(N, code, dtype=np.int32)),
               particle_label_map=PDG_LABEL_MAP)

    assert list(run.particle_indices) == [fit]
    np.testing.assert_array_equal(run.momentum_prediction, getattr(stub, f"{fit}_momentum"))
    np.testing.assert_allclose(run.true_energies, np.sqrt(true_momenta**2 + mass**2), rtol=1e-12, atol=0.0)


def test_class_index_labels_raise_in_the_mass_lookup_when_true_momenta_are_given():
    with pytest.raises(ValueError, match="PDG Monte Carlo particle codes"):
        _fit(_FiTQunStub(N), true_momenta=np.full(N, 550.0), true_labels=np.full(N, 2, dtype=np.int32))
