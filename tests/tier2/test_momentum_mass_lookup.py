"""
Tier 2 — particle mass lookup in the momentum-energy conversion.

`momentum_from_energy` and `energy_from_momentum` compute |p| = sqrt(E^2 - m^2) and
E = sqrt(|p|^2 + m^2), with m looked up from the particle label. The dataset target
'three_momenta' and the momentum paths of `analysis/regression.py` go through them. The
lookup was once an index into the array (0, 0.511, 105.7, 134.98), which failed in two
ways:

* labels stored as PDG codes (11, 13) raised ``IndexError``;
* labels stored as class indices under the {13: 0, 11: 1, 22: 2} mapping gave muons the
  photon mass and photons the muon mass: the momenta were wrong, NaN for photons below
  105.7 MeV, and no error was raised.

The masses are now looked up by |PDG code|. A label whose absolute value has no entry in
the table raises ValueError; a class index equal to a listed code would be accepted as
that species. These tests fix that contract. The expected masses are written out here
rather than read from the module, so a typo in the table fails a test. They need numpy,
pytest and logging only; the dataset path is tested in
tests/tier2/test_three_momenta_target.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from watchmal.utils import math as wm_math
from watchmal.utils.math import PDG_MASSES_MEV, energy_from_momentum, momentum_from_energy

# PDG Review of Particle Physics 2024, MeV.
EXPECTED_MASSES = {
    11: 0.51099895,
    13: 105.6583755,
    22: 0.0,
    111: 134.9768,
    211: 139.57039,
}

# Total energies from 1 keV above threshold to 5 GeV above it, geometrically spaced.
KINETIC = np.geomspace(1e-3, 5000.0, 64)


def test_table_covers_the_expected_species():
    assert dict(PDG_MASSES_MEV) == EXPECTED_MASSES


@pytest.mark.parametrize("code", sorted(EXPECTED_MASSES))
def test_momentum_matches_the_invariant_mass_relation(code):
    mass = EXPECTED_MASSES[code]
    energy = mass + KINETIC
    labels = np.full(energy.shape, code, dtype=np.int32)

    momentum = momentum_from_energy(energy, labels)

    np.testing.assert_allclose(momentum, np.sqrt(energy**2 - mass**2), rtol=1e-12, atol=0.0)
    assert np.all(np.isfinite(momentum))


@pytest.mark.parametrize("code", sorted(EXPECTED_MASSES))
def test_energy_matches_the_invariant_mass_relation(code):
    mass = EXPECTED_MASSES[code]
    momentum = np.geomspace(1e-3, 5000.0, 64)
    labels = np.full(momentum.shape, code, dtype=np.int32)

    np.testing.assert_allclose(energy_from_momentum(momentum, labels), np.sqrt(momentum**2 + mass**2),
                               rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("code", [11, 13, 211])
def test_antiparticle_takes_the_mass_of_its_particle(code):
    energy = EXPECTED_MASSES[code] + KINETIC
    particle = momentum_from_energy(energy, np.full(energy.shape, code))
    antiparticle = momentum_from_energy(energy, np.full(energy.shape, -code))

    np.testing.assert_array_equal(antiparticle, particle)
    np.testing.assert_array_equal(energy_from_momentum(particle, -code), energy_from_momentum(particle, code))


def test_scalar_energy_and_code_give_a_scalar():
    result = momentum_from_energy(500.0, 13)
    assert np.ndim(result) == 0
    assert result == pytest.approx(np.sqrt(500.0**2 - EXPECTED_MASSES[13]**2), rel=1e-12)

    result = energy_from_momentum(500.0, -11)
    assert np.ndim(result) == 0
    assert result == pytest.approx(np.sqrt(500.0**2 + EXPECTED_MASSES[11]**2), rel=1e-12)


def test_vector_energy_with_vector_codes_keeps_the_shape():
    energy = np.array([200.0, 300.0, 400.0, 500.0])
    codes = np.array([11, 13, -13, 22])
    assert momentum_from_energy(energy, codes).shape == (4,)
    assert energy_from_momentum(energy, codes).shape == (4,)


def test_column_energy_with_vector_codes_keeps_the_column_shape_and_dtype():
    """Energies as an (N, 1) float32 column, the layout of the 'energies' array on disk, with (N,) int32 codes.
    The dataset target passes squeezed (N,) energies instead (H5CommonDataset.load_target squeezes every
    array it reads); that path is tested in tests/tier2/test_three_momenta_target.py."""
    energy = np.array([[200.0], [300.0], [400.0], [500.0]], dtype=np.float32)
    codes = np.array([11, 13, 13, 11], dtype=np.int32)

    momentum = momentum_from_energy(energy, codes)

    assert momentum.shape == (4, 1)
    assert momentum.dtype == np.float32
    masses = np.array([EXPECTED_MASSES[c] for c in codes])[:, None]
    np.testing.assert_allclose(momentum, np.sqrt(energy.astype(np.float64)**2 - masses**2), rtol=1e-6)


def test_scalar_energy_with_vector_codes_broadcasts_over_the_codes():
    assert momentum_from_energy(500.0, np.array([11, 13, 111])).shape == (3,)


def test_round_trip_energy_momentum_energy():
    codes = np.tile(np.array([11, 13, 22, 111, 211, -11, -13, -211]), KINETIC.size)
    masses = np.array([EXPECTED_MASSES[abs(c)] for c in codes])
    energy = masses + np.repeat(KINETIC, 8)

    recovered = energy_from_momentum(momentum_from_energy(energy, codes), codes)

    np.testing.assert_allclose(recovered, energy, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("convert", [momentum_from_energy, energy_from_momentum], ids=lambda f: f.__name__)
def test_mapped_class_indices_raise_value_error(convert):
    with pytest.raises(ValueError, match=r"Particle labels \[0, 1, 2\] have no entry in the mass table") as error:
        convert(np.array([500.0, 500.0, 500.0]), np.array([0, 1, 2]))
    assert "must be PDG Monte Carlo particle codes" in str(error.value)
    assert "converted back to PDG codes" in str(error.value)
    assert "rewritten with PDG codes in 'labels'" in str(error.value)


def test_value_error_names_only_the_offending_labels():
    with pytest.raises(ValueError, match=r"Particle labels \[-1, 0\] have no entry"):
        momentum_from_energy(np.full(5, 500.0), np.array([11, 13, 0, -1, 0]))


def test_an_index_ordered_mass_array_is_rejected():
    with pytest.raises(TypeError, match="mapping from absolute PDG code"):
        momentum_from_energy(500.0, 1, particle_masses=np.array((0, 0.511, 105.7, 134.98)))


def test_convention_warning_is_logged_exactly_once(caplog, monkeypatch):
    logger = logging.getLogger(wm_math.__name__)
    # A logging configuration applied by an earlier test may have disabled the logger.
    monkeypatch.setattr(logger, "disabled", False)
    wm_math._reset_mass_convention_warning()
    caplog.set_level(logging.WARNING, logger=wm_math.__name__)

    momentum_from_energy(np.array([200.0, 300.0]), np.array([11, 13]))
    wm_math.log_mass_convention()
    momentum_from_energy(500.0, 22)
    energy_from_momentum(np.array([200.0, 300.0]), np.array([-11, 211]))
    with pytest.raises(ValueError):
        momentum_from_energy(500.0, 0)

    records = [r for r in caplog.records if r.name == wm_math.__name__ and r.levelno == logging.WARNING]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "looked up by PDG code" in message
    assert "matched by absolute value" in message
    for code, species, mass in [(11, "electron", 0.51099895), (13, "muon", 105.6583755), (22, "photon", 0.0),
                                (111, "neutral pion", 134.9768), (211, "charged pion", 139.57039)]:
        assert f"{code} {species} {mass}" in message

    wm_math._reset_mass_convention_warning()
    wm_math.log_mass_convention()
    momentum_from_energy(500.0, 13)
    records = [r for r in caplog.records if r.name == wm_math.__name__ and r.levelno == logging.WARNING]
    assert len(records) == 2
