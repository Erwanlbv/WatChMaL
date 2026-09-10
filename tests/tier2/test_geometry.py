"""
Tier 2 — the geometry file reader.

Every dataset that needs PMT coordinates reads them from a compressed numpy file, using
the PMT identifier as a row number. That is correct only for a file whose rows are already
ordered by identifier, and the geometry files in use disagree on that, on the name of the
identifier column and on the name of the orientation column. `DetectorGeometry` resolves
all three at load time and hands back arrays in identifier order, so the row indexing at
the call sites becomes correct without being rewritten. The graph datasets read through
it; the image and point-cloud datasets do not, and remain subject to the row-order
assumption.

These tests are synthetic and need no detector data, so they run in CI. The one check
against a real file skips when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from watchmal.dataset.common.geometry import DetectorGeometry
from tests.tier2.conftest import DEFAULT_GEOMETRY


def _write(tmp_path: Path, name: str = "geometry.npz", **arrays) -> str:
    path = tmp_path / name
    np.savez(path, **arrays)
    return str(path)


def _ring(n: int) -> np.ndarray:
    """n distinguishable points, so a permutation is visible in the values."""
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angle), np.sin(angle), np.arange(n, dtype=float)], axis=1).astype(np.float32)


def test_descending_identifiers_are_reordered(tmp_path):
    """The shape of hyperk_20inch_pmts.npz: tube_id runs N, N-1, ..., 1."""
    n = 8
    position = _ring(n)
    tube_id = np.arange(n, 0, -1, dtype=np.int32)      # 8, 7, ..., 1

    geometry = DetectorGeometry(_write(tmp_path, position=position, tube_id=tube_id))

    assert geometry.reordered is True
    assert geometry.id_offset == 1
    # Row p must now hold the PMT whose zero-based identifier is p, i.e. tube_id p + 1,
    # which the descending column places at row n - 1 - p.
    for pmt in range(n):
        np.testing.assert_array_equal(geometry.position[pmt], position[n - 1 - pmt])


def test_ordered_identifiers_leave_the_rows_alone(tmp_path):
    """The shape of sk_pmt_geom_dump_from_neut.npz and wcte_mpmt_pmts.npz."""
    n = 8
    position = _ring(n)
    geometry = DetectorGeometry(
        _write(tmp_path, position=position, tube_id=np.arange(1, n + 1, dtype=np.int32))
    )

    assert geometry.reordered is False, "an ordered file must not be permuted"
    np.testing.assert_array_equal(geometry.position, position)


@pytest.mark.parametrize("identifier_column", ["tube_id", "tube_no"])
def test_both_identifier_column_names_are_recognised(tmp_path, identifier_column):
    n = 6
    arrays = {"position": _ring(n), identifier_column: np.arange(n, 0, -1, dtype=np.int32)}
    geometry = DetectorGeometry(_write(tmp_path, **arrays))

    assert geometry.identifier_key == identifier_column
    assert geometry.reordered is True


@pytest.mark.parametrize("direction_column", ["direction", "orientation"])
def test_both_direction_column_names_are_recognised(tmp_path, direction_column):
    """hyperk_20inch_pmts.npz and sk_pmt_geom_dump_from_neut.npz carry only `direction`,
    so code reading `orientation` alone raises KeyError on both."""
    n = 6
    direction = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n, 1))
    arrays = {"position": _ring(n), direction_column: direction}
    geometry = DetectorGeometry(_write(tmp_path, **arrays))

    assert geometry.direction_key == direction_column
    np.testing.assert_array_equal(geometry.directions_of(np.arange(n)), direction)


def test_missing_direction_column_raises_only_when_used(tmp_path):
    geometry = DetectorGeometry(_write(tmp_path, position=_ring(4)))
    assert geometry.direction is None
    with pytest.raises(KeyError, match="no orientation column"):
        geometry.directions_of(np.arange(4))


def test_file_without_identifiers_keeps_its_rows(tmp_path):
    position = _ring(5)
    geometry = DetectorGeometry(_write(tmp_path, position=position))

    assert geometry.identifier_key is None
    assert geometry.reordered is False
    np.testing.assert_array_equal(geometry.position, position)

    with pytest.raises(KeyError, match="no identifier column"):
        DetectorGeometry(_write(tmp_path, "b.npz", position=position), index_by="identifier")


def test_index_by_row_reproduces_the_historical_behaviour(tmp_path):
    """The escape hatch for reproducing a run made without honouring the identifier, and
    for matching a dataset that still indexes the file by row."""
    n = 8
    position = _ring(n)
    path = _write(tmp_path, position=position, tube_id=np.arange(n, 0, -1, dtype=np.int32))

    legacy = DetectorGeometry(path, index_by="row")
    assert legacy.reordered is False
    np.testing.assert_array_equal(legacy.position, position)

    assert not np.array_equal(legacy.position, DetectorGeometry(path).position), (
        "the two indexing modes must differ for a permuted file, or this option is moot"
    )


def test_identifiers_that_are_not_a_permutation_are_rejected(tmp_path):
    path = _write(tmp_path, position=_ring(4), tube_id=np.array([1, 2, 2, 9], dtype=np.int32))
    with pytest.raises(ValueError, match="not a permutation"):
        DetectorGeometry(path)


def test_cylindrical_columns_are_computed_when_absent(tmp_path):
    """r = hypot(x, y) and theta = arctan2(y, x), the convention the stored columns use,
    verified against hyperk_20inch_pmts.npz to 2.4e-4 cm and 2.4e-7 rad."""
    position = np.array([[3.0, 4.0, 0.0], [0.0, 2.0, 1.0]], dtype=np.float32)
    geometry = DetectorGeometry(_write(tmp_path, position=position))

    np.testing.assert_allclose(geometry.r, [5.0, 2.0], atol=1e-5)
    np.testing.assert_allclose(geometry.theta, [np.arctan2(4, 3), np.pi / 2], atol=1e-6)
    np.testing.assert_allclose(geometry.cos_theta, np.cos(geometry.theta), atol=1e-6)
    np.testing.assert_allclose(geometry.sin_theta, np.sin(geometry.theta), atol=1e-6)


def test_stored_cylindrical_columns_are_reordered_with_the_positions(tmp_path):
    n = 6
    position = _ring(n)
    geometry = DetectorGeometry(_write(
        tmp_path,
        position=position,
        tube_id=np.arange(n, 0, -1, dtype=np.int32),
        r=np.arange(n, dtype=np.float32),
        theta=np.arange(n, dtype=np.float32) * 0.1,
    ))

    # Row p holds the values that were at row n - 1 - p.
    np.testing.assert_allclose(geometry.r, np.arange(n - 1, -1, -1))
    np.testing.assert_allclose(geometry.cos_theta, np.cos(geometry.theta), atol=1e-6)


def test_the_shipped_hyperk_geometry_needs_reordering():
    """The regression this component exists for. If this ever fails because the file was
    rewritten in identifier order, the reader is simply a no-op on it and nothing else
    changes."""
    if not Path(DEFAULT_GEOMETRY).is_file():
        pytest.skip(f"geometry not available at {DEFAULT_GEOMETRY}")

    geometry = DetectorGeometry(str(DEFAULT_GEOMETRY))
    assert geometry.identifier_key == "tube_id"
    assert geometry.direction_key == "direction"
    assert geometry.reordered is True, (
        "hyperk_20inch_pmts.npz is expected to be stored out of identifier order"
    )
    assert len(geometry) == 19746
