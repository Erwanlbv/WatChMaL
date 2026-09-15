"""
Detector geometry read from a compressed numpy file, keyed by PMT identifier.

An event HDF5 file records which photomultipliers (PMTs) were hit, as identifiers in
``hit_pmt``, but not where those PMTs are. The coordinates live in a separate geometry
file, one row per PMT, and every dataset in this package that needs positions or
orientations reads one.

Those datasets use the identifier as a **row number**:
``geo_positions[event_hit_pmts]``. That is correct only when the file's rows happen to be
ordered by identifier, which is a property of the file and not of the format. Of the
geometry files in use, ``sk_pmt_geom_dump_from_neut.npz`` and ``wcte_mpmt_pmts.npz`` are
ordered (their ``tube_id`` column is ``1, 2, 3, ...``), while ``hyperk_20inch_pmts.npz``
is not: its ``tube_id`` column runs ``19746, 19745, 19744, ...``. Indexing that file by
row returns a valid point on the detector belonging to a **different** PMT. Nothing
raises, the hits still lie on the tank wall, and the mean distance to the five nearest
hits changes by under one per cent. What changes is the physics: a Cherenkov cone lights
a region downstream of the interaction vertex along the particle's direction, so the
charge-weighted centroid of the hits should lie near that direction. Over 150 events of
``hkfd_emu_rwcs_2k_watchmal.h5`` the median angle between that centroid and the true
``particle_dir`` is 24.9 degrees when the identifier column is honoured and 88.6 degrees
when it is not, the latter being the value expected of a random assignment.

This class resolves the identifier column once, at load time, and returns every array
reordered so that row ``p`` holds the PMT whose zero-based identifier is ``p``. Call
sites therefore keep indexing by row and become correct without further change.

**It is used by the graph datasets only.** The image datasets
(:mod:`watchmal.dataset.cnn`, :mod:`watchmal.dataset.cnn_mpmt`) and the point-cloud
ones (:mod:`watchmal.dataset.pointnet`) still read the file directly and index it by
row, and remain subject to everything described above. Converting them is a change to
those pipelines rather than to the graph one, and is out of the scope of the work that
introduced this module.

The formats differ in three further ways, all absorbed here:

* the identifier column is named ``tube_id`` in some files and ``tube_no`` in others, and
  is absent from a third group;
* the orientation column is named ``direction`` in some files and ``orientation`` in
  others — ``hyperk_20inch_pmts.npz`` and ``sk_pmt_geom_dump_from_neut.npz`` carry only
  the former, so code reading ``orientation`` alone fails on both;
* the cylindrical coordinates ``r``, ``theta``, ``cos_theta`` and ``sin_theta`` are
  stored in some files and absent from others. They are computed from the positions when
  absent, with the convention the stored columns follow, verified against
  ``hyperk_20inch_pmts.npz``: ``r = hypot(x, y)`` and ``theta = arctan2(y, x)``, the
  azimuth about the detector axis (agreement to 2.4e-4 cm and 2.4e-7 rad respectively).
"""

import numpy as np

# WatChMaL imports
from watchmal.utils.logging_utils import setup_logging

log = setup_logging(__name__)


_IDENTIFIER_KEYS = ('tube_id', 'tube_no')
_DIRECTION_KEYS = ('direction', 'orientation')


def _first_present(archive, keys):
    for key in keys:
        if key in archive:
            return key, np.asarray(archive[key])
    return None, None


class DetectorGeometry:
    """PMT positions and orientations, indexed by zero-based PMT identifier.

    Parameters
    ----------
    geometry_file : str
        Path to the ``.npz``. It must hold a ``position`` array of shape ``(n_pmts, 3)``.
        An identifier column (``tube_id`` or ``tube_no``), an orientation column
        (``direction`` or ``orientation``), the cylindrical columns and a ``region``
        column are each used when present.
    id_offset : int or "auto"
        Value added to a zero-based identifier from the HDF5 file to obtain the identifier
        used by the geometry file. ``"auto"`` reads it from the smallest identifier in the
        file, which is 1 for every file shipped with a ``tube_id`` column and 0 for a
        zero-based one.
    index_by : {"auto", "identifier", "row"}
        ``"auto"`` reorders by the identifier column when one is present, and leaves the
        rows alone otherwise. ``"row"`` forces the historical behaviour, in which the
        identifier is used as a row number; it exists to reproduce an earlier run, and
        gives wrong coordinates for a file whose rows are not ordered by identifier.
        ``"identifier"`` requires the column and fails if it is absent.

    Attributes
    ----------
    position : ndarray, shape (n_pmts, 3), float32
        Cartesian coordinates, in the units of the file (centimetres for the files in use).
    direction : ndarray or None, shape (n_pmts, 3), float32
        Unit vector along the PMT axis, when the file carries one.
    r, theta, cos_theta, sin_theta : ndarray, shape (n_pmts,), float32
        Cylindrical coordinates about the detector axis.
    region : ndarray or None, shape (n_pmts,)
        Detector region index, with ``region_names`` giving the labels.
    """

    def __init__(self, geometry_file, id_offset='auto', index_by='auto'):
        if index_by not in ('auto', 'identifier', 'row'):
            raise ValueError(
                f"index_by must be 'auto', 'identifier' or 'row', got {index_by!r}"
            )

        archive = np.load(geometry_file, allow_pickle=True)
        if 'position' not in archive:
            raise KeyError(
                f"{geometry_file} has no 'position' array; it holds {list(archive.keys())}"
            )

        position = np.asarray(archive['position'], dtype=np.float32)
        n_pmts = position.shape[0]

        identifier_key, identifiers = _first_present(archive, _IDENTIFIER_KEYS)
        direction_key, direction = _first_present(archive, _DIRECTION_KEYS)

        order, self.id_offset = self._resolve_order(
            geometry_file, identifier_key, identifiers, n_pmts, id_offset, index_by
        )

        def ordered(array, dtype=np.float32):
            if array is None:
                return None
            array = np.asarray(array, dtype=dtype)
            return array if order is None else array[order]

        self.geometry_file = str(geometry_file)
        self.identifier_key = identifier_key
        self.direction_key = direction_key
        self.reordered = order is not None

        self.position = ordered(position)
        self.direction = ordered(direction)
        self.region = ordered(archive['region'], dtype=np.int64) if 'region' in archive else None
        self.region_names = list(archive['region_names']) if 'region_names' in archive else None

        if 'r' in archive and 'theta' in archive:
            self.r = ordered(archive['r'])
            self.theta = ordered(archive['theta'])
            self.cos_theta = ordered(archive['cos_theta']) if 'cos_theta' in archive else np.cos(self.theta)
            self.sin_theta = ordered(archive['sin_theta']) if 'sin_theta' in archive else np.sin(self.theta)
        else:
            # Cylindrical about the detector axis, the convention the stored columns use.
            self.r = np.hypot(self.position[:, 0], self.position[:, 1]).astype(np.float32)
            self.theta = np.arctan2(self.position[:, 1], self.position[:, 0]).astype(np.float32)
            self.cos_theta = np.cos(self.theta).astype(np.float32)
            self.sin_theta = np.sin(self.theta).astype(np.float32)

        log.info(
            f"Geometry {geometry_file}: {n_pmts} PMTs, "
            f"identifier column {identifier_key or 'absent'}, "
            f"direction column {direction_key or 'absent'}, "
            f"{'reordered by identifier' if self.reordered else 'rows used as identifiers'}"
        )

    def _resolve_order(self, geometry_file, identifier_key, identifiers, n_pmts,
                       id_offset, index_by):
        """Row permutation putting the PMTs in identifier order, or None to keep the rows."""
        if index_by == 'row':
            return None, 0

        if identifier_key is None:
            if index_by == 'identifier':
                raise KeyError(
                    f"index_by='identifier' but {geometry_file} has no identifier column "
                    f"(looked for {list(_IDENTIFIER_KEYS)})"
                )
            log.warning(
                f"{geometry_file} has no identifier column; rows are used as zero-based "
                f"PMT identifiers. That is correct only if the file is already ordered."
            )
            return None, 0

        identifiers = identifiers.astype(np.int64)
        if identifiers.shape[0] != n_pmts:
            raise ValueError(
                f"{geometry_file}: '{identifier_key}' has {identifiers.shape[0]} entries "
                f"for {n_pmts} positions"
            )

        offset = int(identifiers.min()) if id_offset == 'auto' else int(id_offset)
        zero_based = identifiers - offset
        if not np.array_equal(np.sort(zero_based), np.arange(n_pmts, dtype=np.int64)):
            raise ValueError(
                f"{geometry_file}: '{identifier_key}' minus the offset {offset} is not a "
                f"permutation of 0..{n_pmts - 1}, so the identifiers cannot index the "
                f"rows. Pass an explicit id_offset, or index_by='row' to keep the "
                f"historical behaviour."
            )

        if np.array_equal(zero_based, np.arange(n_pmts, dtype=np.int64)):
            # Already ordered: reordering would be a no-op, and saying so is more useful
            # than silently permuting by the identity.
            return None, offset

        order = np.empty(n_pmts, dtype=np.int64)
        order[zero_based] = np.arange(n_pmts, dtype=np.int64)
        log.warning(
            f"{geometry_file}: '{identifier_key}' is not in ascending order, so the rows "
            f"have been permuted into identifier order. Indexing this file by row — which "
            f"the image and point-cloud datasets in this package still do — assigns "
            f"each hit the coordinates of a different PMT."
        )
        return order, offset

    def __len__(self):
        return self.position.shape[0]

    def positions_of(self, pmt_ids):
        """Cartesian coordinates of the given zero-based PMT identifiers."""
        return self.position[pmt_ids]

    def directions_of(self, pmt_ids):
        """Orientation unit vectors of the given zero-based PMT identifiers."""
        if self.direction is None:
            raise KeyError(
                f"{self.geometry_file} has no orientation column "
                f"(looked for {list(_DIRECTION_KEYS)})"
            )
        return self.direction[pmt_ids]

    def cylindrical_of(self, pmt_ids):
        """``(r, theta, cos_theta, sin_theta)`` of the given zero-based PMT identifiers."""
        return (self.r[pmt_ids], self.theta[pmt_ids],
                self.cos_theta[pmt_ids], self.sin_theta[pmt_ids])
