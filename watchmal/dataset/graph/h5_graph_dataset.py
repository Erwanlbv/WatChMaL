"""
Graph events assembled from an HDF5 file at load time, with no stored edges.

The alternative shipped in this package, :mod:`watchmal.dataset.graph.pyg_in_memory_20inch_pmt`,
reads a PyG ``InMemoryDataset`` whose edges were computed offline. That format fixes the
graph at the moment it is written: changing the number of neighbours, or the coordinates
the neighbour search runs on, requires rebuilding the whole file. It is also large —
``edge_index`` accounts for 80 % of the bytes of a stored dataset (40.4 MB of 50.5 MB,
measured on a 200-event subset at k=5) — and the whole file is loaded into memory.

This class keeps the HDF5 file as the stored format, the same one the image datasets
read, and builds one :class:`torch_geometric.data.Data` per event when the event is
requested. The object carries the node features in ``x``, the hit coordinates in ``pos``
and the label in ``y``, and **no** ``edge_index``. The graph is built downstream, inside
the model's forward pass, from ``pos`` (see :mod:`watchmal.model.knn_edges`), on the
device the batch already occupies. The number of neighbours therefore becomes a model
hyper-parameter that can be changed between runs without touching the data.

Hit coordinates are not stored per hit in the HDF5 file; they are looked up in a geometry
file by PMT identifier. The lookup is a scatter table rather than a direct row index,
because the geometry files are not necessarily ordered by identifier: in
``hyperk_20inch_pmts.npz`` the ``tube_id`` column runs 19746, 19745, 19744, … so
``position[i]`` is not the position of PMT ``i``, and indexing directly would displace
every hit without raising anything.
"""

import numpy as np
import torch
import torch_geometric.data as PyGData

# WatChMaL imports
from watchmal.dataset.common.h5_dataset import H5Dataset
from watchmal.utils.logging_utils_caverns import setup_logging

log = setup_logging(__name__)


# Node feature columns that can be requested by name. Charge and time are per hit and
# come from the HDF5 file; the coordinates are per PMT and come from the geometry file.
# The coordinates are available as features as well as in `pos` because a model that
# builds its own edges still needs them as inputs for vertex and direction regression,
# where the graph topology alone cannot express a location.
_FEATURE_SOURCES = ('charge', 'time', 'x', 'y', 'z')


class H5GraphDataset(H5Dataset):
    """One PyG ``Data`` per event, built from an HDF5 file, carrying no edges.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file, in the flat WatChMaL layout: ``hit_pmt``, ``hit_time``,
        ``hit_charge``, ``event_hits_index``, and one array per target.
    geometry_file : str
        Path to a geometry ``.npz`` holding a ``position`` array of shape ``(n_pmts, 3)``.
        A ``tube_id`` array, when present, is used to build the identifier-to-row table.
    feature_keys : sequence of str
        Node feature columns, in order, from ``charge``, ``time``, ``x``, ``y``, ``z``.
        The width of ``data.x`` is ``len(feature_keys)`` and must equal the model's
        ``in_channels``.
    target_key : str
        Name of the HDF5 array read into ``data.y``. Values are passed through unmapped;
        mapping particle codes to class indices is the job of the ``MapLabels`` transform,
        as it is for the stored datasets.
    geometry_tube_id_offset : int
        Value added to a PMT identifier from the HDF5 file to obtain the ``tube_id`` used
        by the geometry file. 1 when the geometry is 1-based and the HDF5 file is 0-based,
        which is the case for the Hyper-K and WCTE files; 0 when the two agree. Ignored
        when the geometry file carries no ``tube_id``.
    transforms : callable, optional
        Applied to each ``Data`` before it is returned, as for the stored datasets. The
        shipped chain ends in ``ConvertAndToDict``, which produces the
        ``{'data', 'target', 'indice'}`` mapping the graph engines read.
    use_memmap : bool
        Memory-map the hit arrays rather than reading them whole.
    """

    def __init__(
        self,
        h5_path,
        geometry_file,
        feature_keys=('charge', 'time'),
        target_key='labels',
        geometry_tube_id_offset=1,
        transforms=None,
        use_memmap=True,
    ):
        super().__init__(h5_path, use_memmap)

        feature_keys = tuple(feature_keys)
        unknown = [k for k in feature_keys if k not in _FEATURE_SOURCES]
        if unknown:
            raise ValueError(
                f"Unknown node feature(s) {unknown}. Choose from {list(_FEATURE_SOURCES)}."
            )
        self.feature_keys = feature_keys
        self.transforms = transforms
        self.geometry_tube_id_offset = int(geometry_tube_id_offset)

        geometry = np.load(geometry_file)
        self.geometry_positions = np.asarray(geometry['position'], dtype=np.float32)
        self.pmt_to_geometry_row = self._build_lookup(geometry)

        # set_target defers until initialize(); the parent reloads it from the h5 file.
        self.set_target(target_key)

        log.info(
            f"H5GraphDataset: {len(self)} events, node features {self.feature_keys}, "
            f"edges built downstream from data.pos"
        )

    def _build_lookup(self, geometry):
        """Table mapping a PMT identifier to its row in the geometry arrays.

        Returns ``None`` when the geometry file carries no ``tube_id``, in which case the
        identifier is used directly as the row index.
        """
        if 'tube_id' not in geometry:
            log.warning(
                f"Geometry file has no 'tube_id' array; PMT identifiers will be used "
                f"directly as row indices into the {self.geometry_positions.shape[0]} "
                f"geometry rows. This is correct only if the rows are already ordered by "
                f"identifier."
            )
            return None

        tube_id = np.asarray(geometry['tube_id']).astype(np.int64)
        lookup = np.full(int(tube_id.max()) + 1, -1, dtype=np.int64)
        lookup[tube_id] = np.arange(tube_id.shape[0], dtype=np.int64)
        return lookup

    def _positions(self, pmts):
        if self.pmt_to_geometry_row is None:
            return self.geometry_positions[pmts]

        identifiers = pmts.astype(np.int64) + self.geometry_tube_id_offset
        rows = self.pmt_to_geometry_row[identifiers]
        if np.any(rows < 0):
            missing = np.unique(identifiers[rows < 0])[:5]
            raise KeyError(
                f"PMT identifier(s) {missing.tolist()} are absent from the geometry file. "
                f"geometry_tube_id_offset is {self.geometry_tube_id_offset}; the usual "
                f"cause is a 0-based file read against a 1-based geometry, or the reverse."
            )
        return self.geometry_positions[rows]

    def __getitem__(self, item):
        # The parent sets event_hit_pmts / event_hit_charges / event_hit_times for this
        # event, and initialises the file handles on first use.
        super().__getitem__(item)

        positions = self._positions(self.event_hit_pmts)
        columns = {
            'charge': np.asarray(self.event_hit_charges, dtype=np.float32),
            'time': np.asarray(self.event_hit_times, dtype=np.float32),
            'x': positions[:, 0],
            'y': positions[:, 1],
            'z': positions[:, 2],
        }

        x = torch.from_numpy(
            np.stack([columns[key] for key in self.feature_keys], axis=1)
        )
        pos = torch.from_numpy(np.ascontiguousarray(positions))
        y = torch.as_tensor(np.atleast_1d(self.targets[self.target_key][item]))

        data = PyGData.Data(x=x, pos=pos, y=y)
        # A plain int, not the sampler's numpy scalar: ConvertAndToDict copies this into
        # the batch dict as 'indice', and PyG's collate rejects a bare numpy.int64 with
        # "DataLoader found invalid type". The stored datasets are indexed by PyG itself
        # and receive a python int, so this only bites the load-time path.
        data.idx = int(item)

        return data if self.transforms is None else self.transforms(data)
