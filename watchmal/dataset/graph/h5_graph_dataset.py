"""
Graph events assembled from an HDF5 file at load time, with no stored edges.

The alternative shipped in this package is :mod:`watchmal.dataset.graph.pyg_in_memory_20inch_pmt`,
which reads a PyG ``InMemoryDataset`` whose edges were computed offline. 
On the other hand this H5GraphDataset format fixes the graph (i.e. computes the edges) at the moment 
it is written. The movitivation for this change was mostly convenience 
1. changing the policy of the edges required to rebuild the whole edge file. 
2. the edges files are quite large — ``edge_index`` accounts for 80 % of the bytes of a stored dataset 
(40.4 MB of 50.5 MB, measured on a 200-event subset at k=5) — and the whole file is loaded into memory.

Besides this class keeps the HDF5 file as the stored format, (the flat hdf5 format used by the image datasets)
it also builds one :class:`torch_geometric.data.Data` per event. 
The object carries the node features in ``x``, the label in ``y``, and **no** ``edge_index``. 
The graph is built downstream, inside the model's forward pass, from ``pos`` (see :mod:`watchmal.model.knn_edges`), 
on the device the batch already occupies. 

Hit coordinates are not stored per hit in the HDF5 file but looked up in a geometry file by PMT identifier, 
through :class:`watchmal.dataset.common.geometry.DetectorGeometry`, which resolves the identifier column and
returns every array in identifier order.
"""

import numpy as np
import torch
import torch_geometric.data as PyGData

# WatChMaL imports
from watchmal.dataset.common.h5_dataset import H5Dataset
from watchmal.dataset.common.geometry import DetectorGeometry
from watchmal.utils.logging_utils_caverns import setup_logging

log = setup_logging(__name__)


# Node feature columns that can be requested by name. Charge and time are per hit and
# come from the HDF5 file; everything else is per PMT and comes from the geometry file.
_FEATURE_SOURCES = ('charge', 'time',
                    'x', 'y', 'z',
                    'dir_x', 'dir_y', 'dir_z',
                    'r', 'theta', 'cos_theta', 'sin_theta')


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
    geometry_index_by : {"auto", "identifier", "row"}
        How the geometry file is indexed. ``"auto"`` honours its identifier column when it
        has one. ``"row"`` reproduces the historical behaviour, in which the identifier is
        used as a row number. See
        :class:`watchmal.dataset.common.geometry.DetectorGeometry`.
    geometry_id_offset : int or "auto"
        Value added to a zero-based identifier from the HDF5 file to obtain the identifier
        used by the geometry file. ``"auto"`` reads it from the file.
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
        geometry_index_by='auto',
        geometry_id_offset='auto',
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

        self.geometry = DetectorGeometry(
            geometry_file, id_offset=geometry_id_offset, index_by=geometry_index_by
        )
        needs_direction = any(k.startswith('dir_') for k in feature_keys)
        if needs_direction and self.geometry.direction is None:
            raise KeyError(
                f"Node features {[k for k in feature_keys if k.startswith('dir_')]} were "
                f"requested but {geometry_file} carries no orientation column."
            )

        # set_target defers until initialize(); the parent reloads it from the h5 file.
        self.set_target(target_key)

        log.info(
            f"H5GraphDataset: {len(self)} events, node features {self.feature_keys}, "
            f"edges built downstream from data.pos"
        )

    def __getitem__(self, item):
        # The parent sets event_hit_pmts / event_hit_charges / event_hit_times for this
        # event, and initialises the file handles on first use.
        super().__getitem__(item)

        pmts = self.event_hit_pmts
        positions = self.geometry.position[pmts]
        columns = {
            'charge': np.asarray(self.event_hit_charges, dtype=np.float32),
            'time': np.asarray(self.event_hit_times, dtype=np.float32),
            'x': positions[:, 0],
            'y': positions[:, 1],
            'z': positions[:, 2],
            'r': self.geometry.r[pmts],
            'theta': self.geometry.theta[pmts],
            'cos_theta': self.geometry.cos_theta[pmts],
            'sin_theta': self.geometry.sin_theta[pmts],
        }
        if self.geometry.direction is not None:
            directions = self.geometry.direction[pmts]
            columns.update(dir_x=directions[:, 0], dir_y=directions[:, 1],
                           dir_z=directions[:, 2])

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
