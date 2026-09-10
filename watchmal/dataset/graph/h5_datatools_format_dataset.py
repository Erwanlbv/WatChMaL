# generic imports
import numpy as np

# torch imports
import torch

# WatChMaL imports
from watchmal.dataset.common.h5_dataset import H5Dataset
from watchmal.dataset.common.geometry import DetectorGeometry

# pyg imports
import torch_geometric.data as PyGData

# torch cluster imports
from torch_cluster import knn_graph


class GnnDataset(H5Dataset):
    def __init__(self, h5file, geometry_file, k_neighbors, transforms=None, is_distributed=True, use_memmap=True,
                 geometry_index_by='auto', geometry_id_offset='auto'):
        """
        Args:
            h5file              ... path to h5 dataset file
            geometry_file       ... path to the geometry file
            k_neighbors         ... number of nearst neighbors used to connect the graph
            transforms          ... transforms to apply
            use_memmap          ... use a memmap and load data into memory as needed (default), otherwise load entire dataset at initialisation
        """
        super().__init__(h5file, use_memmap)

        # Ordered by PMT identifier: indexing this file by row assigns each hit
        # the coordinates of a different PMT whenever the rows are not already in
        # identifier order, which is the case for hyperk_20inch_pmts.npz.
        geometry = DetectorGeometry(geometry_file, id_offset=geometry_id_offset,
                                    index_by=geometry_index_by)
        self.geo_positions = geometry.position
        self.geo_orientations = geometry.direction

        self.k_neighbors = k_neighbors

    def __getitem__(self, item):
        super().__getitem__(item)

        hit_positions = self.geo_positions[self.event_hit_pmts, :]
        hit_orientations = self.geo_orientations[self.event_hit_pmts, :]

        n_hits = self.event_hit_pmts.shape[0]

        # define the training feature matrix (x,y,z, e_x, e_y, e_z, charge, time)
        data = np.zeros((8, n_hits))
        data[:3, :n_hits] = hit_positions[:n_hits].T
        data[3:6, :n_hits] = hit_orientations.T
        data[-2, :n_hits] = self.event_hit_charges[:n_hits]
        data[-1, :n_hits] = self.event_hit_times[:n_hits]

        data = data.T
        # scale the training feature to be almost of the scale
        scale = np.array([100., 100., 100., 1., 1., 1., 1., 1000.])
        data /= scale

        x = torch.tensor(data, dtype=torch.float32)
        # label tensor
        y = torch.tensor(self.labels[item], dtype=torch.int64)
        # graph connecvtivety by k_nn algorithm
        knn_graph_edge_index = knn_graph(x[:, 0:3], k=self.k_neighbors)

        return {"data": PyGData.Data(x=x, y=y, edge_index=knn_graph_edge_index), "labels": y}
