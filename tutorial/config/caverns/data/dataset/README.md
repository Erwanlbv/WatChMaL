# Dataset configs: which class for which data, and why

Each yaml in this folder selects a **dataset class** (`dataset_parameters._target_`) and sets its options. This page explains what each class is for.

Contents

- [Available loading policy](#available-loading-policy)
- `PyGInMemory20inchDataset` [— PMT-level graphs, full loading in RAM](#pyginmemory20inchdataset--pmt-level-graphs-full-loading-in-ram)
- `PyGInMemoryMPMTDataset` [— paired PMT + mPMT graphs (WCTE)](#pyginmemorympmtdataset--paired-pmt--mpmt-graphs-wcte)
- `H5GraphDataset` [— flat HDF5 events, graphs assembled at load time](#h5graphdataset--flat-hdf5-events-graphs-assembled-at-load-time)
- `HyperKSparseCNN3D` [— HDF5 hits → sparse 3D voxels (multi-ring)](#hyperksparsecnn3d--hdf5-hits--sparse-3d-voxels-multi-ring)
- [Common config keys](#common-config-keys)



## Available loading policy


| Config                                                                          | Dataset class              | Input on disk                                                       | Feeds                                        | Why this class                                                         |
| ------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------- |
| `20inch_pmt_knn5_classification.yaml`, `20inch_pmt_knn5_vertex_regression.yaml` | `PyGInMemory20inchDataset` | pre-built PyG graphs (`data.pt`) **not recommended for production** | GNNs (GCN, GAT)                              | small graph samples → load once, keep in RAM                           |
| `wcte_mpmt_classification.yaml`                                                 | `PyGInMemoryMPMTDataset`   | pre-built graph **pairs** (`data.pt` + `data_mPMT.pt`)              | hierarchical mPMT models (`HierarchicalGAT`) | model needs two granularities per event                                |
| `hk_h5_graph_classification.yaml`                                               | `H5GraphDataset`           | flat WatChMaL HDF5 (`hit_pmt`, `hit_time`, `hit_charge`) + geometry `.npz` | GNNs that build their own edges (`GraphAttentionNetwork` with `knn_k`) | no stored edges → the neighbour count is a model hyper-parameter and the search runs on the batch's device |
| `multiring_sparse3d.yaml`                                                       | `HyperKSparseCNN3D`        | raw WCSim HDF5 digi-hits                                            | sparse 3D CNN (`SparseUNet3D` + query head)  | events are sparse in a huge 3D volume → voxelize on the fly for spconv |




## `PyGInMemory20inchDataset` — PMT-level graphs, full loading in RAM

`watchmal.dataset.graph.pyg_in_memory_20inch_pmt.PyGInMemory20inchDataset`
(a `torch_geometric.data.InMemoryDataset`).

Loads a fully-processed PyG graph file (`data.pt`) — one graph per event, edges have to be pre-computed at graph-generation time (e.g. KNN k=5 for the shipped examples) — entirely into RAM, then serves slices of it. Transforms (`data/transforms/` group) are applied on a *clone* at access time.

**Why in-memory**:

- The tutorial graph samples are small; loading once beats re-reading files every epoch.
- On multi-GPU (DDP) runs, `main.py` builds the dataset **once** and shares it across
worker processes (`kind: "pyg_in_memory"` triggers this) — otherwise every process would load the full dataset at run time.

**Multiple folders**:  `pyg_data_folder_path` can take a *list* of folders and one dataset is
built per folder, wrapped in `PyGConcatDataset` — handy to mix particle types
(see `20inch_pmt_knn5_classification.yaml`).

## `PyGInMemoryMPMTDataset` — paired PMT + mPMT graphs (WCTE)

`watchmal.dataset.graph.pyg_in_memory_mpmt.PyGInMemoryMPMTDataset`.

**This part is for mPMT detectors only (WCTE / IWCD).**  It serves **two graphs per event** : a PMT-level graph (`data.pt`) and an mPMT-level graph (`data_mPMT.pt`) — returned together as  
`{'pmt_data': ..., 'mpmt_data': ...}`.

**Two internal datasets are required** because PyG's `InMemoryDataset.load()` can only hold one processed
file per instance, so this class wraps **two** `PyGInMemory20inchDataset` instances (one
per file) and asserts they have the same length. Transforms are currently applied to the
PMT-level graphs only (mPMT transforms not yet supported — see
`watchmal/dataset/graph/data_utils.py`).

## `H5GraphDataset` — flat HDF5 events, graphs assembled at load time

`watchmal.dataset.graph.h5_graph_dataset.H5GraphDataset` (a subclass of
`watchmal.dataset.common.h5_dataset.H5Dataset`).

**What it does.** The class reads the flat WatChMaL HDF5 layout — the one the image
datasets read: `hit_pmt`, `hit_time`, `hit_charge` and `event_hits_index`, plus one array
per target — and assembles one `torch_geometric.data.Data` per event when that event is
requested. The object carries the node features in `x`, the hit coordinates in `pos`, the
unmapped label in `y` and the event index in `idx`. It carries **no** `edge_index`: the
k-nearest-neighbour graph is built inside the model's forward pass, from `pos`, by
`watchmal.model.knn_edges.build_knn_edge_index`. The transforms of the `data/transforms/`
group are applied to each object as it is built; the shipped chain ends in
`ConvertAndToDict`, which produces the `{'data', 'target', 'indice'}` mapping the graph
engines read.

**Why the edges are not stored.** A stored PyG file fixes the graph when it is written:
changing the number of neighbours, or the coordinates the search runs on, requires
rebuilding the file. Such a file is also dominated by its edges, and is loaded whole into
memory: in `e-_200_qtxyz_energy_knn5`, a stored subset of 200 events at k=5, `edge_index`
occupies 39.4 MB of 49.3 MB (80 %), since each node carries ten int64 edge entries
(80 bytes) against five float32 features (20 bytes). Building the edges in the forward pass makes the number of neighbours a model
hyper-parameter (`model.knn_k`) and runs the neighbour search on the device the batch
already occupies. Assembling one event from the HDF5 file takes about 0.09 ms on a laptop
CPU (Apple M1 Pro) before transforms, and 0.35 to 0.6 ms with the shipped transform chain,
averaged over the first 50 events of a Hyper-K file after the first access.

**Hit coordinates.** An event file records which photomultipliers (PMTs) were hit, not
where they are. Positions and orientations are looked up in the geometry file through
`watchmal.dataset.common.geometry.DetectorGeometry`, which honours the file's identifier
column and returns every array in identifier order. The distinction is not cosmetic: the
`tube_id` column of `hyperk_20inch_pmts.npz` runs 19746, 19745, 19744, …, so indexing that
file by row assigns each hit the coordinates of a different PMT, without raising any error.
Measured over 150 events of a Hyper-K file, the median angle between the charge-weighted
hit centroid, taken from the true vertex, and the true particle direction is 24.9° when the
identifier is honoured and 88.6° — the value expected of a random assignment — when it is
not; `tests/tier2/test_local_smoke_runs.py::test_geometry_lookup_places_hits_correctly`
asserts both regimes. The reader accepts `tube_id` or `tube_no` as the identifier column and
`direction` or `orientation` as the orientation column.

**Configuration** (`hk_h5_graph_classification.yaml`, under `dataset_parameters`):

| Key                  | Meaning                                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------------------- |
| `h5_path`            | the flat WatChMaL HDF5 file                                                                               |
| `geometry_file`      | `.npz` holding `position` of shape `(n_pmts, 3)`; it must describe the detector the HDF5 file was produced for |
| `geometry_index_by`  | `auto` (default) honours the identifier column when present; `identifier` requires it; `row` uses the identifier as a row number, which is what the image and point-cloud datasets do |
| `geometry_id_offset` | value added to a zero-based `hit_pmt` to obtain the geometry's identifier; `auto` reads it from the smallest identifier in the file |
| `feature_keys`       | node feature columns, in order, from `charge`, `time`, `x`, `y`, `z`, `dir_x`, `dir_y`, `dir_z`, `r`, `theta`, `cos_theta`, `sin_theta` |
| `target_key`         | HDF5 array read into `data.y`, unmapped (default `labels`)                                                |

**Three settings must agree**, and no code checks them; a mismatch surfaces late, or not at
all:

1. `len(feature_keys)` equals the model's `in_channels` (5 in
   `model/gat_cls_knn_classifier.yaml`). A mismatch raises a shape error in the first
   forward pass.
2. The `Normalize` transform has one `feat_norm` column and one `apply_log` entry per
   feature (`data/transforms/h5_graph_classification.yaml`). More features than entries
   raise `IndexError` when the transform runs in the data loader; fewer pass silently, and
   only the leading bounds are used.
3. The coordinates reach the network as feature columns, where `Normalize` scales them, so
   the model's `position_dim` stays 0. A non-zero `position_dim` raises nothing: it appends
   `data.pos` in centimetres beside features scaled to [0, 1]; measured on one Hyper-K
   event, the charge and time columns then carry 0.00 % of the input variance, and the
   model does not learn.
   `data.pos` itself is never normalised, since the neighbour search needs distances in
   detector units.

**Labels.** `data.y` holds the file's labels unmapped. The `MapLabels` transform maps them
to class indices, and its `label_set` must list the values the file contains, in the order
of `target_names`: `[13, 11]` for a file storing PDG codes; `[0, 1]` for a file in Felix
Cormier's layout (`felix-cormier/DataTools@felix_dev`, `root_utils/event_dump.py`), whose
writer stores the labels already mapped as `{13: 0, 11: 1, 22: 2}` (line 584) and stores
every other PDG code as −1, the antiparticles −13 and −11 included, since the comparison
takes no absolute value (line 583). A label absent from `label_set`, −1 included, raises
`ValueError` inside a data-loader worker; events carrying one must be left out of the split.

**Why not in-memory.** `kind: "pyg_h5_graph"` does not contain `in_memory`, so `main.py`
does not build the dataset once for all processes. Each engine builds its own instance
through the load-time branch of `watchmal/dataset/graph/data_utils.py::get_dataset`, which
accepts dataset parameters that name no folder of stored tensors. The HDF5 file is opened,
and the hit arrays memory-mapped (`use_memmap`, true by default), on first access
(`H5CommonDataset.initialize`, extended by `H5Dataset.initialize` for `hit_charge`). An
instance not yet indexed can therefore be passed to data-loader worker processes; once
indexed in the parent process it holds an `h5py` handle and can no longer be pickled.

**Status: verified.** End to end on a laptop CPU by
`tests/tier2/test_local_smoke_runs.py::test_h5_graph_run_with_forward_pass_knn`, which skips
when its local data are absent, as on CI runners. On a MIG slice of Fir, a Digital Research
Alliance of Canada cluster (NVIDIA H100 80GB, 3g.40gb profile), a memorisation control — 32
events used as training, validation and test set, classification-token readout, k=4,
`hidden_channels=64`, `num_layers=3`, batch size 8, seed 4242 — reached accuracy 1.0 after
2000 iterations in three jobs on three nodes, with a loss on the last 8-event batch of
1.01e-03, 1.05e-03 and 1.30e-03. Two of the jobs used the same 32 events and the third a
partly different set; only the third recorded its code version, commit `3de15ce` of this
branch.

## `HyperKSparseCNN3D` — HDF5 hits → sparse 3D voxels (multi-ring)

`watchmal.dataset.multiring.sparse_cnn.HyperKSparseCNN3D` (referenced via
`dataset.target_3d`, instantiated by the multi-ring engine rather than by Hydra).

**What it does**: reads raw WCSim multi-hit HDF5 files (searched recursively under
`base_dir` for `file_name_pattern`), de-duplicates hits per tube, voxelizes the hit points
into a sparse 3D grid (`spconv PointToVoxel`, grid controlled by `grid.axis_limit` /
`grid.grid_size`), normalizes features (`feat_norm`, stats cached to disk), and yields
per-voxel features plus `voxel_parent_frac` soft targets for ring segmentation.

**Why sparse voxels instead of graphs**:

- A Hyper-K event lights up a tiny fraction of a huge 3D volume; a dense 3D grid would be
almost all zeros. Sparse convolutions (spconv) only compute on occupied voxels.
- Ring **segmentation** is a per-voxel task — the natural output structure is the voxel
grid itself, not a graph.

**Why not in-memory / not Hydra-instantiated**: files are big and per-event processing is
cheap, so each engine builds its own instance lazily (`cache_in_ram` can cache processed
events). Needs `spconv` → run inside the container image
(see [docs/cclyon_available_containers.md](../../../docs/cclyon_available_containers.md)).

**Handy knobs**: `num_batches: 1` caps the run to one file (smoke tests),
`stats_cache_path` redirects the normalization-stats cache when the data dir is read-only.

## Common config keys

Keys of the graph dataset configs. `kind` is read by `main.py`, `dataset_parameters` by `watchmal/dataset/graph/data_utils.py`, and `split_path`, `target_names` and `signal_key` by the graph engines:


| Key                  | Meaning                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------ |
| `kind`               | a value containing `pyg_in_memory` → `main.py` builds the dataset once and passes it to every (DDP) process; a value containing `in_memory` but not `pyg_in_memory` raises `ValueError`; any other value, e.g. `"pyg_h5_graph"`, or none → each engine builds its own instance |
| `split_path`         | `.npz` with `train_idxs` / `val_idxs` / `test_idxs`, indices into the dataset's event order |
| `fully_processed`    | read by no code at present, like `compute_edges_parameters`; present in the stored-graph configs only      |
| `label_set`          | not read from the dataset config by any code (`wcte_mpmt_classification.yaml` still carries one). The class mapping is applied only by the `MapLabels` transform, whose `label_set` is set in the transform config |
| `target_names`       | one name per model output column — class names for classification, components for regression; its length must equal the model's `out_channels`, because the engine reshapes the evaluation output with it; it also labels the plots |
| `signal_key`         | which class is "signal" in plots                                                           |
| `dataset_parameters` | everything below is passed to the dataset class (`_target_`, paths, file names)            |


To add your own dataset class: implement it under `watchmal/dataset/`, point
`dataset_parameters._target_` at it, and keep the keys above — then mirror one of these
configs in your own `config/data/dataset/`.