# Quickstart

This page trains and evaluates a graph attention network to separate electrons from muons,
on a published dataset, entirely on a CPU. It takes a few minutes and requires no cluster
allocation.

It assumes the framework has been [installed](install.md) by either route.

## 1. Obtain the dataset

```bash
bash setup/download_data.sh
```

The bundle is 25 MB compressed and unpacks to about 145 MB in `data/quickstart`:

```
data/quickstart/
  graph/e-_200_qtxyz_pid_knn10/    200 simulated electron events
  graph/mu-_200_qtxyz_pid_knn10/   200 simulated muon events
  splits/pid_e_mu_400.npz          train 280 / validation 60 / test 60
```

Each event is a graph: one node per struck photomultiplier tube, with features
(charge, time, *x*, *y*, *z*), and edges to the ten nearest neighbouring tubes. The
simulated tank has a radius of 32.4 m and a half-height of 33 m.

!!! note "Publication of the bundle is in progress"
    The release asset the script fetches is not yet published. Until it is, the script
    accepts `--url` to point at a local or alternative copy, and the datasets can be
    produced from any larger production with `setup/make_smoke_subset.py`.

The split file indexes the two datasets **concatenated in the order the configuration
lists them** — electrons first, then muons — so the indices are not interchangeable with a
configuration that lists them the other way round.

## 2. Create a workspace

The shipped configuration tree under `tutorial/` is a reference and should not be edited.
Copy it:

```bash
bash setup/make_dirs.sh
```

This produces `config/` and `launch/`, both ignored by git, so local dataset paths are
never committed. See [Your own workspace](workspace.md).

## 3. Point a configuration at the data

In `config/caverns/data/dataset/20inch_pmt_knn5_classification.yaml`, set the two dataset
directories and the split file:

```yaml
split_path: data/quickstart/splits/pid_e_mu_400.npz

dataset_parameters:
  graph_folder_path:
    - data/quickstart/graph/e-_200_qtxyz_pid_knn10
    - data/quickstart/graph/mu-_200_qtxyz_pid_knn10
```

## 4. Train

```bash
python main.py \
  --config-path config/caverns/main \
  --config-name gat_classification \
  tasks.train.epochs=2
```

!!! note "`--config-path` is required for this configuration"
    `main.py` declares `tutorial/config/watchmal` as its default configuration path.
    Configurations in the caverns tree, and configurations in a personal workspace, are
    not found without an explicit `--config-path`.

The configuration performs three tasks in sequence: `train`; `restore_best_state`, which
loads the checkpoint with the lowest validation loss; and `evaluate`, which runs the test
split. Results are written under `<dump_path>/<run-id>/outputs/`:

```
softmax.npy            per-event class probabilities
predicted_labels.npy   the argmax of the above
labels.npy             true labels
indices.npy            dataset indices, in evaluation order
log_train.csv          per-step training metrics
log_val.csv            validation metrics
```

`indices.npy` records which event each row corresponds to. It is required because the
evaluation order is not the dataset order under distributed execution.

## 5. Read the results

```python
from analysis.read import WatChMaLOutput

run = WatChMaLOutput('outputs/<run-id>')
run.plot_training_progression()
```

For a classification run:

```python
from analysis.classification import WatChMaLClassification

result = WatChMaLClassification('outputs/<run-id>')
result.softmaxes.shape       # (120, 2) for the test split above
```

A script performs the whole round trip and asserts that it succeeded:

```bash
python setup/check_analysis_pipeline.py outputs/<run-id>
```

!!! warning "The analysis layer needs `uproot` and `tabulate`"
    `analysis/read.py` imports `uproot` and `analysis/regression.py` imports `tabulate`,
    both at module scope. Not every container provides them; on CC-IN2P3 one image does.
    See [containers](../clusters/cc-in2p3-containers.md#analysis-support).

## Interpreting a two-epoch result

Two epochs on 280 training events is a check that the pipeline runs, not a measurement of
performance. The expected outcome is that training loss decreases, that the evaluation
step writes the files listed above, and that the analysis layer reads them. Separation
between electrons and muons at this scale is not meaningful.

## Next

- [Model zoo](../model-zoo.md) — the other tasks, models and datasets, with the extent to
  which each has been verified.
- [Clusters](../clusters/index.md) — containers and reference datasets for full-scale
  training.

## Common failures

- **`Could not find 'gat_classification'`** — `--config-path` is missing or points at the
  wrong tree; see the note in step 4.
- **`FileNotFoundError` on `processed/data.pt`** — a `graph_folder_path` entry is not a
  built PyTorch Geometric dataset directory. Each must contain `processed/data.pt`.
- **`Test loader must have at least one batch`** — the test split is smaller than the
  batch size. Reduce `batch_size`.
- **Loss becomes `nan` immediately** — usually a feature-order mismatch. Node features are
  ordered `(charge, time, x, y, z)` in this bundle, but other productions use
  `(time, charge, x, y, z)`; a configuration that names the wrong index feeds time to the
  network as charge.
- **`undefined symbol` importing `torch_scatter`** — the compiled extensions do not match
  the installed PyTorch. See [Install](install.md).
