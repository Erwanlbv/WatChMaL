# Quickstart

This page trains and evaluates a graph attention network to separate electrons from muons,
on a published dataset, entirely on a CPU. It takes a few minutes, requires no cluster
allocation, and **edits no files**: everything the run needs is supplied on the command
line.

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

The split file indexes the two datasets **concatenated in the order the configuration
lists them** — electrons first, then muons — so the indices are not interchangeable with a
configuration that lists them the other way round.

!!! note "Publication of the bundle is in progress"
    The release asset the script fetches is not yet published. Until it is, the script
    accepts `--url` to point at an alternative copy, and equivalent datasets can be
    produced from any larger production with `setup/make_smoke_subset.py`.

## 2. Run

```bash
python main.py \
  --config-path tutorial/config/caverns/main \
  --config-name gat_classification \
  'hydra.searchpath=[file://tutorial/config/caverns]' \
  data.dataset.split_path=data/quickstart/splits/pid_e_mu_400.npz \
  'data.dataset.dataset_parameters.graph_folder_path=[data/quickstart/graph/e-_200_qtxyz_pid_knn10,data/quickstart/graph/mu-_200_qtxyz_pid_knn10]' \
  tasks.train.epochs=2
```

Four things are happening in that command, and each is a Hydra mechanism worth
recognising:

| Argument | Mechanism |
|---|---|
| `--config-name gat_classification` | selects the top-level configuration |
| `--config-path .../caverns/main` | where that file lives |
| `hydra.searchpath=[...]` | where the **config groups** it composes live |
| `key=value` | overrides a value in the composed configuration |

!!! warning "`hydra.searchpath` is required for the caverns tree"
    In this tree the entry configurations sit in `main/` while the config groups they
    compose (`model/`, `engine/`, `sampler/`, …) sit one level above, in
    `tutorial/config/caverns/`. `--config-path` only establishes where the entry file is;
    without `hydra.searchpath` pointing at the parent, composition fails with
    `Could not find 'sampler/subset_sequential'`. The path is resolved relative to the
    working directory, so run from the repository root.

The configuration performs three tasks in sequence: `train`; `restore_best_state`, which
loads the checkpoint with the lowest validation loss; and `evaluate`, which runs the test
split.

## 3. What the run writes

Under `<dump_path>/<run-id>/outputs/`:

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

## 4. Read the results

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

## 5. Varying the run without editing anything

Everything below is a command-line change. No file in the repository is modified, and no
copy of a configuration is made.

**Inspect the composed configuration before running it.** This resolves every `_target_`,
so it also verifies that every class the run would instantiate can be imported:

```bash
python main.py --config-path tutorial/config/caverns/main --config-name gat_classification \
  'hydra.searchpath=[file://tutorial/config/caverns]' -c job
```

**Substitute a whole config group.** A group is a directory of alternatives; naming one
replaces the entire subtree. Here the graph attention network is exchanged for a graph
convolutional network:

```bash
... model=gcn_classifier
```

**Override nested values**, at any depth, using dotted paths:

```bash
... tasks.train.epochs=40 tasks.train.optimizers.lr=1e-4 data.dataset.dataset_parameters.batch_size=32
```

**Add or remove a key** that the configuration does not already define, with `+` and `~`:

```bash
... +tasks.train.num_val_batches=8      # add
... ~tasks.train.early_stopping         # remove
```

**Sweep.** `--multirun` executes the cartesian product of comma-separated values, one run
per combination, each in its own output directory:

```bash
python main.py --config-path tutorial/config/caverns/main --config-name gat_classification \
  'hydra.searchpath=[file://tutorial/config/caverns]' \
  --multirun \
  model=vanilla_gat_classifier,gcn_classifier \
  tasks.train.optimizers.lr=1e-3,1e-4
```

That is four runs. A hyper-parameter scan therefore requires no new configuration files
and no shell loop.

!!! note "`--multirun` and `-c job` are mutually exclusive"
    Hydra permits only one of `--run`, `--multirun`, `--cfg` and `--info` per invocation.
    A sweep cannot be previewed with `-c job`; inspect a single combination first, then
    drop `-c job` and add `--multirun`.

Full command-line documentation is available from Hydra itself:

```bash
python main.py --hydra-help
```

## 6. When to make a workspace instead

Command-line overrides suit exploration and anything scripted. Once a set of values is
settled — a dataset path used every day, an optimiser configuration worth keeping — it
belongs in a file rather than in shell history.

```bash
bash setup/make_dirs.sh
```

produces `config/` and `launch/`, copies of the shipped trees that are ignored by git, so
local paths are never committed. Configurations there are used by pointing
`--config-path` and `hydra.searchpath` at them instead. See
[Your own workspace](workspace.md).

## Next

- [Model zoo](../model-zoo.md) — the other tasks, models and datasets, with the extent to
  which each has been verified.
- [Clusters](../clusters/index.md) — containers and reference datasets for full-scale
  training.

## Interpreting a two-epoch result

Two epochs on 280 training events is a check that the pipeline runs, not a measurement of
performance. The expected outcome is that the training loss decreases, that the evaluation
step writes the files listed above, and that the analysis layer reads them. Separation
between electrons and muons at this scale is not meaningful.

## Common failures

- **`Could not find 'sampler/subset_sequential'`** — `hydra.searchpath` is missing; see the
  warning in step 2.
- **`Could not find 'gat_classification'`** — `--config-path` is missing or names the wrong
  tree. `main.py` defaults to `tutorial/config/watchmal`.
- **`FileNotFoundError` on `processed/data.pt`** — a `graph_folder_path` entry is not a
  built PyTorch Geometric dataset directory. Each must contain `processed/data.pt`.
- **`Test loader must have at least one batch`** — the test split is smaller than the
  batch size. Reduce `batch_size`.
- **A list override has no effect** — the shell split it on the commas. Quote the whole
  argument: `'key=[a,b]'`.
- **Loss becomes `nan` immediately** — usually a feature-order mismatch. Node features are
  ordered `(charge, time, x, y, z)` in this bundle, but other productions use
  `(time, charge, x, y, z)`; a configuration naming the wrong index feeds time to the
  network as charge.
- **`undefined symbol` importing `torch_scatter`** — the compiled extensions do not match
  the installed PyTorch. See [Install](install.md).
