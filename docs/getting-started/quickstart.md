# Quickstart

This page trains and evaluates a graph attention network to separate electrons from muons,
on a published dataset, entirely on a CPU. It takes under a minute, requires no cluster
allocation, and needs no files to be edited: a configuration for exactly this dataset is
shipped with the repository.

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

## 2. Run

From the repository root:

```bash
python main.py \
  --config-path tutorial/config/caverns/main \
  --config-name quickstart \
  'hydra.searchpath=[file://tutorial/config/caverns]'
```

It completes in well under a minute on a laptop CPU, and needs no arguments beyond
locating the configuration.

| Argument | Mechanism |
|---|---|
| `--config-name quickstart` | selects the top-level configuration |
| `--config-path .../caverns/main` | where that file lives |
| `hydra.searchpath=[...]` | where the **config groups** it composes live |

!!! warning "`hydra.searchpath` is required for the caverns tree"
    In this tree the entry configurations sit in `main/` while the config groups they
    compose (`model/`, `engine/`, `sampler/`, …) sit one level above, in
    `tutorial/config/caverns/`. `--config-path` only establishes where the entry file is;
    without `hydra.searchpath` pointing at the parent, composition fails with
    `Could not find 'sampler/subset_sequential'`. That path is resolved against the
    working directory, so run from the repository root.

The configuration performs three tasks in sequence: `train`; `restore_best_state`, which
loads the checkpoint with the lowest validation loss; and `evaluate`, which runs the test
split.

### What `quickstart` differs in

It is `gat_classification` pointed at the published bundle. Two details are worth reading
before adapting it to your own data, because both are easy to get wrong and neither fails
in an obvious way.

**Dataset paths are written against `${hydra:runtime.cwd}`.** Hydra changes the working
directory for the job, to the run directory it creates under `outputs/`, *before* the
dataset is constructed. A plain relative path would therefore resolve inside `outputs/`
and fail with `FileNotFoundError`. `${hydra:runtime.cwd}` interpolates the directory
`main.py` was launched from, which is why the command must be run from the repository
root. Absolute paths work equally well; `hydra.job.chdir=false` also makes relative paths
resolve, but it moves where results are written and the analysis tooling then cannot find
them.

**The node-feature schema is not the shipped one.** `20inch_pmt_classification` describes
graphs with **two** node features, time and charge, with the geometry carried on the
edges. The published bundle carries **five** — charge, time, *x*, *y*, *z*. Three values
follow from that difference, and they live in
`data/transforms/quickstart_e_mu_pid.yaml` and the entry config: `charge_index` (charge
is first here, not second), `feat_norm` (bounds for five features rather than two), and
the model's `in_channels`. Feeding this bundle through the two-feature transforms raises
`IndexError: index 2 is out of bounds for dimension 1 with size 2`.

## 3. What the run writes

Hydra creates one directory per run, `outputs/<date>/<time>/`, containing its own record
of the job and the framework's results:

```
outputs/2026-07-31/13-37-06/
  main.log                     the run's log
  .hydra/config.yaml           the fully composed configuration, as executed
  outputs/
    softmax.npy                per-event class probabilities, (60, 2)
    preds.npy                  raw model output before the softmax
    targets.npy                true labels, (60,)
    indices.npy                dataset indices, in evaluation order
    log_train_0.csv            per-step training metrics, one file per rank
    log_val.csv                validation metrics
    ClassifierEngine_GraphAttentionNetwork_BEST.pth   best-validation checkpoint
```

`indices.npy` records which event each row corresponds to. It is required because the
evaluation order is not the dataset order under distributed execution.

Both `preds` and `softmax` are written on purpose: the raw values retain scale
information that the softmax discards, which matters when debugging a model, while
`analysis/` keys on `softmax`.

## 4. Read the results

The analysis classes take the **run** directory — the one holding `.hydra/` — not the
inner `outputs/`:

```python
from analysis.read import WatChMaLOutput
from analysis.classification import WatChMaLClassification

run = WatChMaLOutput('outputs/2026-07-31/13-37-06')
run.plot_training_progression()

result = WatChMaLClassification('outputs/2026-07-31/13-37-06', 'quickstart')
result.softmaxes.shape       # (60, 2) — the test split
```

`WatChMaLClassification` takes a second positional argument, a label used in plot
legends.

A script performs the whole round trip and asserts that it succeeded:

```bash
python setup/check_analysis_pipeline.py outputs/2026-07-31/13-37-06 --kind classification
```

```
PASS  softmax read by WatChMaLClassification  (60, 2), rows sum to 1: True
PASS  plot_training_progression               ok
PASS  plot_rocs (consumes softmax)            ok
```

Plots are written to `plots/<run-name>/`.

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
python main.py --config-path tutorial/config/caverns/main --config-name quickstart \
  'hydra.searchpath=[file://tutorial/config/caverns]' -c job
```

**Substitute a whole config group.** A group is a directory of alternatives; naming one
replaces the entire subtree — here the optimiser:

```bash
... optimizers@tasks.train.optimizers=adam_lr1e-3
```

The `model` group substitutes the same way, but a model must declare the right number of
input features for the dataset: `model=gcn_classifier` composes cleanly and then fails at
runtime, because that configuration declares two where this bundle has five.

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
python main.py --config-path tutorial/config/caverns/main --config-name quickstart \
  'hydra.searchpath=[file://tutorial/config/caverns]' \
  --multirun \
  tasks.train.optimizers.lr=1e-3,1e-4 \
  tasks.train.epochs=2,4
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
- **`FileNotFoundError` on `processed/data.pt`** — either a `graph_folder_path` entry is
  not a built PyTorch Geometric dataset directory (each must contain
  `processed/data.pt`), or the path is relative and Hydra has changed directory; use
  absolute paths.
- **`IndexError: index 2 is out of bounds for dimension 1 with size 2`** — the
  normalisation bounds describe fewer features than the data has. See the schema note in
  step 2.
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
