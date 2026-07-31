# Quickstart

This page trains and evaluates a graph attention network to separate electrons from muons,
on a published dataset, entirely on a CPU. It takes under a minute, requires no cluster
allocation, and needs no files to be edited. 
A configuration for exactly this dataset is shipped with the repository.

It assumes the framework has been [installed](install.md) by either route.
_(To run it quickly, we recommend skipping the Apptainer container and using Route 2.)_

!!! note "This example needs `requirements-ci.txt`"
    It trains a **graph** model, so PyTorch Geometric is required and `requirements.txt`
    alone is not enough — the run stops at
    `ModuleNotFoundError: No module named 'torch_geometric'`. On a laptop:
    `pip install -r requirements-ci.txt`, which also selects CPU PyTorch wheels.

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

Each event is a graph: one node per hit photomultiplier tube, with features
(charge, time, *x*, *y*, *z*), and edges to the ten nearest neighbouring tubes. The
simulated tank has a radius of 32.4 m and a half-height of 33 m.

The split file indexes the two datasets **concatenated in the order the configuration
lists them** — electrons first, then muons — so the indices are not interchangeable with a
configuration that lists them the other way round.

## 2. Run

From the repository root (remember to activate your Python environment):

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

<!-- ### What `quickstart` differs in

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
`IndexError: index 2 is out of bounds for dimension 1 with size 2`. -->

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

The duplication between `preds` and `softmax` is deliberate: the raw values retain scale
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

Hydra composes the configuration at launch, so it can be changed from the command line
without editing any configuration file.

**Use -c job to inspect the composed configuration before running it.** This resolves every `_target_`,
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

The `model` group substitutes the same way.

**Override nested values**, at any depth, using dotted paths:

```bash
... tasks.train.epochs=40 tasks.train.optimizers.lr=1e-4 tasks.train.data_loaders.train.batch_size=32
```

**Add or remove a key** that the configuration does not already define, with `+` and `~`:

```bash
... +tasks.train.data_loaders.train.persistent_workers=False   # add
... '~tasks.train.early_stopping'                              # remove
```

### All of it at once

The mechanisms combine in one invocation. This substitutes the optimiser group, overrides
two nested values, adds a key and removes another:

```bash
python main.py \
  --config-path tutorial/config/caverns/main \
  --config-name quickstart \
  'hydra.searchpath=[file://tutorial/config/caverns]' \
  optimizers@tasks.train.optimizers=adam_lr1e-3 \
  tasks.train.epochs=4 \
  tasks.train.data_loaders.train.batch_size=32 \
  +tasks.train.data_loaders.train.persistent_workers=False \
  '~tasks.train.early_stopping'
```

| Argument | Mechanism |
|---|---|
| `optimizers@tasks.train.optimizers=adam_lr1e-3` | substitutes a config group into a specific position |
| `tasks.train.epochs=4` | overrides an existing value |
| `tasks.train.data_loaders.train.batch_size=32` | overrides a nested value |
| `+...persistent_workers=False` | adds a key the configuration did not define |
| `'~tasks.train.early_stopping'` | removes a key |

_Every run records the configuration it was actually given, so the overrides can be
confirmed after the fact:_

```bash
grep -E "epochs:|batch_size:|persistent_workers:|lr:" outputs/<date>/<time>/.hydra/config.yaml
grep -c early_stopping outputs/<date>/<time>/.hydra/config.yaml      # 0
```

!!! note "Quote the `~`"
    A bare `~tasks.train.early_stopping` is expanded by the shell as a home directory
    before Hydra sees it, and the command fails with `no such user or named directory`.
    Single quotes prevent that, as they do for list values.

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


!!! warning "Using wandb sweep [sweep](https://docs.wandb.ai/models/sweeps)"
    WatChMaL also support the sweep package from Weights and Biases for more
    complex multiruns. The documentation is being built but feel free to contact
    one of the administators for more informations.



## Next

- [Make your own workspace](workspace.md) - how to setup your task of interest using the watchmal framework.
- [Model zoo](../model-zoo.md) — the other tasks, models and datasets, with the extent to
  which each has been verified.
- [Clusters](../clusters/index.md) — containers and reference datasets for full-scale
  training.


<!-- ## Common failures

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
- **`no such user or named directory`** — an unquoted `~key` was expanded by the shell as
  a home directory. Quote it: `'~key'`.
- **`TypeError: train() got an unexpected keyword argument`** — a key was added under
  `tasks.train` that the engine method does not accept; see the warning in section 5.
- **Loss becomes `nan` immediately** — usually a feature-order mismatch. Node features are
  ordered `(charge, time, x, y, z)` in this bundle, but other productions use
  `(time, charge, x, y, z)`; a configuration naming the wrong index feeds time to the
  network as charge.
- **`undefined symbol` importing `torch_scatter`** — the compiled extensions do not match
  the installed PyTorch. See [Install](install.md). -->
