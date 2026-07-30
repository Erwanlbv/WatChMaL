# Quickstart

Train and evaluate a graph attention network for particle ID, end to end, on a CPU. This
is the path that has actually been run on a laptop — no GPU, no cluster.

After [installing](install.md) the base requirements plus the graph extras:

```bash
pip install -r requirements.txt -r requirements-graph.txt -r requirements-graph-extensions.txt
```

## 1. Create your workspace

Never edit the shipped `tutorial/` tree. Copy it:

```bash
bash setup/make_dirs.sh          # creates config/ and launch/
```

Both are gitignored, so your dataset paths and tweaks stay yours. The script **refuses to
overwrite** an existing workspace — pass `--force` if you really mean it. See
[Your own workspace](workspace.md).

## 2. Get a small dataset

The shipped graph datasets are tens of GB in a single `processed/data.pt`, which an
`InMemoryDataset` loads whole — too big for a laptop. Carve a subset **on the cluster**,
then copy the result:

```bash
# on the cluster, next to the big dataset
python setup/make_smoke_subset.py SRC_DATASET_DIR DST_DATASET_DIR --n 200

# inspect one without copying anything
python setup/make_smoke_subset.py SRC_DATASET_DIR --inspect
```

200 events is about 30 MB. The carve costs a fraction of a second on a 39 GB file, and
needs only `torch` — no PyTorch Geometric on the login node.

!!! note "Feature order differs between datasets"
    The node-feature order is **not** portable: the PID datasets are `qtxyz` (charge
    first) while the vertex-regression one is `tqxyz` (time first). A config's
    `charge_index` therefore does not carry across datasets. Check before reusing one.

## 3. Point a config at it

In your `config/` copy, edit the dataset config's `graph_folder_path` to your subset, and
the split file to a matching index list:

```yaml
# config/caverns/data/dataset/20inch_pmt_knn5_classification.yaml
graph_folder_path: /path/to/your/smoke/subset
```

## 4. Train

```bash
python main.py \
  --config-path config/caverns/main \
  --config-name gat_classification \
  tasks.train.epochs=2
```

The config runs three tasks in order: `train`, then `restore_best_state` (loads the
best-validation checkpoint), then `evaluate`. Outputs land under
`<dump_path>/<run-id>/outputs/`.

```
outputs/
  softmax.npy              per-event class probabilities
  predicted_labels.npy     argmax of the above
  labels.npy               truth
  indices.npy              which events, in dataset order
  log_train.csv            per-step training metrics
  log_val.csv              validation metrics
```

!!! note "Every caverns example needs `--config-path`"
    `main.py` declares `config_path='tutorial/config/watchmal'`, so without an explicit
    `--config-path` Hydra looks in the wrong tree and reports a missing config.

## 5. Read the results back

The run directory is readable by the analysis layer:

```python
from analysis.read import WatChMaLOutput

run = WatChMaLOutput('outputs/<run-id>')
run.plot_training_progression()          # loss/accuracy vs step, from the CSVs
```

For classification specifically:

```python
from analysis.classification import WatChMaLClassification

c = WatChMaLClassification('outputs/<run-id>')
c.softmaxes.shape                        # (n_events, n_classes)
```

There is also a script that runs the whole round trip and asserts it:

```bash
python setup/check_analysis_pipeline.py outputs/<run-id>
```

## Where to go next

- [Model zoo](../model-zoo.md) — every other task, model and dataset, and how far each is
  verified.
- [Your own workspace](workspace.md) — how `config/` and `launch/` relate to the shipped
  tree.
- [Clusters](../clusters/index.md) — containers and reference data for running at scale.

## If something goes wrong

- **`Could not find 'gat_classification'`** — missing or wrong `--config-path`; see the
  note in step 4.
- **`FileNotFoundError` on `processed/data.pt`** — `graph_folder_path` points at a
  directory that is not a built PyG dataset. It must contain `processed/data.pt`.
- **`Test loader must have at least one batch`** — the split's test index list is smaller
  than the batch size. Lower `batch_size`, or carve a bigger subset.
- **Loss is `nan` immediately** — usually the wrong `charge_index` for the dataset, i.e.
  the feature-order trap in step 2; the network is being fed time as charge.
- **`import uproot` fails inside a container** — `analysis/read.py` imports it at module
  level and not every image ships it. See
  [containers](../clusters/cc-in2p3-containers.md#analysis-support).
- **Two runs give different numbers with the same seed** — `num_workers > 0` plus a
  non-deterministic op; set the deterministic mode and compare again.
