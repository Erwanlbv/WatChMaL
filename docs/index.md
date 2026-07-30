# WatChMaL

Machine learning for Water Cherenkov detectors — training, evaluating and comparing
models across three very different data representations, behind one entrypoint and one
configuration system.

WatChMaL is a **framework, not a library**. You do not `import watchmal`; you write a YAML
config and run `main.py`. Everything below follows from that.

## The three families

A Water Cherenkov event is a set of PMT hits — charge and time per tube. There are three
established ways to hand that to a network, and WatChMaL supports all three with a shared
engine hierarchy, shared distributed-training layer and shared output format:

| Family | Event becomes | Typical models | Extra dependency |
|---|---|---|---|
| **Image** | a 2-D array, PMTs laid out on a grid (mPMT modules become channels) | ResNet, Vision Transformer, Swin | none — this is the base install |
| **Graph** | nodes = hit PMTs, edges = k nearest neighbours | GCN, GAT, hierarchical mPMT GAT, PointNet | `torch_geometric` (+ compiled extensions) |
| **Sparse 3-D** | occupied voxels in the detector volume | sparse UNet3D with query-decoder heads | `spconv` |

What is **shared**: the entrypoint, the `BaseEngine` hierarchy, the distributed layer, the
seed policy, the run-output format, and the analysis tooling that reads it.

What is deliberately **not** shared: the training loops and the dataset pipelines. Those
differ because the data genuinely differs — collapsing them would freeze the loop shape
for every future family. "One core" means one class hierarchy and one API surface, never
one training loop.

## Where to go

<div class="grid cards" markdown>

- **[Quickstart](getting-started/quickstart.md)** — install, build a small dataset, train
  and evaluate a model end to end on a laptop CPU.

- **[Model zoo](model-zoo.md)** — every task × model × dataset combination, the config
  that runs it, and how far it has actually been verified.

- **[Install](getting-started/install.md)** — the base install and the four optional
  dependency groups, and why they are optional.

- **[Clusters](clusters/index.md)** — containers, reference datasets and launch scripts
  for CC-IN2P3.

</div>

## What a run looks like

```bash
# train a graph attention network for particle ID
python main.py \
  --config-path tutorial/config/caverns/main \
  --config-name gat_classification
```

A top-level config names a model, a dataset, an engine and a list of **tasks** — typically
`train`, then `restore_best_state`, then `evaluate`. Hydra composes those from config
groups, and any value can be overridden on the command line:

```bash
python main.py --config-path tutorial/config/caverns/main \
  --config-name gat_classification \
  tasks.train.epochs=40 tasks.train.optimizers.lr=1e-4
```

!!! note "Two config trees"
    The repository currently ships **two** config trees: `tutorial/config/watchmal` and
    `tutorial/config/caverns`. `main.py` defaults to the `watchmal` tree, so **every
    caverns example needs an explicit `--config-path`**. Merging the two is planned but
    not done; until then, treat the tree a config lives in as part of its identity.

## Status of this documentation

These pages describe the unified core. They are new and incomplete: install, quickstart,
workspace, the model zoo and the cluster pages are written; per-task guides, the config
group reference and the extension guides are being added.

Where something does not work, this documentation says so rather than omitting it — see
the status column in the [model zoo](model-zoo.md).
