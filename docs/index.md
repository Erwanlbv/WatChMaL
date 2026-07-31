# WatChMaL

WatChMaL is a framework for training and evaluating machine-learning models on data from
water Cherenkov detectors. A single entrypoint and one configuration system cover three
data representations — images, graphs and sparse three-dimensional voxel grids — and the
reconstruction tasks built on them: particle identification, vertex and energy
regression, and multi-ring segmentation.

The framework is configured rather than imported. A run is specified by a YAML file that
names a model, a dataset, an engine and a sequence of tasks; `main.py` composes that
specification and executes it. There is no Python API to call.

## Detector data and its representations

A water Cherenkov detector records, for each photomultiplier tube (PMT), the collected
charge and the arrival time of Cherenkov light. A charged particle above threshold emits
light in a cone, which intersects the detector wall in a ring whose sharpness reflects
how much the particle scattered.

That per-PMT record admits three representations, and WatChMaL implements all three
behind a shared engine hierarchy:

| Representation | Construction | Models | Additional dependency |
|---|---|---|---|
| **Image** | PMTs projected onto a two-dimensional grid; multi-PMT (mPMT) modules become channels | ResNet, Vision Transformer, Swin | none |
| **Graph** | hit PMTs as nodes, edges to the *k* nearest neighbours | GCN, GAT, hierarchical mPMT GAT, PointNet | PyTorch Geometric |
| **Sparse 3-D** | occupied voxels of the detector volume | sparse UNet3D with query-decoder heads | `spconv` |

The entrypoint, the engine base class, the distributed-training layer, the random-seed
policy, the output format and the analysis tooling are common to all three. The training
loops and dataset pipelines are not, and are not intended to be: the representations
differ substantially, and a single loop would fix its control flow for every
representation added later.

## Orientation

| To | See |
|---|---|
| install the framework | [Install](getting-started/install.md) |
| train and evaluate a model end to end | [Quickstart](getting-started/quickstart.md) |
| find the configuration for a given task | [Model zoo](model-zoo.md) |
| run on a computing cluster | [Clusters](clusters/index.md) |

## Invoking a run

```bash
python main.py \
  --config-path tutorial/config/caverns/main \
  --config-name gat_classification
```

Configuration values may be overridden on the command line, so a parameter scan requires
no additional files:

```bash
python main.py --config-path tutorial/config/caverns/main \
  --config-name gat_classification \
  tasks.train.epochs=40 tasks.train.optimizers.lr=1e-4
```

!!! note "Two configuration trees"
    Two trees are currently shipped, `tutorial/config/watchmal` and
    `tutorial/config/caverns`. `main.py` resolves configuration against the first by
    default, so configurations in the second require an explicit `--config-path`. The
    trees are to be merged; until then, the tree a configuration belongs to is part of
    its identity.

## Scope of this documentation

The installation, quickstart, model zoo and cluster pages are complete. Guides for
individual tasks, the configuration reference and the extension guides are in
preparation.

Capability that is implemented but not exercised by any shipped configuration is
recorded as such rather than omitted; the [model zoo](model-zoo.md) states, for every
task, whether it has been run end to end, whether it is only verified to compose, or
whether no working configuration exists.
