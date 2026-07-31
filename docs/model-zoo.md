# Model zoo

Every task, model and dataset combination the repository ships, the config that runs it,
and **how far it has actually been verified**.

## How to read the status column

| | Meaning |
|---|---|
| ✅ **verified** | Shipped config, run end to end (train → validate → restore best → evaluate), and its output was read back by `analysis/` |
| 🟡 **ships** | The config resolves and is exercised by CI (targets resolve, config composes), but no recent full run is recorded |
| 🚧 **code only** | The implementation exists, but no config selects it — you would have to write one |
| ⛔ **broken** | A config exists but its `_target_` does not resolve. Tracked in `tests/data/known_broken_targets.txt` |

The distinction between ✅ and 🟡 is deliberate. A config that composes is not a config
that trains.
✅ refers to config that run
🟡 refers to config that compose.

---

## Particle identification

Classifying the primary particle — typically `e⁻` / `μ⁻`, or the 4-class
`e / μ / γ / π⁰` set.

| Family | Model | Engine | Dataset | Config | Status |
|---|---|---|---|---|---|
| Graph | `GraphAttentionNetwork` | `ClassifierEngine` | 20-inch PMT k-NN graphs (k=5) | `caverns/main/gat_classification` | ✅ verified |
| Graph | `BaseGCN` | `ClassifierEngine` | 20-inch PMT k-NN graphs (k=5) | `caverns/main/gcn_classification` | 🟡 ships |
| Graph | `HierarchicalGAT` (intra- + inter-mPMT attention) | `ClassifierEngine` | WCTE mPMT graph pairs | `caverns/main/wcte_mpmt_gat_classification` | 🟡 ships |
| Image | `resnet18` | `ImageClassifierEngine` | IWCD mPMT images | `watchmal/resnet_train`, `watchmal/resnet_test` | 🟡 ships |
| Image | `resnet50` | `ImageClassifierEngine` | WCTE images | `watchmal/wcte_classification` | 🟡 ships |
| Image | `resnet18` | `ImageClassifierEngine` | tutorial dataset | `watchmal/tutorial_classification` | 🟡 ships |
| Image + PyG loader | `GCN` | `ImageClassifierEngine` | `GnnDataset` (graphs built on the fly) | `watchmal/gnn_train` | 🟡 ships |
| Image + PyG loader | `PointNet` | `ImageClassifierEngine` | `PointNetDataset` | `watchmal/pointnet_train` | 🟡 ships |

`engine/classifier.yaml` sets `target_key: labels` and a `label_set` of `[0, 1, 2, 3]`;
the graph engines use `target_key: target`.

!!! note "Two routes to a graph model"
    `gnn_train` and `pointnet_train` run **image-family engines** with a PyTorch Geometric
    *loader* (`data.is_graph: True`), which is upstream WatChMaL's original GNN support.
    The `caverns/main/*` configs run the **graph engines** proper. They are different code
    paths with different training loops — pick one deliberately.

---

## Vertex reconstruction

Regressing the interaction vertex position.

| Family | Model | Engine | Target | Config | Status |
|---|---|---|---|---|---|
| Graph | `GraphAttentionNetwork` | `RegressionEngine` | `target` (per-component `vtx_x/y/z`) | `caverns/main/gat_vertex_regression` | ✅ verified |
| Image | `resnet18` | `ImageRegressionEngine` | `positions` | `watchmal/reg_train`, `watchmal/reg_test` | 🟡 ships |
| Image | `resnet18` | `ImageRegressionEngine` | `positions` | `watchmal/tutorial_regression`, `watchmal/wcte_regression` | 🟡 ships |
| Image | `SwinRegressor` | `ImageRegressionEngine` | `positions` | `watchmal/transformer_regression` | 🟡 ships |

!!! warning "Component vs quantity naming"
    `analysis/regression.py` keys on the **quantity** (`predicted_positions`, shape
    `(N, 3)`), whereas the graph configs name targets **per component** (`vtx_x`, `vtx_y`,
    `vtx_z`). The graph engines write both; set `predictions_name: positions` to match 
    the shipped analysis classes exactly.

---

## Energy reconstruction

🚧 **Code only — no working config ships today.**

`ImageRegressionEngine` regresses whatever `target_key` names, so `target_key: energies`
is supported by the code. But the only shipped config that mentions energies is
`watchmal/engine/regression_DI.yaml`, whose `_target_`
(`watchmal.engine.images.regression.ImageRegressionDIEngine`) **does not exist** — it is
one of three ledgered broken targets, all belonging to an unfinished dual-image (DI)
feature that predates the unified core. (work is ongoing, [PR#97](https://github.com/WatChMaL/WatChMaL/pull/97) and [PR#93](https://github.com/WatChMaL/WatChMaL/pull/93#pullrequestreview-4826434530))

To run energy regression today you would copy `engine/regression.yaml`, set
`target_key: 'energies'` and choose a `target_scale_factor`. That path is untested; treat
it as a starting point, not a recipe.

---

## Multi-ring segmentation

Assigning each occupied voxel to a parent ring.

| Family | Model | Engine | Target | Config | Status |
|---|---|---|---|---|---|
| Sparse 3-D | `SparseUNet3D` + `QueryPerVoxelSoftmaxHead` (built by `build_segmentation_model`) | `MultiRingSegEngine` | `voxel_parent_frac` | `caverns/main/multiring_segmentation_train`, `..._test` | 🟡 ships |

!!! warning "Requires `spconv`, so it does not run on a laptop"
    The sparse convolutions need `spconv`, which is published as one distribution per
    CUDA build (`spconv-cu118`, `spconv-cu121`, …).
    `spconv-cu121` is declared in `requirements-gpu-images.txt` and
    `requirements-full.txt`, and deliberately not in `requirements-ci.txt`, which is why
    the test suite does not cover this family. On CC-IN2P3 exactly one container
    provides it — see [containers](clusters/cc-in2p3-containers.md). For other cluster
    contact your administator.

Multi-ring is also the one family whose evaluation output does **not** follow the
`.npy` contract the other two share: it delegates to an optional `diagnostic_multiring`
submodule, [diagnostic multiring](https://github.com/Erwanlbv/caverns-diagnosis-public-) so `analysis/` cannot read its results directly. Please refer to the dedicated documentation on this external repository.

---

## Multi-ring regression

🚧 **Code only — no config selects it.**

`watchmal/model/multiring/heads/query_transformer.py` defines
`VertexEnergyRegressionHead` alongside the segmentation head, but **no shipped config
references it**, and `build_segmentation_model` is what the multi-ring model config calls.
Wiring it up means adding a model config that selects the head, and an engine path that
consumes its output.

---

## Known-broken configs

Three `_target_`s in the `watchmal` tree do not resolve. All three are dual-image (DI)
variants of an unfinished feature, all predate the unified core, and all are pinned in
`tests/data/known_broken_targets.txt` as a ratchet — CI fails if a *new* broken target
appears, and equally if one of these is fixed without removing its ledger line.

| Config | Broken `_target_` |
|---|---|
| `data/dataset/hk_dit2tvit.yaml` | `watchmal.dataset.dit2tvit.dit2tvit_dataset.DoubleImageDataset` — the class is `DualImageDataset` |
| `engine/regression_DI.yaml` | `watchmal.engine.images.regression.ImageRegressionDIEngine` — no DI variant exists |
| `model/transformer_dit2t.yaml` | `watchmal.model.T2TViTransformer2I.DualT2T_ViTRegressor` — the module is `T2TViTransformerDI.py` |

---

## What is not here yet

Benchmark numbers and downloadable checkpoints. This table records *what runs*, not the performances
to expect. Work on-going, stay tuned.