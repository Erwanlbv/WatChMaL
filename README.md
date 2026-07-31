# WatChMaL — Water Cherenkov Machine Learning

Unified framework for training, evaluating and using machine learning models for Water
Cherenkov detectors.

**📖 Documentation: <https://watchmal.github.io/WatChMaL/>**

WatChMaL is a *framework*, not a library: you do not `import watchmal`, you write a Hydra
config and run `main.py`. It supports three data representations behind one entrypoint and
one engine hierarchy:

| Family | Event becomes | Models | Extra dependency |
|---|---|---|---|
| **Image** | a 2-D array of PMT hits | ResNet, ViT, Swin | none |
| **Graph** | nodes = hit PMTs, edges = k-NN | GCN, GAT, hierarchical mPMT GAT, PointNet | `torch_geometric` |
| **Sparse 3-D** | occupied voxels | sparse UNet3D + query heads | `spconv` |

## Quick start

```bash
git clone https://github.com/WatChMaL/WatChMaL.git
cd WatChMaL

# one self-contained bundle; see the install page for which
pip install -r requirements-ci.txt     # laptop / CPU development
# pip install -r requirements-gpu-graph.txt    # GPU, graph models
# pip install -r requirements-gpu-images.txt   # GPU, image + multi-ring
# pip install -r requirements-full.txt         # GPU, everything

# make your own (gitignored) config + launch workspace
bash setup/make_dirs.sh

# print a fully composed config without running anything
python main.py --config-name resnet_train -c job

# train
python main.py --config-name resnet_train
```

> `main.py` defaults to the `tutorial/config/watchmal` tree. Configs in
> `tutorial/config/caverns` need an explicit
> `--config-path tutorial/config/caverns/main`.

Full walkthrough: **[Quickstart](https://watchmal.github.io/WatChMaL/docs/getting-started/quickstart/)**.

## Where things are

| | |
|---|---|
| **[Model zoo](https://watchmal.github.io/WatChMaL/docs/model-zoo/)** | every task × model × dataset, the config that runs it, and how far it is verified |
| **[Install](https://watchmal.github.io/WatChMaL/docs/getting-started/install/)** | the base install and the optional dependency groups |
| **[Clusters](https://watchmal.github.io/WatChMaL/docs/clusters/)** | containers and reference data on CC-IN2P3 |

In the repository:

```
main.py                 entrypoint
watchmal/
  engine/               BaseEngine + images/, graph/, multiring/ families
  model/                network architectures
  dataset/              Dataset classes, transforms, samplers
  utils/                tracking, distributed, determinism, build helpers
analysis/               reads a run's outputs: metrics, ROC curves, residuals
tutorial/config/        reference configs (copy to config/ before editing)
tutorial/launch/        reference launch scripts (copy to launch/)
setup/                  workspace, smoke-subset and verification helpers
tests/                  tiered CI suite (tier0 static → tier2 seams)
docs/ + landing/        this documentation site
```

## Tests

```bash
pip install pytest ruff pyyaml
pytest tests/tier0 tests/tier1 -q     # seconds, almost no dependencies
pytest -q                              # everything available in your environment
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
