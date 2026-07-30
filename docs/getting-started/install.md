# Install

WatChMaL is run from a checkout, not installed as a package — there is no
`pip install watchmal` and no `pip install -e .` today.

```bash
git clone https://github.com/WatChMaL/WatChMaL.git
cd WatChMaL
pip install -r requirements.txt
```

That base install is enough to run the **image family** end to end.

## The dependency boundary

The optional groups exist because most users need one family, and the heavy dependencies
belong to the other two. This is not a convention — CI has a job that installs
`requirements.txt` **only** and fails if anything on the image path has grown an import of
PyTorch Geometric or `wandb`.

| File | Packages | Needed for |
|---|---|---|
| `requirements.txt` | `torch`, `hydra-core>=1.1`, `omegaconf`, `h5py`, `numpy`, `uproot`, `matplotlib`, `scikit-learn`, `tabulate` | everything; runs the image family on its own |
| `requirements-graph.txt` | `torch_geometric` | graph models |
| `requirements-graph-extensions.txt` | `torch_scatter`, `torch_cluster` | graph models that build k-NN edges or scatter-reduce |
| `requirements-multiring.txt` | `scipy` (+ `spconv`, see below) | multi-ring sparse-3D segmentation |
| `requirements-tracking.txt` | `wandb` | Weights & Biases logging (CSV logging needs nothing) |
| `requirements-transformer.txt` | `timm` | ViT / Swin / T2T models |

So a graph user runs:

```bash
pip install -r requirements.txt -r requirements-graph.txt -r requirements-graph-extensions.txt
```

!!! warning "The graph extensions are compiled, and version-fragile"
    `torch_scatter` and `torch_cluster` are C++/CUDA extensions built against a specific
    torch and CUDA version. Installing them from source against a mismatched torch is the
    most common way a graph install fails. Prefer the prebuilt wheels matching your torch
    exactly, or use a [container](../clusters/cc-in2p3-containers.md).

!!! warning "`spconv` is not in any requirements file, on purpose"
    It ships one distribution per CUDA build — `spconv-cu118`, `spconv-cu120`, … — so
    pinning one would break every other machine. Install the one matching your CUDA:
    `pip install spconv-cu120`. On CC-IN2P3, exactly one container provides it.

## Checking the install

```bash
python -c "import torch, hydra, h5py; print(torch.__version__)"

# graph extras
python -c "import torch_geometric, torch_scatter, torch_cluster; print(torch_geometric.__version__)"

# multi-ring
python -c "import spconv; print(spconv.__version__)"
```

To check the framework itself composes, ask Hydra to print a config and exit — this
resolves every `_target_` without training anything:

```bash
python main.py --config-name resnet_train -c job
```

## Running the test suite

The repository ships a tiered test suite. Tiers 0 and 1 need almost nothing:

```bash
pip install pytest ruff pyyaml
pytest tests/tier0 tests/tier1 -q
```

Tier 0 is static (compiles every module, parses every YAML, checks every `_target_`
resolves) and takes seconds. Tier 1 checks the import and configuration contract. Tier 2
adds CPU unit tests on the merge seams, including distributed collectives over `gloo`.

## If something goes wrong

- **`ModuleNotFoundError: torch_geometric`** running an image config — that should not
  happen; it means an image-path module grew an eager PyG import. Please open an issue,
  the `core` CI job exists to catch exactly this.
- **`undefined symbol` from `torch_scatter`** — the compiled extension does not match your
  torch build. Reinstall both together, or use a container.
- **Hydra `Could not find 'X'`** — you are pointing at the wrong config tree. `main.py`
  defaults to `tutorial/config/watchmal`; caverns configs need
  `--config-path tutorial/config/caverns/main`.
- **`MissingMandatoryValue` on `MASTER_PORT`** — multi-GPU runs require it to be set in
  the config; single-GPU and CPU runs do not.
