# Install

Two routes are available. **Containers are the supported route** and should be preferred
wherever a container runtime exists. A local Python environment is the fallback for
machines where one does not.

The reason is the dependency stack rather than convenience. The graph representation
requires PyTorch Geometric together with its compiled extensions (`torch_scatter`,
`torch_cluster`, `pyg_lib`), each built against an exact PyTorch and CUDA version; the
sparse three-dimensional representation requires `spconv`, which is distributed as a
separate package per CUDA build. A mismatch between any of these and the installed
PyTorch produces a linker error at import time rather than at install time. The
containers pin a combination that is known to work.

!!! note "On CC-IN2P3"
    The images are already present on the shared filesystem, and nothing needs to be
    built or downloaded. Go directly to
    [Clusters → CC-IN2P3 containers](../clusters/cc-in2p3-containers.md), which lists
    the available images, their contents and which one each task requires.

## Route 1 — container

### Obtaining an image

One image covers all three representations. On a machine with
[Apptainer](https://apptainer.org) (formerly Singularity):

```bash
apptainer pull ml_image.sif oras://ghcr.io/erwanlbv/watchmal-ml:2.2.2-cu121
```

The result is a single 10.4 GiB file that can be copied between machines. The tag names
the PyTorch and CUDA versions the image is built around, since those are what the
compiled extensions inside it are matched to.

It provides PyTorch 2.2.2 (CUDA 12.1), PyTorch Geometric 2.5.3 with `torch_scatter` and
`torch_cluster`, `spconv` 2.3.8, `timm`, `wandb`, `h5py`, `scikit-learn`, `scipy` and
`matplotlib` — that is, every representation and every model family in one environment.

!!! note "`oras://`, not `docker://`"
    The image is stored as a native Apptainer `.sif` rather than an OCI image, so it is
    fetched with the `oras://` transport. `docker://` will not resolve it.

!!! warning "Two packages the analysis layer needs are absent"
    `analysis/read.py` imports `uproot` and `analysis/regression.py` imports `tabulate`,
    both at module scope, and neither is in this image. Training and evaluation are
    unaffected — the run writes plain `.npy` and `.csv` — but reading results back inside
    this container requires `pip install --user uproot tabulate` once, which persists in
    your home directory. See
    [containers](../clusters/cc-in2p3-containers.md#analysis-support).

### Running the framework inside it

The repository is not built into the image; it is bound into the container at runtime, so
edits to the code take effect without rebuilding anything.

```bash
git clone https://github.com/WatChMaL/WatChMaL.git
cd WatChMaL

apptainer exec --nv \
  --bind "$PWD":/workspace/ml \
  --bind /path/to/your/data:/workspace/data \
  --pwd /workspace/ml \
  ml_image.sif \
  python main.py --config-name resnet_train -c job
```

`--nv` exposes the host GPU and may be omitted for CPU-only work. `--bind` makes a host
directory visible inside the container; both the checkout and the data must be bound,
because a container sees no host path that has not been.

## Route 2 — local Python environment

```bash
git clone https://github.com/WatChMaL/WatChMaL.git
cd WatChMaL
```

Then install **one** bundle. Each is self-contained: it chains to `requirements.txt` and
sets whichever package index it needs, so no bundle is combined with another.

| Bundle | Installs | For |
|---|---|---|
| `requirements.txt` | torch, hydra, omegaconf, h5py, numpy, uproot, matplotlib, scikit-learn, tabulate | the image representation, and everything `analysis/` needs |
| `requirements-ci.txt` | the above, CPU torch, `torch_geometric`, `wandb`, and the test tools | development on a laptop, and continuous integration |
| `requirements-gpu-graph.txt` | CUDA torch 2.2.2, `torch_geometric`, `torch_scatter`, `torch_cluster`, `timm`, `wandb` | training graph models on a GPU |
| `requirements-gpu-images.txt` | CUDA torch 2.2.2, `spconv-cu121`, `scipy`, `timm`, `wandb` | training image and multi-ring models on a GPU |
| `requirements-full.txt` | both GPU bundles combined | one environment for every representation |

```bash
pip install -r requirements-gpu-graph.txt      # for example
```

Two properties are worth noting, because they explain the shape of the table.

**The GPU bundles pin `torch==2.2.2` and select a matching wheel index.** The pinned
version is the one the cluster's own training image runs, and `torch_scatter` and
`torch_cluster` are published as prebuilt wheels compiled against an exact PyTorch and
CUDA combination. The `--find-links` line in those files is what selects the matching
builds. Installing a different PyTorch alongside them produces an `undefined symbol`
error on import rather than a failure during installation.

**`requirements-ci.txt` deliberately omits `torch_scatter`, `torch_cluster`, `spconv` and
`timm`.** None is needed by anything the test suite exercises, `spconv` has no macOS or
CPU wheel at all, and omitting them keeps a laptop environment installable in one command.

!!! note "The base install is a tested boundary"
    `requirements.txt` alone is sufficient for the image representation, and this is
    verified rather than intended: one continuous-integration job installs it and nothing
    else, and fails if PyTorch Geometric or `wandb` become reachable from an eager import
    on a shared code path.

## Verifying the installation

Hydra can compose and print a configuration without executing it. This resolves every
`_target_` in the configuration, and therefore imports every class a run would
instantiate:

```bash
python main.py --config-name resnet_train -c job
```

A successful run prints the composed configuration and exits. A `ModuleNotFoundError` or
`undefined symbol` at this point identifies a missing or mismatched dependency before any
data is read.

## Next

[Quickstart](quickstart.md) trains and evaluates a model on a small published dataset.
