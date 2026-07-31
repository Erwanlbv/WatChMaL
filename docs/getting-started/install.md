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

On a machine with [Apptainer](https://apptainer.org) (formerly Singularity) or Docker:

```bash
apptainer build watchmal.sif docker://ghcr.io/watchmal/watchmal:latest
```

The resulting `.sif` is a single file and can be copied between machines.

!!! warning "Publication of the public images is in progress"
    The `ghcr.io/watchmal` registry is not yet populated, so the command above does not
    resolve at the time of writing. Until it does, the images are available on CC-IN2P3
    at `/sps/hyperk/containers/ml/` (see
    [Clusters](../clusters/cc-in2p3-containers.md)), and a local Python environment
    (route 2) is the alternative elsewhere.

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
  watchmal.sif \
  python main.py --config-name resnet_train -c job
```

`--nv` exposes the host GPU and may be omitted for CPU-only work. `--bind` makes a host
directory visible inside the container; both the checkout and the data must be bound,
because a container sees no host path that has not been.

## Route 2 — local Python environment

```bash
git clone https://github.com/WatChMaL/WatChMaL.git
cd WatChMaL
pip install -r requirements.txt
```

This is sufficient for the image representation. The graph and sparse-3-D
representations require additional packages, which are declared in further requirements
files in the repository root; install the ones matching the representation to be used.

`spconv` is deliberately absent from all of them, because it is published as one
distribution per CUDA build (`spconv-cu118`, `spconv-cu120`, and so on) and pinning a
single one would be wrong on every other machine. Install the variant matching the local
CUDA version.

!!! warning "Compiled extensions"
    `torch_scatter`, `torch_cluster` and `pyg_lib` are C++/CUDA extensions. Installing
    them from source against a PyTorch version other than the one they are built for is
    the most common cause of a failed graph installation, and the failure appears as an
    `undefined symbol` error on import rather than during installation. Prefer wheels
    matching the installed PyTorch exactly.

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
