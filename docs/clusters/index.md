# Running on a cluster

WatChMaL runs from a checkout, so a cluster run is: get the code somewhere the compute
nodes can see, get an environment with the right dependencies, and point a config at the
shared data.

The environment is the hard part, and on CC-IN2P3 the answer is **containers** rather than
a conda environment. A conda tree on GPFS is thousands of small files; `import torch` from
one has been measured at over 100 s on a login node, while an apptainer image is a single
squashfs file that imports normally.

## Pages

- [CC-IN2P3 — containers](cc-in2p3-containers.md): which images exist, what each ships,
  and which one you need.
- [CC-IN2P3 — detectors & data](cc-in2p3-detectors.md): what can be trained out of the
  box, per detector, with reference dataset paths.
- [CC-IN2P3 — paths to change](cc-in2p3-paths.md): what in the shipped configs is yours
  and what is a shared reference you can leave alone.

## Which container for which task

The images are not interchangeable. This table is the short version of
[containers](cc-in2p3-containers.md), verified 2026-07-30 by importing each package inside
each image:

| Your task | Image under `/sps/hyperk/containers/ml/` | Why |
|---|---|---|
| Multi-ring segmentation | `ml_image.sif` | the only image with `spconv` |
| Graph models, newest stack | `pytorch_pyg_cu130_v1.1.sif` | full PyG incl. `pyg_lib`; **H100 only** (CUDA 13.0) |
| Graph models, or anything needing `analysis/` | `container_base_ml_v4.0.0.sif` | full PyG **and** the only image that runs the whole analysis layer |
| Transformers (ViT / Swin / T2T) | `ml_image.sif` | the only image with `timm` |

!!! warning "`analysis/` does not run in every image"
    `analysis/read.py` imports `uproot` and `analysis/regression.py` imports `tabulate`,
    both at module level and both in base `requirements.txt` — but both are missing from
    images people are told to use. Only `container_base_ml_v4.0.0.sif` has both. Details
    and the workaround: [containers](cc-in2p3-containers.md#analysis-support).

## The usage pattern

Every launch script has the same shape: bind your checkout and your data into the image,
run `main.py` from the bound working directory.

```bash
apptainer exec --nv \
  --bind /path/to/WatChMaL:/workspace/work/ml \
  --bind /path/to/your/data:/workspace/work/data \
  --pwd /workspace/work/ml \
  /sps/hyperk/containers/ml/<IMAGE>.sif \
  python main.py --config-path config/caverns/main --config-name gat_classification
```

Ready-made wrappers live in `tutorial/launch/` (copy them to `launch/` with
`setup/make_dirs.sh` first) — see the READMEs in
[`tutorial/launch/caverns/`](https://github.com/WatChMaL/WatChMaL/tree/master/tutorial/launch/caverns)
and its `container/` subdirectory.

## Other sites

Only CC-IN2P3 is documented here, because it is the only site whose paths and images have
been verified. WatChMaL is known to run in containers on other clusters (Compute Canada,
IPMU) — if you run it somewhere else, a page describing the images and data locations
would be a welcome contribution. The three CC-IN2P3 pages are a usable template.
