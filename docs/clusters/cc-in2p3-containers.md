# CC-IN2P3: available apptainer images

Ready-to-use `.sif` images for running this framework on CC-IN2P3 (CC-Lyon). Which one you
need depends on the model family — see [detectors & data](cc-in2p3-detectors.md).

**Contents verified 2026-07-30** by importing each package inside each image, rather than
by carrying forward an earlier version of this page.

## The shared directory

```
/sps/hyperk/containers/ml/
```

is group-readable and holds all three ML images. **Prefer these paths** — they are stable,
whereas per-user copies under `/sps/t2k/<someone>/…` move when that person reorganises.

| Image | Size | python | torch | CUDA | PyG | spconv | timm |
|---|---|---|---|---|---|---|---|
| `ml_image.sif` | 11 G | 3.10.14 | 2.2.2 | 12.1 | 2.5.3 (+ scatter, cluster) | **2.3.8** | 1.0.24 |
| `pytorch_pyg_cu130_v1.1.sif` | 6.1 G | 3.12.3 | 2.11.0 | **13.0** | 2.7.0 (+ scatter, cluster, pyg_lib 0.6.0) | — | — |
| `container_base_ml_v4.0.0.sif` | 14 G | 3.12.3 | 2.6.0a0 (nv25.01) | 12.8 | 2.7.0 (+ scatter, cluster, pyg_lib 0.4.0) | — | — |

All three also ship hydra, wandb, h5py, scikit-learn, scipy, matplotlib and seaborn.

### Which one

- **Multi-ring segmentation** → `ml_image.sif`. The only image with `spconv`.
- **Transformers (ViT / Swin / T2T)** → `ml_image.sif`. The only image with `timm`.
- **Graph models on H100** → `pytorch_pyg_cu130_v1.1.sif`. Newest stack, full PyG
  including `pyg_lib`. Its CUDA 13.0 build means **H100 only — it will not run on V100.**
- **Graph models elsewhere, or any run whose results you want to analyse** →
  `container_base_ml_v4.0.0.sif`.

Build recipe for the PyG image:
`mini-Caverns-toolsbox/env_settings/containers/pytorch_pyg_cu130.def`, with seaborn added
on top via `containers/add_seaborn_v1.1.sh`.

## Analysis support

The trap worth knowing before you plan a run. `analysis/read.py` does a module-level
`import uproot`, and `analysis/regression.py` a module-level `import tabulate`. Both are in
the base `requirements.txt`, and both are missing from images that are otherwise
recommended:

| Image | `analysis/read.py` | `analysis/regression.py` |
|---|---|---|
| `ml_image.sif` | ✗ no `uproot` | ✗ no `tabulate` |
| `pytorch_pyg_cu130_v1.1.sif` | ✓ | ✗ no `tabulate` |
| `container_base_ml_v4.0.0.sif` | ✓ | ✓ |

So **`container_base_ml_v4.0.0.sif` is the only image that runs the analysis layer end to
end**, and the recommended *graph* image cannot run `analysis/regression.py` — exactly the
path the vertex-regression examples need.

Workarounds, in order of preference:

1. Train in whichever image suits the model, then **analyse in
   `container_base_ml_v4.0.0.sif`**. Run outputs are plain `.npy` and `.csv`, so switching
   images between the two steps costs nothing.
2. `pip install --user tabulate uproot` inside the container — writes to your home and
   persists across runs.

This is a gap in the images, not in the code: rebuilding them to satisfy
`requirements.txt` fully would remove the need for either workaround.

## Other images

| Image | Notes |
|---|---|
| `/sps/t2k/melbaz/env/ml_image.sif` | python 3.10.14 / torch 2.2.2 — the **same build** as `containers/ml/ml_image.sif`. Prefer the shared copy. |
| `/sps/hyperk/zhu/CAVERNS/env/ml_image.sif` | likewise the same build. |
| `/sps/t2k/eleblevec/cours/cmhk_ml_tutorial/container_base_ml_vpyg24.05.sif` | python 3.12.3 / torch 2.6.0a0, **PyG 2.6.1** (an earlier version of this page said 2.4), scatter + cluster + `pyg_lib` 0.4.0, `uproot` and `tabulate` present, no `spconv`, no `timm`. A valid graph image — the previous "⚠️ to be verified" flag is resolved. |

!!! note "torch 2.2.2 is old"
    `ml_image.sif` predates a few APIs: `init_process_group(device_id=…)` is absent, and
    `torch.load(..., mmap=True)` rejects a `pathlib.Path` — pass `str()`. The framework
    does not hard-require the newer behaviour, but scripts you write against a newer torch
    may.

## Usage pattern

All launch scripts follow the same shape: bind your checkout and your data into the image,
run `main.py` from the bound repo.

```bash
apptainer exec --nv \
  --bind /path/to/WatChMaL:/workspace/work/ml \
  --bind /path/to/your/data:/workspace/work/data \
  --pwd /workspace/work/ml \
  /sps/hyperk/containers/ml/<IMAGE>.sif \
  python main.py --config-path config/caverns/main --config-name <NAME>
```

Add `--bind /sps:/sps` if your configs reference `/sps` paths directly rather than through
the data bind.

Notes:

- `--nv` exposes the node's GPU; request one via SLURM (`#SBATCH --gres=gpu:v100:1`).
  Remember that `pytorch_pyg_cu130_v1.1.sif` needs an H100, not a V100.
- **`$HOME` is read-only inside some images** — redirect writable state (wandb directories,
  matplotlib cache) as the shipped `train_mr_in_container.sh` and `train_sr_in_container.sh`
  scripts do.
- For the multi-ring image, set `SPCONV_ALGO=native` (see the shipped scripts).

Ready-made wrappers live in `tutorial/launch/caverns/container/`. Copy them into your own
`launch/` with `setup/make_dirs.sh` first — see
[Your own workspace](../getting-started/workspace.md).
