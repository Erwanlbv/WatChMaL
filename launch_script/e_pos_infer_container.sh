#!/bin/bash
#SBATCH -A hyperk
#SBATCH -p gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G                     # Reduced from 256G to fit node capacity (150-186 GB max)
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH -J mr_seg_epos_infer
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.err

# Usage: sbatch e_pos_infer_container.sh

# 1. Prevent PyTorch CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail

echo "Job: ${SLURM_JOB_NAME:-} (${SLURM_JOB_ID:-}) on $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# Root of NeuralNetworks_Software — auto-detected from this script's location
# (<repo>/tutorial_launch/caverns/container -> three levels up; override with NEUNET_ROOT).
REPO_ROOT=/sps/hyperk/zhu/WatChMaL
echo "Binding repo: ${REPO_ROOT}"
if [[ ! -f "${REPO_ROOT}/main.py" ]]; then
    echo "ERROR: ${REPO_ROOT} does not look like the repo root (no main.py)." >&2
    echo "       Run sbatch from the NeuralNetworks_Software root." >&2
    exit 1
fi

IMAGE=/sps/hyperk/zhu/WatChMaL/env/ml_image.sif

HOST_DATA="/sps/hyperk/zhu/WatChMaL/datasets/seg_mathias"
WORKDIR="/workspace/work/ml"
DATA_MOUNT="/workspace/work/data"

BIND_DATA="${HOST_DATA}:${DATA_MOUNT}"
BIND_CODE="${REPO_ROOT}:${WORKDIR}"
BIND_INDEX=/sps/hyperk/zhu/WatChMaL/datasets/seg_mathias/index_lists:/workspace/work/index_lists
BIND_GEO="/sps/hyperk/common/resnet_hkfd_jun26_modelcards/geometry:/workspace/work/geometry"
BIND_BEST="/sps/hyperk/common/resnet_hkfd_jun26_modelcards:/workspace/work/best"

echo "Image: ${IMAGE}"
echo "Data : ${HOST_DATA}  ->  ${DATA_MOUNT}"
echo "Code : ${REPO_ROOT}  ->  ${WORKDIR}"

# Build the in-container command (outer shell expands the paths below). Uses the
# same launch pattern as run_main_cclyon.sh: --config-path points at main/, and
# hydra.searchpath adds the config/caverns root so the group defaults resolve.
PYCMD="HYDRA_FULL_ERROR=1 python main.py \
    --config-path ${WORKDIR}/config/main --config-name hkfd_jun26_hx2_regression_e_pos_infer \
    hydra.searchpath=[${WORKDIR}/config] \
    gpu_list=[0]"

srun apptainer exec --nv \
  --bind "$BIND_CODE" \
  --bind "$BIND_DATA" \
  --bind "$BIND_INDEX" \
  --bind "$BIND_GEO" \
  --bind "$BIND_BEST" \
  --pwd /workspace/work/ml \
  "$IMAGE" \
  bash -lc "$PYCMD" "$@"
