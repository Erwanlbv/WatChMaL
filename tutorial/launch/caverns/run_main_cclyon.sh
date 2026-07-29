#!/bin/bash

# ============================================================================
# Tutorial: Run Training Interactively (CC-Lyon cluster)
# ============================================================================
# This script runs training directly in your current terminal session.
# Use this for quick tests, debugging, or when you want to see output in real-time.
#
# Usage:
#   1. Edit the settings below (config name, paths, etc.)
#   2. Make sure you're in the correct conda environment
#   3. Run: ./run_main_cclyon.sh
#   4. Press Ctrl+C to stop if needed
# ============================================================================

# ============================================================================
# Training Configuration - EDIT THESE
# ============================================================================
MINICONDA_DIR=${MINICONDA_DIR:-"/sps/t2k/eleblevec/miniconda3"}
PYTHON_ENV_NAME=${PYTHON_ENV_NAME:-"pt28_cuda129"}


# Repo root: found by searching UPWARD for the marker files, not by counting
# directories. setup/make_dirs.sh copies this tree from tutorial/launch/ to launch/,
# which removes one level, so any fixed number of ".." is correct in exactly one of the
# two locations. Override with NEUNET_ROOT to run from outside the repo.
_find_repo_root() {
  local dir="$1"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/main.py" && -d "$dir/watchmal" ]]; then printf '%s' "$dir"; return 0; fi
    dir="$(dirname "$dir")"
  done
  echo "ERROR: no WatChMaL root (main.py + watchmal/) found above $1 - set NEUNET_ROOT" >&2
  return 1
}
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to WatChMaL repository — auto-detected from this script's location, so it
# works on any cluster / any checkout (also from your launch/ copy).
# Override by exporting NEUNET_ROOT if you keep the script outside the repo.
NeuNet_folder_path="${NEUNET_ROOT:-$(_find_repo_root "$_SCRIPT_DIR")}"

# Config tree to compose from: 'tutorial/config/caverns' (shipped examples)
# or 'config' (your own workspace, see the main README part 1 - 4.)
config_folder=tutorial/config/caverns

# Config file name (without .yaml extension)
# Examples:
#   - gcn_classification
#   - gat_classification
#   - gat_vertex_regression
#   - wcte_mpmt_gat_classification
# For multiring_segmentation_train see the container config part.
config_name=gat_vertex_regression

# GPU configuration
# For single GPU: 'gpu_list=[0]'
# For multiple GPUs: 'gpu_list=[0,1]'
# For CPU: 'gpu_list=[]'
gpu_list='gpu_list=[0]'

# Master port for distributed training (only needed for multi-GPU)
# master_port='MASTER_PORT=12357'

# Hydra search path (derived — usually don't need to change)
hydra_searchpath=${NeuNet_folder_path}/${config_folder}

# Pytorch configuration
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# Execution Code - DO NOT MODIFY BELOW
# ============================================================================
export HYDRA_FULL_ERROR=1

echo "=========================================="
echo "Running training interactively"
echo "Config: $config_name"
echo "GPU: $gpu_list"
echo "=========================================="

cd $NeuNet_folder_path
source $MINICONDA_DIR/bin/activate $PYTHON_ENV_NAME

# add -c job aat the end to launch a dry run
# (main config will be displayed & no training will be performed), 
python \
    main.py \
    --config-path=${hydra_searchpath}/main \
    --config-name=$config_name \
    hydra.searchpath=[$hydra_searchpath] \
    $gpu_list \
    $master_port \
    # launch_wandb=False \
