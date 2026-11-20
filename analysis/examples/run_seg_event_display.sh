#!/bin/bash
# Opens the segmentation event display (analysis/seg_display.py). See README.md next to this file.
#
#   SEG_DISPLAY_DATA=/path/to/data bash analysis/examples/run_seg_event_display.sh [FILE] [EVENT] [VIEW]
#
# FILE   a ring-segmented h5 in the data folder (default recon_seg_b0000.h5)
# EVENT  first event shown (default 4)
# VIEW   first view: all (default), true, reco or custom
#
# For seg_b0000 / recon_seg_b0000, the WCSim file they come from (wcsim_src_b0000.h5), if present in
# the data folder, is passed as --truth: the real whole event and the true hits of each particle.
# In the window: <- / -> events, 1 2 3 4 views, C / T charge / time, L log, D dark / light,
# double-click a panel to show it alone, Esc quit.
set -euo pipefail

FILE=${1:-recon_seg_b0000.h5}
EVENT=${2:-4}
VIEW=${3:-all}
DATA=${SEG_DISPLAY_DATA:?set SEG_DISPLAY_DATA to the folder holding the h5 files and the mapping npz}
MAPPING=$DATA/HK_HybridmPMT_WithOD_Realistic_mapping_squashed.npz
REPO=$(cd "$(dirname "$0")/../.." && pwd)      # WatChMaL repo root

for f in "$DATA/$FILE" "$MAPPING"; do
    [ -f "$f" ] || { echo "[run] missing $f" >&2; exit 1; }
done

TRUTH=()
case "$FILE" in
    seg_b0000.h5|recon_seg_b0000.h5)
        [ -f "$DATA/wcsim_src_b0000.h5" ] && TRUTH=(--truth "$DATA/wcsim_src_b0000.h5") ;;
esac

# python env with h5py, numpy, PyQt6 and pyqtgraph: edit this line for yours
# conda activate <python-env>

cd "$REPO"                                     # so that `analysis.*` imports
python -m analysis.seg_display "$DATA/$FILE" --mapping "$MAPPING" --event "$EVENT" --view "$VIEW" \
    ${TRUTH[@]+"${TRUTH[@]}"}
