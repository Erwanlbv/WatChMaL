#!/bin/bash

# Create your personal workspace from the shipped tutorial tree.
#
# Usage (from the repo root):
#   bash setup/make_dirs.sh              create config/ and launch/
#   bash setup/make_dirs.sh -f           overwrite them even if they already exist
#   bash setup/make_dirs.sh --help
#
# The copies are your local scratch; the tracked tutorial/ tree stays the reference.
# Copies whichever of config/launch exist, so it also works mid-merge while the caverns
# and launch trees are still being added.
#
# It REFUSES to run when config/ or launch/ is already there, because the copy
# overwrites file by file: an existing workspace would silently lose every local edit
# (dataset paths, index lists, tweaked configs), and both directories are gitignored,
# so there is nothing to recover from. Pass --force once you are sure.

set -euo pipefail

FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--force) FORCE=1 ;;
    -h|--help)  sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)          echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
  esac
  shift
done

# Check we are at the repo root
if [[ ! -d "tutorial" ]]; then
    echo "Error: run this from the WatChMaL root directory ('tutorial/' not found)" >&2
    exit 1
fi

# Refuse to clobber an existing workspace unless asked
if [[ "$FORCE" -eq 0 ]]; then
    existing=()
    [[ -e config ]] && existing+=("config/")
    [[ -e launch ]] && existing+=("launch/")
    if [[ ${#existing[@]} -gt 0 ]]; then
        echo "Error: ${existing[*]} already exists in $(pwd)." >&2
        echo "" >&2
        echo "Copying over it would overwrite your local edits file by file, and these" >&2
        echo "directories are gitignored, so the previous contents cannot be recovered." >&2
        echo "" >&2
        echo "  - to keep what you have:  do nothing, your workspace is ready to use" >&2
        echo "  - to start again:         back it up, then re-run with -f (or --force)" >&2
        echo "" >&2
        echo "      bash setup/make_dirs.sh --force" >&2
        exit 1
    fi
fi

echo "Welcome to WatChMaL!"

mkdir -p config launch
# `src/.` copies the *contents* of src (portable across GNU/BSD cp, no nesting)
[[ -d tutorial/config ]] && cp -R tutorial/config/. config/
[[ -d tutorial/launch ]] && cp -R tutorial/launch/. launch/

echo "Copied tutorial/config -> config/ and tutorial/launch -> launch/ (whichever exist)."
echo "Customize these freely; keep the tracked tutorial/ tree as the reference."
