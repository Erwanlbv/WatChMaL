#!/bin/bash
#
# Fetch the small published dataset used by the quickstart.
#
# Usage (from the repository root):
#   bash setup/download_data.sh                 download into ./data/quickstart
#   bash setup/download_data.sh -d /some/path   download somewhere else
#   bash setup/download_data.sh --url URL       override the source
#
# The bundle holds 200 simulated electron and 200 simulated muon events from a Hyper-K
# style tank, already built as PyTorch Geometric graph datasets with k-nearest-neighbour
# edges, plus the train/validation/test split that indexes them. It is 25 MB compressed
# (145 MB unpacked) and is enough to train, evaluate and analyse a model on a CPU.
#
# The bundle is NOT in git: graph datasets are dominated by their edge lists, which are
# derived data and would bloat the repository for every clone.

set -euo pipefail

DEST="data/quickstart"
URL="https://github.com/WatChMaL/WatChMaL/releases/download/quickstart-data-v1/watchmal-quickstart-v1.tar.gz"
EXPECTED_SHA="b26d97c9c30fba3becf2ecda2e4b4636375f8e1aeacd13de60937c2e6624377a"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dest) DEST="$2"; shift 2 ;;
    --url)     URL="$2"; shift 2 ;;
    --sha256)  EXPECTED_SHA="$2"; shift 2 ;;
    -h|--help) sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
  esac
done

if [[ ! -d tutorial ]]; then
  echo "Error: run this from the WatChMaL root directory ('tutorial/' not found)" >&2
  exit 1
fi

if [[ -d "$DEST" ]] && [[ -n "$(ls -A "$DEST" 2>/dev/null)" ]]; then
  echo "$DEST already exists and is not empty; nothing to do."
  echo "Remove it first to re-download."
  exit 0
fi

mkdir -p "$DEST"
ARCHIVE="$(mktemp -t watchmal-quickstart.XXXXXX).tar.gz"
trap 'rm -f "$ARCHIVE"' EXIT

echo "Downloading $(basename "$URL") ..."
if command -v curl >/dev/null 2>&1; then
  curl -fL --progress-bar "$URL" -o "$ARCHIVE"
elif command -v wget >/dev/null 2>&1; then
  wget -q --show-progress -O "$ARCHIVE" "$URL"
else
  echo "Error: neither curl nor wget is available" >&2
  exit 1
fi

if [[ -n "$EXPECTED_SHA" ]]; then
  echo "Verifying checksum ..."
  if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL="$(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
  else
    ACTUAL="$(shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)"
  fi
  if [[ "$ACTUAL" != "$EXPECTED_SHA" ]]; then
    echo "Error: checksum mismatch" >&2
    echo "  expected $EXPECTED_SHA" >&2
    echo "  actual   $ACTUAL" >&2
    exit 1
  fi
fi

echo "Extracting into $DEST ..."
tar -xzf "$ARCHIVE" -C "$DEST" --strip-components=1

echo
echo "Done. Contents of $DEST:"
find "$DEST" -maxdepth 1 -mindepth 1 | sort | sed 's/^/  /'
echo
echo "Point a dataset config at these paths - see the quickstart:"
echo "  https://watchmal.github.io/WatChMaL/docs/getting-started/quickstart/"
