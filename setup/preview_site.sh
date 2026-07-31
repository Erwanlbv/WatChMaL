#!/bin/bash
#
# Build and serve the documentation site locally, exactly as it is published.
#
# Usage (from the repository root):
#   bash setup/preview_site.sh            build both layers and serve on :8000
#   bash setup/preview_site.sh -p 9000    use another port
#   bash setup/preview_site.sh --docs     mkdocs live-reload, docs only
#
# The published site has two layers: the hand-written landing page at the root, and the
# MkDocs site beneath it at /docs/. This serves them under /WatChMaL/ so that
# root-relative links - notably the docs header logo, which points at /WatChMaL/ - resolve
# the same way locally as they do on GitHub Pages.
#
# --docs runs `mkdocs serve` instead, which rebuilds on save. It is the faster loop while
# writing prose, but it serves only the docs and does not reproduce the root-relative
# links, so check the full build before pushing.

set -euo pipefail

PORT=8000
DOCS_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port) PORT="$2"; shift 2 ;;
    --docs)    DOCS_ONLY=1; shift ;;
    -h|--help) sed -n '3,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
  esac
done

if [[ ! -f mkdocs.yml ]]; then
  echo "Error: run this from the WatChMaL root directory ('mkdocs.yml' not found)" >&2
  exit 1
fi

if ! python -c "import mkdocs" 2>/dev/null; then
  echo "Error: mkdocs-material is not installed. Install it with:" >&2
  echo "    pip install 'mkdocs-material~=9.7'" >&2
  exit 1
fi

if [[ "$DOCS_ONLY" -eq 1 ]]; then
  echo "Serving docs with live reload on http://127.0.0.1:${PORT}/"
  exec python -m mkdocs serve -a "127.0.0.1:${PORT}"
fi

OUT="$(mktemp -d -t watchmal-site.XXXXXX)"
trap 'rm -rf "$OUT"' EXIT

mkdir -p "$OUT/WatChMaL"
cp -r landing/. "$OUT/WatChMaL/"
python -m mkdocs build --strict -d "$OUT/WatChMaL/docs" >/dev/null

echo
echo "  landing page : http://127.0.0.1:${PORT}/WatChMaL/"
echo "  documentation: http://127.0.0.1:${PORT}/WatChMaL/docs/"
echo
echo "Ctrl-C to stop. Re-run after editing; this build is not live-reloading."
echo
exec python -m http.server "$PORT" --bind 127.0.0.1 -d "$OUT"
