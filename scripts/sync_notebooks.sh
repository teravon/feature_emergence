#!/bin/bash
# Regenerate the committed analysis notebooks from their .py sources.
#
# The "canonical" source of truth is the percent-format script in analysis/;
# the .ipynb files in notebooks/ are generated artifacts kept in sync with
# jupytext, so they render on GitHub without manual maintenance.
#
# Usage:
#   scripts/sync_notebooks.sh           # sync all
#   scripts/sync_notebooks.sh analysis/01_plot_traces.py
set -euo pipefail
cd "$(dirname "$0")/.."

SRC_DIR="analysis"
NB_DIR="notebooks"

mkdir -p "$NB_DIR"

fail() { echo "error: $1 (try: pip install -e .[dev])" >&2; exit 1; }
if ! python -c "import jupytext" >/dev/null 2>&1; then
    fail "jupytext not found"
fi

convert() {
    local src="$1"
    local name
    name="$(basename "$src")"
    python -m jupytext --to notebook -o "$NB_DIR/${name%.py}.ipynb" "$src"
}

if [ "$#" -ge 1 ]; then
    for src in "$@"; do
        convert "$src"
    done
else
    for src in "$SRC_DIR"/*.py; do
        [ -e "$src" ] || continue
        convert "$src"
    done
fi

echo "Notebooks synced into $NB_DIR/"