#!/bin/bash
# Sync generated artifacts into the docs so they render both on GitHub and on
# the MkDocs site.
#
#   outputs/figures/*.png   →  docs/assets/figures/      (figures shown in chapters)
#   data/README.md          →  docs/appendix_datasets.md (datasets guide page)
#
# (Notebooks live directly in docs/notebooks/ — see scripts/sync_notebooks.sh.)
#
# The sources are the source of truth; the copies under docs/ are committed
# build artifacts. Re-run after regenerating figures or editing data/README.md.
#
# Usage:
#   scripts/sync_docs.sh
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p docs/assets/figures
cp outputs/figures/*.png docs/assets/figures/
cp data/README.md docs/appendix_datasets.md

echo "Docs assets synced (figures + datasets guide)"
