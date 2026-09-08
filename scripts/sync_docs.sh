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

# The appendix is a verbatim copy of data/README.md plus a book-navigation
# footer appended here (the source lives outside the docs and stays clean).
cp data/README.md docs/appendix_datasets.md
cat >> docs/appendix_datasets.md <<'EOF'

---

← Previous: [Chapter 4 — The attack](04_the_attack.md) · [Index](README.md)
EOF

echo "Docs assets synced (figures + datasets guide)"
