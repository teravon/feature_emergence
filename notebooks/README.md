# Notebooks

This folder holds only the **original** notebooks shipped upstream, kept in
`original/` as unmodified reference material.

Our own notebook — the hands-on companion of the manual's chapters — lives in
[`docs/notebooks/`](../docs/notebooks/) so that a single copy renders both on
GitHub and on the documentation site. It is generated from the scripts in
`analysis/` via `scripts/sync_notebooks.sh` (Jupytext).

## `original/`

Byte-for-byte copies of the original notebooks shipped in the upstream
`Sengim/feature_emergence` repository. They are kept as reference material and
are **not** modified.

Note: those notebooks were written for the original repo layout (`./datasets`,
`model_checkpoints/`, top-level `utils.py`/`profiling_and_attack.py` imports).
To execute them as-is:

1. `pip install -e .`
2. Patch the dataset path in their second cell:
   `path = "./datasets"` -> `path = "./../data"`
3. Patch checkpoint paths: `model_checkpoints/...` -> `models/...`
