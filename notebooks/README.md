# Notebooks

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

Our own, clean notebooks (using the `feature_emergence` package) live directly
in this folder.