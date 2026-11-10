# Notebooks

## Companion notebooks

| Notebook | Source | What it does | Guide chapter |
|----------|--------|--------------|---------------|
| [01_the_data](01_the_data.ipynb) | `analysis/01_the_data.py` | opens the three datasets: traces, statistics, SNR | [Datasets](https://qwertylab.dev/guides/feature-emergence/05-the-datasets/), [SNR](https://qwertylab.dev/guides/feature-emergence/06-where-the-key-leaks/) |
| [02_the_models](02_the_models.ipynb) | `analysis/02_the_models.py` | builds the MLP/CNN, inspects and loads a checkpoint | [Network](https://qwertylab.dev/guides/feature-emergence/07-the-network/), [Training](https://qwertylab.dev/guides/feature-emergence/08-training-and-checkpoints/) |
| [03_the_attack](03_the_attack.ipynb) | `analysis/03_the_attack.py` | recovers a key byte; evidence, ranking, guessing entropy | [Scoring the attack](https://qwertylab.dev/guides/feature-emergence/09-scoring-the-attack/) |
| [04_perceived_information](04_perceived_information.ipynb) | `analysis/04_perceived_information.py` | Perceived Information per training epoch | [Perceived Information](https://qwertylab.dev/guides/feature-emergence/10-perceived-information/) |

The notebooks are generated from the percent-format scripts in
[`analysis/`](../analysis/), which are the source of truth. After editing a
script, regenerate its notebook:

```bash
scripts/sync_notebooks.sh                      # all
scripts/sync_notebooks.sh analysis/01_the_data.py
```

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
