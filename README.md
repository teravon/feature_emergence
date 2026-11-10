# Interpreting Feature Emergence in Deep Learning-based Side-Channel Analysis

This repository reproduces and extends the study
[*"You have to be Realistic: On Investigating Feature Emergence in Deep
Learning-based Side-channel Analysis"*](https://github.com/Sengim/feature_emergence)
(Karayalcin et al., 2025): when a neural network learns to recover a secret
key from power measurements, at what point during training does it learn the
features that make the attack work?

**To understand the ideas, read the guide:
[qwertylab.dev/guides/feature-emergence](https://qwertylab.dev/guides/feature-emergence/).**
It explains side channels, masking, the datasets, the networks and the attack
from scratch. This repository holds the code, data pointers and notebooks
behind it; this README covers only how to run them.

## Setup

```bash
scripts/download_files.sh        # trace datasets + model checkpoints from Zenodo (Windows: download_files.ps1)
docker build -t realistic-env .  # verified environment: Python 3.9, TensorFlow 2.15
docker run -it --rm -p 8888:8888 realistic-env
```

Then open <http://localhost:8888>: Jupyter starts in `notebooks/`. No GPU is
required; the published checkpoints in `models/` are used instead of
re-training.

The download fills `data/` (three trace datasets, about 2.2 GB) and `models/`
(601 checkpoints, one per training epoch, about 2 GB). Both are excluded from
git. What each file contains is documented in [`data/README.md`](data/README.md).

## Reproducing the figures

Each script in `analysis/` writes its figures to `outputs/figures/`
(gitignored). Run them from the repository root inside the environment above,
e.g. `python analysis/01_the_data.py`.

| Script | Produces | Guide chapter |
|--------|----------|---------------|
| `analysis/01_the_data.py` | traces, dataset statistics, SNR | [Datasets](https://qwertylab.dev/guides/feature-emergence/05-the-datasets/), [SNR](https://qwertylab.dev/guides/feature-emergence/06-where-the-key-leaks/) |
| `analysis/02_the_models.py` | model summaries, weight heatmaps | [Network](https://qwertylab.dev/guides/feature-emergence/07-the-network/), [Training](https://qwertylab.dev/guides/feature-emergence/08-training-and-checkpoints/) |
| `analysis/03_the_attack.py` | prediction, evidence progress, key ranking, guessing entropy | [Scoring the attack](https://qwertylab.dev/guides/feature-emergence/09-scoring-the-attack/) |
| `analysis/04_perceived_information.py` | Perceived Information per epoch | [Perceived Information](https://qwertylab.dev/guides/feature-emergence/10-perceived-information/) |
| `analysis/05_traces_to_ge1.py` | traces to GE=1 on ASCADr, ESHARD and CHES_CTF | [How many traces](https://qwertylab.dev/guides/feature-emergence/09b-how-many-traces/) |
| `analysis/06_learning_evidence.py` | PI and P(true value) per epoch; one trace early vs late | [What changes during learning](https://qwertylab.dev/guides/feature-emergence/09c-what-learning-changes/) |

Scripts `01`–`04` are also notebooks (Jupytext percent format), executed and
committed in [`notebooks/`](notebooks/README.md). After editing one of them,
regenerate its notebook with `scripts/sync_notebooks.sh`.

## Tests

```bash
pip install --no-deps -e . && pip install pytest
pytest
```

The smoke tests check that the package imports and resolves its paths; they
need neither TensorFlow nor the datasets, so any recent Python works.

## Repository layout

```
src/feature_emergence/   the Python package: loaders, models, metrics
analysis/                the scripts behind every figure
notebooks/               executed companion notebooks; original/ holds the paper's notebooks, unmodified
data/                    the trace datasets (downloaded, not tracked); README.md documents them
models/                  model checkpoints, one per epoch (downloaded, not tracked)
outputs/                 generated figures (not tracked)
scripts/                 download and notebook-sync scripts
tests/                   smoke tests
```

## Credits

Original code and research by [Sengim Karayalcin](https://github.com/Sengim)
and co-authors. This repository is a fork of
[Sengim/feature_emergence](https://github.com/Sengim/feature_emergence);
the original authors retain all rights to their work. See [LICENSE](LICENSE).
