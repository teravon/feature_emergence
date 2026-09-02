# Interpreting Feature Emergence in Deep Learning-based Side-Channel Analysis

This repository reproduces and extends the study
[*"You have to be Realistic: On Investigating Feature Emergence in Deep
Learning-based Side-channel Analysis"*](https://github.com/Sengim/feature_emergence)
(Karayalcin et al., 2025).

The question it investigates: when a neural network is trained to recover a
secret cryptographic key from power-consumption measurements, **at what point
during training does it learn the features that make the attack work?**

The repository is written for a reader with no prior knowledge of
side-channel analysis or deep learning. Every claim is backed by a figure,
and every figure can be regenerated from the scripts in this repository.

## Reading guide

The repository is organized as a guided manual in [`docs/`](docs/README.md).
Each section below links to its chapter and to the notebook or code that
produces its figures.

| # | Section | Chapter | Companion code |
|---|---------|---------|----------------|
| 1 | **Introduction** — what a side-channel attack is, power traces, leakage models | [docs/01_introduction.md](docs/01_introduction.md) | — |
| 2 | **The data** — three AES trace datasets, trace shapes, scale and noise, SNR leakage | [docs/02_the_data.md](docs/02_the_data.md) | [docs/notebooks/01_the_data.ipynb](docs/notebooks/01_the_data.ipynb) |

Start at [docs/README.md](docs/README.md) for the full tour.

## Quick start

```bash
scripts/download_files.sh        # datasets + model checkpoints (from Zenodo)
docker build -t realistic-env .  # verified environment: Python 3.9, TF 2.15
docker run -it --rm -p 8888:8888 realistic-env
```

Then open <http://localhost:8888> and run the notebook in `notebooks/`, or run
the analysis script directly with `python analysis/01_the_data.py`.

No GPU is required: the published checkpoints in `models/` are used instead
of re-training.

## Repository layout

```
docs/         the guided manual, incl. the executed notebook (docs/notebooks/)
analysis/     the scripts behind every figure and notebook in the docs
notebooks/    the original paper's notebooks (reference, unmodified)
data/         the three trace datasets (downloaded, not tracked in git)
models/       model checkpoints, one per training epoch (downloaded, not tracked)
src/          the feature_emergence Python package (loaders, models, metrics)
scripts/      download/setup/utility scripts
outputs/      generated figures (not tracked; docs show docs/assets/figures/)
```

## Credits

Original code and research by [Sengim Karayalcin](https://github.com/Sengim)
and co-authors. This repository is a fork of
[Sengim/feature_emergence](https://github.com/Sengim/feature_emergence);
the original authors retain all rights to their work. See [LICENSE](LICENSE).
