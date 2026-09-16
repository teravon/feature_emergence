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

The repository is organized as guided chapters in [`docs/`](docs/README.md).
Each section below links to its chapter and to the notebook or code that
produces its figures.

| # | Section | Chapter | Companion code |
|---|---------|---------|----------------|
| 1 | **Secrets that leak** — side channels and power traces | [docs/01_secrets_that_leak.md](docs/01_secrets_that_leak.md) | — |
| 2 | **Intermediate values** — S-box labels (ID / HW) | [docs/02_intermediate_values.md](docs/02_intermediate_values.md) | — |
| 3 | **Masking** — shares, why one trace fails, SNR | [docs/03_masking.md](docs/03_masking.md) | — |
| 4 | **The datasets** — ASCADr / ESHARD / CHES_CTF | [docs/04_the_datasets.md](docs/04_the_datasets.md) | [docs/notebooks/01_the_data.ipynb](docs/notebooks/01_the_data.ipynb) |
| 5 | **The network** — MLP/CNN as functions on traces | [docs/05_the_network.md](docs/05_the_network.md) | [docs/notebooks/02_the_models.ipynb](docs/notebooks/02_the_models.ipynb) |
| 6 | **Training and checkpoints** — loss, split, epoch weights | [docs/06_training_and_checkpoints.md](docs/06_training_and_checkpoints.md) | (same notebook) |
| 7 | **Scoring the attack** — hypotheses, evidence, GE | [docs/07_scoring_the_attack.md](docs/07_scoring_the_attack.md) | [docs/notebooks/03_the_attack.ipynb](docs/notebooks/03_the_attack.ipynb) |
| 8 | **Perceived Information** — PI vs training epoch | [docs/08_perceived_information.md](docs/08_perceived_information.md) | — |

Start at [docs/README.md](docs/README.md) for the full tour.

## Quick start

```bash
scripts/download_files.sh        # trace datasets + model checkpoints (from Zenodo)
docker build -t realistic-env .  # verified environment: Python 3.9, TF 2.15
docker run -it --rm -p 8888:8888 realistic-env
```

Then open <http://localhost:8888> and run the notebook in `notebooks/`, or run
the analysis script directly with `python analysis/01_the_data.py`.

No GPU is required: the published checkpoints in `models/` are used instead
of re-training.

## Repository layout

```
docs/         the guided chapters, incl. the executed notebook (docs/notebooks/)
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
