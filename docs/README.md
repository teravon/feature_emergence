# Home — A guided tour of feature emergence in side-channel analysis

This repository is a readable, self-contained companion to the paper
*"You have to be Realistic: On Investigating Feature Emergence in
Deep Learning-based Side-channel Analysis"*. Everything here is written to be
followed step by step — no prior knowledge of side-channel analysis or deep
learning is required.

## How to read this material

```
docs/
├── 00_HOME.md                 ← you are here
├── 01_introduction.md         ← what a side-channel attack is and why it matters
├── 02_the_datasets.md         ← the three datasets we work with
├── 03_exploring_the_data.md   ← how to read power traces and quantify leakage
├── 04_models_and_training.md  ← the neural networks used for the attack
├── 05_feature_emergence.md    ← the central question: when are features learned?
└── 06_reproducibility.md      ← how to run everything yourself
```

Each chapter is written as plain Markdown and illustrated with figures that
are committed in this repository, so the whole story reads comfortably on
GitHub without running any code.

## Where the figures come from

The figures you see in the docs are produced by small, readable scripts in
`analysis/` — one per chapter:

| Chapter | Analysis script | Produces |
|---------|-----------------|----------|
| 03 | `analysis/01_plot_traces.py` | raw trace plots |
| 03 | `analysis/02_snr.py` | SNR leakage curves |
| 03 | `analysis/03_statistics.py` | dataset statistics |
| 05 | `analysis/04_pi_curves.py` | Perceived Information vs epochs |

Those same scripts double as interactive Jupyter notebooks (kept in sync with
Jupytext under `notebooks/`), so you can re-run, tweak and experiment.

## What you need to follow along

- For **reading** the manual: nothing. It is plain Markdown.
- For **running** the analysis: the Python environment described in
  `06_reproducibility.md` (Docker image or conda) plus the datasets in `data/`.

## Repository layout (one glance)

```
src/feature_emergence/   reusable Python package (loaders, utilities, models)
analysis/                the scripts behind every figure in the docs
notebooks/               interactive copies of the analysis scripts
docs/                    the guided manual you are reading now
data/                    the datasets (large, downloaded, gitignored)
models/                  601 model checkpoints (gitignored)
outputs/figures/         the committed figures shown in the docs
```

Next: [Chapter 1 — Introduction](01_introduction.md).