# Chapter 6 — Reproducibility

If you want to run everything yourself — explore, re-train, or just play with
the notebooks — here is how.

## 1. Get the datasets and model checkpoints

Large binary artifacts are **not** stored in git. Download them with the
provided script (reads from the Zenodo record):

```bash
scripts/download_files.sh
```

This creates:

```text
data/     ← eshard.h5, ASCADr/…, ches_ctf/…
models/   ← 602 checkpoints (one per epoch, per model)
```

> You can also copy files manually into `data/` following
> [`data/README.md`](../data/README.md).

## 2. Set up the environment

### Option A — Docker (easiest, recommended)

```bash
docker build -t realistic-env .
docker run -it --rm -p 8888:8888 realistic-env
```

Then open <http://localhost:8888>. Jupyter shows the `notebooks/` folder.

### Option B — conda

```bash
bash scripts/install_environment.sh          # creates env "you-have-to-be-realistic"
conda activate you-have-to-be-realistic
pip install -e .[dev]                        # installs the package + jupytext
```

## 3. Run the analysis scripts

From the repository root:

```bash
python analysis/01_plot_traces.py    # raw traces    → outputs/figures/03_traces_overview.png
python analysis/02_snr.py            # SNR curves    → outputs/figures/03_snr_curves.png
python analysis/03_statistics.py     # stats table   → outputs/figures/03_stats_table.png
python analysis/04_pi_curves.py      # PI curves     → outputs/figures/05_pi_curves.png
```

## 4. Regenerate the notebooks

The notebooks are derived from the analysis scripts (single source of truth):

```bash
scripts/sync_notebooks.sh
```

## 5. Reproduce the paper's results

The author's original notebooks are kept byte-for-byte in
`notebooks/original/`. They contain the full pipelines (PI curves, layer
diagnostics, patching). To run them against the new layout:

- install the package (`pip install -e .`)
- set `path = "../data"` (they were written for `"./datasets"`)
- change checkpoint paths from `model_checkpoints/` to `models/`

## Performance notes

- No GPU assumed: with checkpoints you **do not re-train**, so CPU is enough.
- `04_pi_curves.py` evaluates 100 checkpoints → allow several minutes on CPU.
- Patching notebooks are memory hungry: `docker run -m 12g` if the kernel dies.