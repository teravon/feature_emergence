# %% [markdown]
# # Plot raw power traces
#
# The simplest way to get a feel for the data: plot a handful of profiling
# traces from each dataset and look at their *shape*.
#
# This script produces `outputs/figures/03_traces_overview.png`, used in
# `docs/03_exploring_the_data.md`.

# %%
import matplotlib

matplotlib.use("Agg")  # headless backend (no display needed)
import matplotlib.pyplot as plt

from feature_emergence.config import OUTPUTS_DIR
from feature_emergence.utils import load_dataset

N_TRACES = 5  # how many traces to overlay per dataset

# %%
# We load all three datasets. To keep the script light and fast we limit the
# number of profiling traces actually read from disk.
CONFIGS = [
    ("ASCADr",   load_dataset("ascad-variable", traces_dim=2000,  leakage_model="ID", n_prof=5000)),
    ("ESHARD",   load_dataset("eshard",         traces_dim=1400,  leakage_model="HW", n_prof=5000)),
    ("CHES_CTF", load_dataset("ches_ctf",       traces_dim=15000, leakage_model="HW", n_prof=5000)),
]

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, ds) in zip(axes, CONFIGS):
    for i in range(N_TRACES):
        ax.plot(ds.x_profiling[i], lw=0.8, label=f"trace {i}")
    ax.set_title(f"{name}  ·  {ds.x_profiling.shape[1]} samples")
    ax.set_xlabel("time sample index")
    ax.set_ylabel("power (ADC units)")
    ax.legend(fontsize=8)

fig.suptitle("Example profiling power traces (first 5 traces per dataset)", fontsize=14)
fig.tight_layout()

out = OUTPUTS_DIR / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "03_traces_overview.png", dpi=150)
print("saved:", out / "03_traces_overview.png")

# %%