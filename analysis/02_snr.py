# %% [markdown]
# # Quantify leakage with the Signal-to-Noise Ratio (SNR)
#
# The SNR tells us *at which time samples* the power consumption depends on our
# target intermediate value. We compute it on the profiling set of each
# dataset, for the leakage model each dataset actually uses in the paper
# (identity for ASCADr, Hamming weight for ESHARD and CHES_CTF).
#
# This script produces `outputs/figures/03_snr_curves.png`.

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_emergence.config import OUTPUTS_DIR
from feature_emergence.utils import load_dataset, snr_fast

# %% [markdown]
# ### Loading and computing
#
# `snr_fast(x, y)` groups the traces `x` by their label `y` and computes, for
# every time sample, the ratio:
#
# ```
# SNR(t) = variance(means of each group) / mean(variances inside each group)
# ```
#
# High SNR at sample `t`  → that sample leaks information about the label.

# %%
CONFIGS = [
    ("ASCADr",   load_dataset("ascad-variable", traces_dim=2000,  leakage_model="ID", n_prof=20000)),
    ("ESHARD",   load_dataset("eshard",         traces_dim=1400,  leakage_model="HW", n_prof=20000)),
    ("CHES_CTF", load_dataset("ches_ctf",       traces_dim=15000, leakage_model="HW", n_prof=30000)),
]

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, ds) in zip(axes, CONFIGS):
    snr = snr_fast(ds.x_profiling, ds.profiling_labels)
    ax.plot(snr, lw=0.8)
    ax.set_title(f"{name}")
    ax.set_xlabel("time sample index")
    ax.set_ylabel("SNR")
    ax.grid(alpha=0.3)

fig.suptitle("Signal-to-Noise Ratio per dataset (profiling set)", fontsize=14)
fig.tight_layout()

out = OUTPUTS_DIR / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "03_snr_curves.png", dpi=150)
print("saved:", out / "03_snr_curves.png")

# %%