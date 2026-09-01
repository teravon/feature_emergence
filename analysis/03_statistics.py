# %% [markdown]
# # Dataset statistics
#
# Quick numerical profile of each dataset: shape, dtype, and the basic
# statistics of the power traces. This helps to understand the *scale* and
# *noise* we are dealing with before training anything.
#
# This script produces `outputs/figures/03_stats_table.png` (the table is also
# printed in the console).

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from feature_emergence.config import OUTPUTS_DIR
from feature_emergence.utils import load_dataset

# %%
# We work on a subset of each dataset to keep things fast; the full datasets
# are much larger but the numbers are representative.
CONFIGS = [
    ("ASCADr",   load_dataset("ascad-variable", traces_dim=2000,  leakage_model="ID", n_prof=20000), 20000),
    ("ESHARD",   load_dataset("eshard",         traces_dim=1400,  leakage_model="HW", n_prof=20000), 20000),
    ("CHES_CTF", load_dataset("ches_ctf",       traces_dim=15000, leakage_model="HW", n_prof=30000), 30000),
]

# %%
def trace_stats(x):
    # cast to float32: CHES_CTF traces are float16 and global sums overflow
    x = x.astype(np.float32)
    per_trace_mean = x.mean(axis=1)
    per_trace_std = x.std(axis=1)
    return (
        x.shape,
        str(x.dtype),
        round(float(x.mean()), 3),
        round(float(x.std()), 3),
        float(x.min()),
        float(x.max()),
        round(float(per_trace_std.mean()), 3),
    )

rows = [("dataset", "shape", "dtype", "mean", "std", "min", "max", "mean(trace std)")]
for name, ds, _ in CONFIGS:
    rows.append((name,) + trace_stats(ds.x_profiling))

for row in rows:
    print(" | ".join(str(c) for c in row))

# %%
fig, ax = plt.subplots(figsize=(12, 3))
ax.axis("off")
tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center")
tbl.set_fontsize(11)
tbl.scale(1, 1.8)
ax.set_title("Profiling trace statistics per dataset", pad=12)

out = OUTPUTS_DIR / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "03_stats_table.png", dpi=150, bbox_inches="tight")
print("saved:", out / "03_stats_table.png")

# %%