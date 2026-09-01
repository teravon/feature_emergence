# %% [markdown]
# # Perceived Information vs. training epoch
#
# The central figure of the paper: how much does the model leak the key, as a
# function of how long it has been trained?
#
# Thanks to the **checkpoints** (the weights saved after *every* epoch) we
# don't need to retrain anything: we just reload the model as it was at epoch
# k, and measure its Perceived Information (PI) on both the training and attack
# sets. This works on a plain CPU.
#
# This script produces `outputs/figures/05_pi_curves.png` for the ASCADr model
# (`mlp_ascadr`). Same idea applies to the other checkpoints.

# %%
import gc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from feature_emergence.config import MODELS_DIR, OUTPUTS_DIR
from feature_emergence.utils import load_dataset
from feature_emergence.profiling_and_attack import information, mlp

# %%
# --- configuration ---------------------------------------------------------
N_EPOCHS = 100
MODEL_NAME = "mlp_ascadr"
NUM_CLASSES = 256          # identity model -> 256 S-box output classes
WITHIN_EPOCH_PI_SAMPLE = 10000  # traces used to estimate PI on the profiling set

# %%
# --- load ASCADr and scale it exactly like the original notebook ----------
from sklearn.preprocessing import StandardScaler

from feature_emergence.utils import load_dataset, scale_dataset

dataset = load_dataset("ascad-variable", traces_dim=2000, leakage_model="ID", n_prof=100000)
dataset.x_profiling, dataset.x_attack = scale_dataset(
    dataset.x_profiling, dataset.x_attack, StandardScaler()
)

# %%
# --- build the model and measure PI at every epoch -------------------------
model = mlp(NUM_CLASSES, dataset.x_profiling.shape[1])

pi_prof, pi_attack = [], []

for epoch in range(1, N_EPOCHS + 1):
    ckpt = MODELS_DIR / f"{MODEL_NAME}_{epoch:02d}.weights.h5"
    model.load_weights(str(ckpt))
    pi_prof.append(information(model.predict(dataset.x_profiling[:WITHIN_EPOCH_PI_SAMPLE]),
                               dataset.profiling_labels[:WITHIN_EPOCH_PI_SAMPLE],
                               NUM_CLASSES))
    pi_attack.append(information(model.predict(dataset.x_attack),
                                 dataset.attack_labels,
                                 NUM_CLASSES))
    gc.collect()
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}: PI(train)={pi_prof[-1]:.3f}  PI(attack)={pi_attack[-1]:.3f}")

# %%
fig, ax = plt.subplots(figsize=(10, 5))
epochs = np.arange(1, N_EPOCHS + 1)
ax.plot(epochs, pi_prof, label="profiling (train)")
ax.plot(epochs, pi_attack, label="attack (test)")
ax.set_xlabel("training epoch")
ax.set_ylabel("Perceived Information (bits)")
ax.set_title("MLP × ASCADr: when does the key start leaking?")
ax.set_yscale("log")
ax.grid(alpha=0.3)
ax.legend()

out = OUTPUTS_DIR / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "05_pi_curves.png", dpi=150)
print("saved:", out / "05_pi_curves.png")

# %%