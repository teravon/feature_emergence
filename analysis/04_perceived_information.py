# %% [markdown]
# # Perceived Information vs training epoch
#
# Companion to Chapter 8 (`docs/08_perceived_information.md`). For each
# checkpoint of the ASCADr MLP, measure Perceived Information (bits) on a
# profiling subset and on the attack set. No retraining.
#
# Figure: `outputs/figures/08_pi_curves.png`.

# %%
import gc

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from feature_emergence.config import MODELS_DIR, OUTPUTS_DIR
from feature_emergence.profiling_and_attack import information, mlp
from feature_emergence.utils import load_dataset, scale_dataset

out = OUTPUTS_DIR / "figures"
out.mkdir(parents=True, exist_ok=True)

# %%
N_EPOCHS = 100
# Evaluate every EPOCH_STEP epochs (and always epoch 1 and N_EPOCHS).
EPOCH_STEP = 5
MODEL_NAME = "mlp_ascadr"
NUM_CLASSES = 256
WITHIN_EPOCH_PI_SAMPLE = 10000

# %%
dataset = load_dataset("ascad-variable", traces_dim=2000, leakage_model="ID", n_prof=20000)
dataset.x_profiling, dataset.x_attack = scale_dataset(
    dataset.x_profiling, dataset.x_attack, StandardScaler()
)
print(f"profiling: {dataset.x_profiling.shape}  attack: {dataset.x_attack.shape}")

# %%
model = mlp(NUM_CLASSES, dataset.x_profiling.shape[1])

epochs = sorted(
    set([1, N_EPOCHS] + list(range(EPOCH_STEP, N_EPOCHS + 1, EPOCH_STEP)))
)
pi_prof, pi_attack = [], []

for epoch in epochs:
    ckpt = MODELS_DIR / f"{MODEL_NAME}_{epoch:02d}.weights.h5"
    model.load_weights(str(ckpt))
    pred_prof = model.predict(
        dataset.x_profiling[:WITHIN_EPOCH_PI_SAMPLE], verbose=0
    )
    pred_attack = model.predict(dataset.x_attack, verbose=0)
    pi_prof.append(
        information(
            pred_prof,
            dataset.profiling_labels[:WITHIN_EPOCH_PI_SAMPLE],
            NUM_CLASSES,
        )
    )
    pi_attack.append(
        information(pred_attack, dataset.attack_labels, NUM_CLASSES)
    )
    print(
        f"epoch {epoch:3d}: PI(train)={pi_prof[-1]:.4f}  "
        f"PI(attack)={pi_attack[-1]:.4f}"
    )
    gc.collect()

# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, pi_prof, marker="o", ms=3, label="profiling (train)")
ax.plot(epochs, pi_attack, marker="o", ms=3, label="attack (test)")
ax.set_xlabel("training epoch")
ax.set_ylabel("Perceived Information (bits)")
ax.set_title("MLP × ASCADr: PI of softmax outputs vs training epoch")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(out / "08_pi_curves.png", dpi=150)
print("saved:", out / "08_pi_curves.png")
plt.show()
