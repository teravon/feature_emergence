"""PI and P(true unmasked y) vs epoch on ASCADr; one-trace early vs late.

Writes:
  outputs/figures/learning_pi_and_ptrue.png
  outputs/figures/learning_one_trace_early_late.png
"""
import gc

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from feature_emergence.config import MODELS_DIR, OUTPUTS_DIR
from feature_emergence.profiling_and_attack import information, mlp
from feature_emergence.utils import load_dataset, scale_dataset

OUT = OUTPUTS_DIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)

EPOCHS = [1, 5, 10, 20, 40, 60, 80, 100]
N_PROF_PI = 10000
TRACE = 0


def main():
    dataset = load_dataset(
        "ascad-variable", traces_dim=2000, leakage_model="ID", n_prof=20000
    )
    dataset.x_profiling, dataset.x_attack = scale_dataset(
        dataset.x_profiling, dataset.x_attack, StandardScaler()
    )
    model = mlp(256, 2000)
    labels_a = np.asarray(dataset.attack_labels)
    labels_p = np.asarray(dataset.profiling_labels[:N_PROF_PI])

    pi_prof, pi_atk = [], []
    mean_p_true, median_p_true = [], []
    # one-trace story
    true_y = int(labels_a[TRACE])
    p_true_one, p_peak_one, peak_class = [], [], []

    for e in EPOCHS:
        model.load_weights(MODELS_DIR / f"mlp_ascadr_{e:02d}.weights.h5")
        pred_p = model.predict(dataset.x_profiling[:N_PROF_PI], verbose=0)
        pred_a = model.predict(dataset.x_attack, verbose=0)

        pi_prof.append(information(pred_p, labels_p, 256))
        pi_atk.append(information(pred_a, labels_a, 256))

        p_true = pred_a[np.arange(len(labels_a)), labels_a]
        mean_p_true.append(float(p_true.mean()))
        median_p_true.append(float(np.median(p_true)))

        probs = pred_a[TRACE]
        p_true_one.append(float(probs[true_y]))
        peak = int(probs.argmax())
        peak_class.append(peak)
        p_peak_one.append(float(probs[peak]))

        print(
            f"epoch {e:3d}: PI_atk={pi_atk[-1]:.3f}  "
            f"mean P(true y)={mean_p_true[-1]:.5f}  "
            f"trace0 P(true)={p_true_one[-1]:.2e} peak={peak}"
        )
        gc.collect()

    # --- figure 1: PI and mean P(true) vs epoch
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(EPOCHS, pi_prof, marker="o", ms=4, label="profiling")
    axes[0].plot(EPOCHS, pi_atk, marker="o", ms=4, label="attack")
    axes[0].set_xlabel("training epoch")
    axes[0].set_ylabel("PI vs unmasked y (bits)")
    axes[0].set_title("ASCADr MLP: PI against unmasked identity label")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(EPOCHS, mean_p_true, marker="o", ms=4, color="#4c78a8")
    axes[1].axhline(1 / 256, color="grey", ls=":", label="uniform 1/256")
    axes[1].set_xlabel("training epoch")
    axes[1].set_ylabel("mean P(true unmasked y)")
    axes[1].set_title("Attack set: average probability on the true y")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT / "learning_pi_and_ptrue.png", dpi=150)
    print("saved:", OUT / "learning_pi_and_ptrue.png")

    # --- figure 2: one attack trace, early vs late softmax (true y marked)
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, e, title in zip(
        axes2,
        [1, 100],
        ["epoch 1", "epoch 100"],
    ):
        model.load_weights(MODELS_DIR / f"mlp_ascadr_{e:02d}.weights.h5")
        probs = model.predict(dataset.x_attack[TRACE : TRACE + 1], verbose=0)[0]
        ax.bar(np.arange(256), probs, color="#9ecae1", width=1.0)
        ax.axvline(true_y, color="#d62728", lw=1.5, label=f"true y = {true_y}")
        peak = int(probs.argmax())
        ax.annotate(
            f"peak {peak}\nP={probs[peak]:.3f}",
            xy=(peak, probs[peak]),
            xytext=(peak + 20, max(probs) * 0.85),
            arrowprops=dict(arrowstyle="->", color="0.3"),
            fontsize=9,
        )
        ax.set_ylabel("P(y)")
        ax.set_title(
            f"Same attack trace 0 — {title}  "
            f"(P(true y)={probs[true_y]:.2e})"
        )
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3, axis="y")
    axes2[1].set_xlabel("S-box output class y")
    fig2.suptitle(
        "Unmasked true y vs model peak — learning changes the softmax, not always toward true y",
        y=1.02,
    )
    fig2.tight_layout()
    fig2.savefig(OUT / "learning_one_trace_early_late.png", dpi=150, bbox_inches="tight")
    print("saved:", OUT / "learning_one_trace_early_late.png")

    # print table for the markdown
    print("\nepoch | PI_attack | mean_P(true) | trace0_P(true) | trace0_peak")
    for i, e in enumerate(EPOCHS):
        print(
            f"{e:5d} | {pi_atk[i]:9.3f} | {mean_p_true[i]:12.5f} | "
            f"{p_true_one[i]:14.2e} | {peak_class[i]}"
        )


if __name__ == "__main__":
    main()
