"""Compare traces-to-GE=1 across the three published MLP targets.

Writes:
  outputs/figures/08_traces_to_ge1.png
  outputs/figures/08_ge_curves_three_targets.png
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from feature_emergence.config import MODELS_DIR, OUTPUTS_DIR
from feature_emergence.profiling_and_attack import guessing_entropy, mlp, mlp_eshard
from feature_emergence.utils import load_dataset, scale_dataset

OUT_DIR = OUTPUTS_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Finished checkpoints as published (see docs/appendix_datasets.md).
CONFIGS = [
    {
        "name": "ASCADr",
        "identifier": "ascad-variable",
        "traces_dim": 2000,
        "leakage_model": "ID",
        "n_prof": 20000,
        "weights": MODELS_DIR / "mlp_ascadr_100.weights.h5",
        "build": lambda c, n: mlp(c, n),
        "note": "ID labels, 256 classes; masked power",
    },
    {
        "name": "ESHARD",
        "identifier": "eshard",
        "traces_dim": 1400,
        "leakage_model": "HW",
        "n_prof": 20000,
        "weights": MODELS_DIR / "mlp_eshard__100.weights.h5",
        "build": lambda c, n: mlp_eshard(c, n),
        "note": "HW labels, 9 classes; masked EM",
    },
    {
        "name": "CHES_CTF",
        "identifier": "ches_ctf",
        "traces_dim": 15000,
        "leakage_model": "HW",
        "n_prof": 20000,
        "weights": MODELS_DIR / "mlp_ches_ctf_200.weights.h5",
        "build": lambda c, n: mlp_eshard(c, n),
        "note": "HW labels, 9 classes; power CTF target",
    },
]

KEY_RANK_ATTACK_TRACES = 4000


def run_one(cfg):
    dataset = load_dataset(
        cfg["identifier"],
        traces_dim=cfg["traces_dim"],
        leakage_model=cfg["leakage_model"],
        n_prof=cfg["n_prof"],
    )
    dataset.x_profiling, dataset.x_attack = scale_dataset(
        dataset.x_profiling, dataset.x_attack, StandardScaler()
    )
    model = cfg["build"](dataset.classes, cfg["traces_dim"])
    model.load_weights(cfg["weights"])
    preds = model.predict(dataset.x_attack, verbose=0)
    H = dataset.labels_key_hypothesis_attack
    true_key = int(dataset.correct_key_attack)
    n_use = min(KEY_RANK_ATTACK_TRACES, preds.shape[0])
    np.random.seed(42)
    final_ge, ge_curve, nt_ge1 = guessing_entropy(
        preds, H, true_key, key_rank_attack_traces=n_use
    )
    return {
        "name": cfg["name"],
        "note": cfg["note"],
        "n_attack": int(preds.shape[0]),
        "final_ge": float(final_ge),
        "nt_ge1": int(nt_ge1),
        "ge_curve": ge_curve,
    }


def main():
    rows = []
    for cfg in CONFIGS:
        print(f"=== {cfg['name']} ===")
        row = run_one(cfg)
        print(
            f"  traces to GE=1: {row['nt_ge1']}  "
            f"final GE@{KEY_RANK_ATTACK_TRACES}: {row['final_ge']:.2f}  "
            f"({row['note']})"
        )
        rows.append(row)

    names = [r["name"] for r in rows]
    vals = [r["nt_ge1"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, vals, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylabel("attack traces to reach GE = 1")
    ax.set_title("Finished published MLPs: how many traces to recover one key byte?")
    ax.grid(alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(v),
            ha="center",
            va="bottom",
            fontsize=11,
        )
    fig.tight_layout()
    out = OUT_DIR / "08_traces_to_ge1.png"
    fig.savefig(out, dpi=150)
    print("saved:", out)

    # Also overlay GE curves (optional companion)
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    for r in rows:
        n = np.arange(1, len(r["ge_curve"]) + 1)
        ax2.plot(n, r["ge_curve"], label=f"{r['name']} (GE=1 at {r['nt_ge1']})")
    ax2.axhline(1, color="#d62728", ls="--", lw=1, label="GE = 1")
    ax2.axhline(128, color="grey", ls=":", lw=1)
    ax2.set_xlabel("attack traces used")
    ax2.set_ylabel("mean rank of true key (GE)")
    ax2.set_yscale("log")
    ax2.set_title("GE vs traces — three published targets")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    out2 = OUT_DIR / "08_ge_curves_three_targets.png"
    fig2.savefig(out2, dpi=150)
    print("saved:", out2)


if __name__ == "__main__":
    main()
