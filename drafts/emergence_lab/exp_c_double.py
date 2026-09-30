"""Experiment C — double emergence: one network, two booms.

Same grokking setup as experiment B but with (a + b) mod 16, and Perceived
Information computed against three readings of the same label:

  - parity        y mod 2   (coarse structure, 2 classes)
  - quarters      y mod 4   (4 classes)
  - full value    y mod 16  (fine structure, 16 classes)

Hypothesis (the toy version of the paper's CHES_CTF result, where the model
first separates high/low Hamming weights and later even/odd ones): the
coarse PI jumps first, the fine PI later — two emergences in one network.

This is a hypothesis, not a guarantee; the sweep over seeds x weight decays
exists to find whether any configuration separates the two jumps in time.
Per-run logs land in results/exp_c/*.npz; the run with the largest
separation is rendered into:
  double_curves.png   PI and accuracy per label granularity vs step
  double_pca.png      activations before jump 1, between jumps, after jump 2
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import Adam, MLP, onehot_pairs, pca2, pi_metric

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
RUNS = os.path.join(OUT, "exp_c")
os.makedirs(RUNS, exist_ok=True)

SMOKE = os.environ.get("SMOKE") == "1"

P = 16
HIDDEN = 128
LR = 1e-3
STEPS = 3000 if SMOKE else 120000
LOG_EVERY = 100 if SMOKE else 250
TRAIN_FRAC = 0.5
SNAP_STEPS = [100, 500, 1000, 2000, 5000, 10000, 20000, 40000, 60000, 80000, 120000]
GRANS = [2, 4, 16]

SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4, 5]
WDS = [0.1] if SMOKE else [0.3, 0.5, 1.0, 2.0]


def marginalize(probs, num_classes, g):
    """Coarsen a softmax over num_classes to one over label % g."""
    out = np.zeros((len(probs), g))
    for k in range(num_classes):
        out[:, k % g] += probs[:, k]
    return out


def run_one(seed, wd, Xtr, ytr, Xte, yte):
    net = MLP(Xtr.shape[1], HIDDEN, P, seed=seed)
    opt = Adam(net.params(), lr=LR)
    log = {"step": [], "loss": [], "acc_tr": []}
    for g in GRANS:
        log[f"acc_te_g{g}"] = []
        log[f"pi_te_g{g}"] = []
    snaps = {}
    best_streak = 0
    for step in range(1, STEPS + 1):
        loss, grads = net.loss_grads(Xtr, ytr)
        opt.step(net.params(), grads, decay=wd)
        if step in SNAP_STEPS:
            snaps[f"hidden_{step}"] = net.hidden(Xte)
        if step % LOG_EVERY == 0:
            p_te = net.predict_proba(Xte)
            log["step"].append(step)
            log["loss"].append(loss)
            log["acc_tr"].append(float((net.predict_proba(Xtr).argmax(1) == ytr).mean()))
            for g in GRANS:
                pg = marginalize(p_te, P, g)
                log[f"acc_te_g{g}"].append(float((pg.argmax(1) == yte % g).mean()))
                log[f"pi_te_g{g}"].append(pi_metric(pg, yte % g, g))
            best_streak = best_streak + 1 if log["acc_te_g16"][-1] >= 0.98 else 0
            if best_streak >= 5:
                break
    return {k: np.array(v) for k, v in log.items()}, snaps


def half_max_step(log, key):
    """First logged step where the metric reaches half of its run maximum."""
    v = log[key]
    peak = v.max()
    if peak <= 0:
        return None
    hit = np.where(v >= 0.5 * peak)[0]
    return int(log["step"][hit[0]]) if len(hit) else None


def main():
    X, y = onehot_pairs(P)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    n_tr = int(TRAIN_FRAC * len(X))
    Xtr, ytr = X[perm[:n_tr]], y[perm[:n_tr]]
    Xte, yte = X[perm[n_tr:]], y[perm[n_tr:]]

    results = []
    for seed in SEEDS:
        for wd in WDS:
            log, snaps = run_one(seed, wd, Xtr, ytr, Xte, yte)
            tag = f"seed{seed}_wd{wd:g}"
            np.savez_compressed(os.path.join(RUNS, f"{tag}.npz"), **log, yte=yte, **snaps)
            t2 = half_max_step(log, "pi_te_g2")
            t16 = half_max_step(log, "pi_te_g16")
            sep = (t16 - t2) if (t2 and t16) else None
            results.append({
                "tag": tag, "seed": seed, "wd": wd,
                "final_acc_full": float(log["acc_te_g16"][-1]),
                "final_pi_full": float(log["pi_te_g16"][-1]),
                "t_half_pi_g2": t2, "t_half_pi_g16": t16, "separation": sep,
            })
            print(f"C {tag}: acc_full={log['acc_te_g16'][-1]:.3f} "
                  f"t_half(g2)={t2} t_half(g16)={t16} sep={sep}", flush=True)

    with open(os.path.join(RUNS, "runs.json"), "w") as f:
        json.dump(results, f, indent=2)

    candidates = [r for r in results if r["final_acc_full"] >= 0.7 and r["separation"]]
    if not candidates:
        print("C: NO RUN WITH A CLEAN DOUBLE JUMP — check runs.json", flush=True)
        with open(os.path.join(OUT, "SUMMARY.md"), "a") as f:
            f.write("\n## Experiment C (double emergence)\n\n"
                    "No run showed a clean double jump. See exp_c/runs.json.\n")
        return
    best = max(candidates, key=lambda r: r["separation"])
    d = np.load(os.path.join(RUNS, f"{best['tag']}.npz"))
    print(f"C best run: {best['tag']} (separation {best['separation']} steps)", flush=True)

    steps = d["step"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for g, color in zip(GRANS, ["tab:green", "tab:orange", "tab:blue"]):
        axes[0].plot(steps, d[f"pi_te_g{g}"], color=color,
                     label=f"PI vs y mod {g}")
        axes[1].plot(steps, d[f"acc_te_g{g}"], color=color,
                     label=f"accuracy vs y mod {g}")
    axes[0].set_xscale("log"); axes[1].set_xscale("log")
    axes[0].set_ylabel("Perceived Information (bits)")
    axes[1].set_ylabel("accuracy"); axes[1].set_ylim(0, 1.02)
    for ax in axes:
        ax.set_xlabel("training step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Double emergence in one network: coarse structure first, "
                 f"fine structure later ({best['tag']})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "double_curves.png"), dpi=150)

    # snapshots: plateau, between the two jumps, final
    snap_names = sorted(
        (k for k in d.files if k.startswith("hidden_")),
        key=lambda k: int(k.split("_")[1]),
    )
    snap_steps = [int(k.split("_")[1]) for k in snap_names]
    t2, t16 = best["t_half_pi_g2"], best["t_half_pi_g16"]

    def nearest(target, candidates_steps, names):
        i = int(np.argmin([abs(s - target) for s in candidates_steps]))
        return names[i]

    keys = [
        nearest(t2 / 2, snap_steps, snap_names),
        nearest(np.sqrt(t2 * t16), snap_steps, snap_names),
        snap_names[-1],
    ]
    titles = ["during the plateau", "between the two jumps", "after both jumps"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for col, (key, title) in enumerate(zip(keys, titles)):
        proj = pca2(d[key])
        step = key.split("_")[1]
        axes[0, col].scatter(proj[:, 0], proj[:, 1], c=d["yte"] % 2,
                             cmap="coolwarm", s=10, alpha=0.7)
        axes[0, col].set_title(f"{title} (step {step})\ncolored by parity", fontsize=9)
        axes[1, col].scatter(proj[:, 0], proj[:, 1], c=d["yte"],
                             cmap="hsv", s=10, alpha=0.7)
        axes[1, col].set_title("colored by full value mod 16", fontsize=9)
        for row in range(2):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    fig.suptitle("What emerged inside, step by step")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "double_pca.png"), dpi=150)

    with open(os.path.join(OUT, "SUMMARY.md"), "a") as f:
        f.write("\n## Experiment C (double emergence)\n\n")
        f.write("| run | final acc full | t_half PI g2 | t_half PI g16 | separation |\n"
                "|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['tag']} | {r['final_acc_full']:.3f} | "
                    f"{r['t_half_pi_g2']} | {r['t_half_pi_g16']} | {r['separation']} |\n")
        f.write(f"\nBest run: **{best['tag']}** (separation "
                f"{best['separation']} steps)\n")


if __name__ == "__main__":
    main()
