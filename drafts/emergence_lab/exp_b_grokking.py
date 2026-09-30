"""Experiment B — single emergence (grokking): (a + b) mod p.

A tiny MLP (one-hot a,b -> 128 tanh -> softmax over p classes) learns
(a + b) mod 32 from half of the 1024 pairs. With weight decay, training
shows the grokking pattern: a long plateau where test accuracy stays at
chance (the network memorizes the training pairs) followed by a sudden
jump to generalization. We log train/test accuracy and Perceived
Information (same formula as the repository) every LOG_EVERY steps.

Sweep: seeds x weight decays (grid below). Per-run logs land in
results/exp_b/*.npz; the run with the cleanest plateau-then-jump is
rendered into:
  grok_curves.png   accuracy and PI vs training step
  grok_pca.png      PCA of test hidden activations, plateau vs after the jump
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
RUNS = os.path.join(OUT, "exp_b")
os.makedirs(RUNS, exist_ok=True)

SMOKE = os.environ.get("SMOKE") == "1"

P = 32
HIDDEN = 128
LR = 1e-3
STEPS = 3000 if SMOKE else 120000
LOG_EVERY = 100 if SMOKE else 250
TRAIN_FRAC = 0.5
SNAP_STEPS = [100, 500, 1000, 2000, 5000, 10000, 20000, 40000, 60000, 80000, 120000]

SEEDS = [0] if SMOKE else [0, 1, 2, 3]
WDS = [0.1] if SMOKE else [0.3, 0.5, 1.0, 2.0]


def run_one(seed, wd, Xtr, ytr, Xte, yte):
    net = MLP(Xtr.shape[1], HIDDEN, P, seed=seed)
    opt = Adam(net.params(), lr=LR)
    log = {"step": [], "loss": [], "acc_tr": [], "acc_te": [], "pi_tr": [], "pi_te": []}
    snaps = {}
    best_streak = 0
    for step in range(1, STEPS + 1):
        loss, grads = net.loss_grads(Xtr, ytr)
        opt.step(net.params(), grads, decay=wd)
        if step in SNAP_STEPS:
            snaps[f"hidden_{step}"] = net.hidden(Xte)
        if step % LOG_EVERY == 0:
            p_tr = net.predict_proba(Xtr)
            p_te = net.predict_proba(Xte)
            log["step"].append(step)
            log["loss"].append(loss)
            log["acc_tr"].append(float((p_tr.argmax(1) == ytr).mean()))
            log["acc_te"].append(float((p_te.argmax(1) == yte).mean()))
            log["pi_tr"].append(pi_metric(p_tr, ytr, P))
            log["pi_te"].append(pi_metric(p_te, yte, P))
            best_streak = best_streak + 1 if log["acc_te"][-1] >= 0.98 else 0
            if best_streak >= 5:  # fully grokked, stop early
                break
    return net, {k: np.array(v) for k, v in log.items()}, snaps


def grok_step(log, thr=0.9):
    """First logged step at which test accuracy reaches thr; None if never."""
    hit = np.where(log["acc_te"] >= thr)[0]
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
            net, log, snaps = run_one(seed, wd, Xtr, ytr, Xte, yte)
            tag = f"seed{seed}_wd{wd:g}"
            np.savez_compressed(
                os.path.join(RUNS, f"{tag}.npz"),
                **log,
                yte=yte,
                **snaps,
            )
            g80 = grok_step(log, 0.8)
            t50 = grok_step(log, 0.5)
            results.append({
                "tag": tag, "seed": seed, "wd": wd,
                "final_acc_te": float(log["acc_te"][-1]),
                "final_pi_te": float(log["pi_te"][-1]),
                "t50": t50,
                "grok_step_08": g80,
                "steps_run": int(log["step"][-1]),
            })
            print(f"B {tag}: final acc_te={log['acc_te'][-1]:.3f} "
                  f"pi_te={log['pi_te'][-1]:.2f} t50={t50} grok@0.8={g80}", flush=True)

    with open(os.path.join(RUNS, "runs.json"), "w") as f:
        json.dump(results, f, indent=2)

    # pick the cleanest grokking run: generalized (final acc >= 0.7) with the
    # latest 0.5-crossing, i.e. the longest plateau before the jump
    grokked = [r for r in results if r["final_acc_te"] >= 0.7 and r["t50"]]
    if not grokked:
        print("B: NO RUN GROKKED — check runs.json", flush=True)
        return
    best = max(grokked, key=lambda r: r["t50"])
    d = np.load(os.path.join(RUNS, f"{best['tag']}.npz"))
    print(f"B best run: {best['tag']} (t50 = {best['t50']})", flush=True)

    steps, acc_tr, acc_te = d["step"], d["acc_tr"], d["acc_te"]
    pi_tr, pi_te = d["pi_tr"], d["pi_te"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(steps, acc_tr, label="train")
    axes[0].plot(steps, acc_te, label="test (unseen pairs)")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1.02)
    axes[0].axhline(1 / P, color="grey", ls=":", lw=1, label="chance")
    axes[0].set_ylabel("accuracy")
    axes[0].set_xlabel("training step")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(steps, pi_tr, label="train")
    axes[1].plot(steps, pi_te, label="test")
    axes[1].set_xscale("log")
    axes[1].set_ylabel("Perceived Information (bits)")
    axes[1].set_xlabel("training step")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Grokking (a+b) mod {P}: plateau, then sudden generalization "
                 f"({best['tag']})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "grok_curves.png"), dpi=150)

    # PCA: last snapshot before the grok point vs final
    snap_names = sorted(
        (k for k in d.files if k.startswith("hidden_")),
        key=lambda k: int(k.split("_")[1]),
    )
    before = [k for k in snap_names if int(k.split("_")[1]) < best["t50"]]
    key_before = before[-1] if before else snap_names[0]
    key_after = snap_names[-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, title in [
        (axes[0], key_before, f"during the plateau (step {key_before.split('_')[1]})"),
        (axes[1], key_after, f"after the jump (step {key_after.split('_')[1]})"),
    ]:
        proj = pca2(d[key])
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=d["yte"], cmap="hsv", s=8, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"What emerged inside: test activations colored by (a+b) mod {P}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "grok_pca.png"), dpi=150)

    with open(os.path.join(OUT, "SUMMARY.md"), "a") as f:
        f.write("\n## Experiment B (grokking, single jump)\n\n")
        f.write("| run | final acc_te | final PI_te | t50 | grok@0.8 |\n|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['tag']} | {r['final_acc_te']:.3f} | "
                    f"{r['final_pi_te']:.2f} | {r['t50']} | {r['grok_step_08']} |\n")
        f.write(f"\nBest run: **{best['tag']}**\n")


if __name__ == "__main__":
    main()
