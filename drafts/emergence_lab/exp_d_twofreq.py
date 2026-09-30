"""Experiment D — double emergence from a sum of two functions (spectral bias).

The task is honestly two functions added together:

    y = sin(2 pi x) + 0.5 * sin(2 pi * 8 x)
        ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^
        slow wave       fast ripple

Neural networks exhibit spectral bias: they fit the low-frequency component
first and the high-frequency one much later. So training should show two
emergences: the loss drops when the slow wave is captured, plateaus while
the residual is pure ripple, then drops again when the ripple is captured.

Nothing is simulated: it is one real function, learned from samples, and the
"two booms" are measured as the correlation of the network output with each
hidden component versus training step.

Figures (results/):
  twofreq_fit_stages.png   the fit at three stages (slow only / plateau / full)
  twofreq_components.png   correlation with each component + MSE vs step
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import Adam

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

SMOKE = os.environ.get("SMOKE") == "1"

N_POINTS = 512
HIDDEN = 256
STEPS = 4000 if SMOKE else 60000
LOG_EVERY = 250 if not SMOKE else 100
LR = 1e-3
SEED = 0


def f_slow(x):
    return np.sin(2 * np.pi * x)


def f_fast(x):
    return 0.5 * np.sin(2 * np.pi * 8 * x)


class TinyRegressor:
    """1 -> H tanh -> H tanh -> 1 linear, MSE."""

    def __init__(self, n_hidden, seed):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1.0, (1, n_hidden))
        self.b1 = rng.normal(0, 0.5, n_hidden)
        self.W2 = rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, n_hidden))
        self.b2 = np.zeros(n_hidden)
        self.W3 = rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, 1))
        self.b3 = np.zeros(1)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x):
        a1 = np.tanh(x @ self.W1 + self.b1)
        a2 = np.tanh(a1 @ self.W2 + self.b2)
        return a2 @ self.W3 + self.b3, (x, a1, a2)

    def loss_grads(self, x, y):
        n = len(x)
        pred, (xc, a1, a2) = self.forward(x)
        err = pred - y
        loss = float(np.mean(err**2))
        d = 2 * err / n
        dW3 = a2.T @ d
        db3 = d.sum(axis=0)
        da2 = (d @ self.W3.T) * (1 - a2**2)
        dW2 = a1.T @ da2
        db2 = da2.sum(axis=0)
        da1 = (da2 @ self.W2.T) * (1 - a1**2)
        dW1 = xc.T @ da1
        db1 = da1.sum(axis=0)
        return loss, [dW1, db1, dW2, db2, dW3, db3]

    def predict(self, x):
        return self.forward(x)[0]


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0, 1, N_POINTS)[:, None]
    y = f_slow(x) + f_fast(x)
    grid = np.linspace(0, 1, 1200)[:, None]
    y_slow_grid, y_fast_grid = f_slow(grid), f_fast(grid)
    y_grid = y_slow_grid + y_fast_grid

    net = TinyRegressor(HIDDEN, SEED)
    opt = Adam(net.params(), lr=LR)

    steps_seen, mses, c_slow, c_fast = [], [], [], []
    fits = {}
    snap_at = set()
    # stages picked adaptively: record predictions whenever corr with the slow
    # component first crosses 0.9, and when corr with the fast one does
    stage_pred = {}
    for step in range(1, STEPS + 1):
        loss, grads = net.loss_grads(x, y)
        opt.step(net.params(), grads)
        if step % LOG_EVERY == 0 or step == 1:
            pred = net.predict(grid)
            cs, cf = corr(pred, y_slow_grid), corr(pred, y_fast_grid)
            steps_seen.append(step)
            mses.append(float(np.mean((net.predict(x) - y) ** 2)))
            c_slow.append(cs)
            c_fast.append(cf)
            if cs >= 0.9 and "slow" not in stage_pred:
                stage_pred["slow"] = (step, pred.copy())
            if cs >= 0.9 and cf < 0.5:
                stage_pred.setdefault("plateau", (step, pred.copy()))
            if cf >= 0.9 and "fast" not in stage_pred:
                stage_pred["fast"] = (step, pred.copy())

    steps_seen = np.array(steps_seen)
    c_slow, c_fast = np.array(c_slow), np.array(c_fast)
    np.savez_compressed(
        os.path.join(OUT, "exp_d_twofreq.npz"),
        step=steps_seen, mse=mses, c_slow=c_slow, c_fast=c_fast,
    )

    # --- figure 1: the fit at the stages we actually observed
    order = [k for k in ["slow", "plateau", "fast"] if k in stage_pred]
    fig, axes = plt.subplots(1, len(order), figsize=(4.2 * len(order), 3.8),
                             sharey=True)
    if len(order) == 1:
        axes = [axes]
    titles = {"slow": "slow wave captured",
              "plateau": "plateau: ripple still missing",
              "fast": "both components learned"}
    for ax, key in zip(axes, order):
        step, pred = stage_pred[key]
        ax.plot(grid, y_grid, color="grey", lw=1, alpha=0.6, label="target")
        ax.plot(grid, y_slow_grid, color="tab:blue", lw=1, ls="--",
                alpha=0.7, label="slow component")
        ax.plot(grid, pred, "k-", lw=1.4, label="network")
        ax.set_title(f"{titles[key]}\n(step {step})", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("y = sin(2 pi x) + 0.5 sin(16 pi x): the network learns the "
                 "slow part first")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "twofreq_fit_stages.png"), dpi=150)

    # --- figure 2: two emergences, measured
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(steps_seen, c_slow, label="corr with slow component")
    axes[0].plot(steps_seen, c_fast, label="corr with fast component")
    axes[0].set_xscale("log")
    axes[0].set_ylim(-0.05, 1.02)
    axes[0].set_ylabel("correlation with the hidden component")
    axes[0].set_xlabel("training step")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(steps_seen, mses)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("MSE")
    axes[1].set_xlabel("training step")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Double emergence: the coarse structure emerges first, "
                 "the fine one much later")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "twofreq_components.png"), dpi=150)

    def cross(c, thr=0.9):
        hit = np.where(c >= thr)[0]
        return int(steps_seen[hit[0]]) if len(hit) else None

    print(f"D done: corr(slow)>=0.9 at {cross(c_slow)}, "
          f"corr(fast)>=0.9 at {cross(c_fast)}, final MSE {mses[-1]:.2e}")
    with open(os.path.join(OUT, "SUMMARY.md"), "a") as f:
        f.write("\n## Experiment D (two-frequency sum)\n\n"
                f"- corr with slow component >= 0.9 at step {cross(c_slow)}\n"
                f"- corr with fast component >= 0.9 at step {cross(c_fast)}\n"
                f"- final MSE {mses[-1]:.3e}\n")


if __name__ == "__main__":
    main()
