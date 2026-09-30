"""Experiment A — learning a basic function, gradually.

A minimal network (1 -> 16 tanh -> 1, pure numpy) learns y = sin(2 pi x)
from 70% of the points; the remaining 30% are never seen during training.
Generalization is visible directly: the predicted curve stays smooth where
no training data exists. We also watch each hidden neuron's activation as a
function of x: neurons start as random tilts and become local detectors.

Figures (results/):
  sine_fit_epochs.png   fit at four training steps, train vs test points
  sine_features.png     hidden-neuron activation curves, early vs late
  sine_loss.png         train/test MSE vs step
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

N_POINTS = 400
N_HIDDEN = 16
STEPS = 2000 if SMOKE else 20000
LR = 3e-3
SNAP = [50, 500, 3000, STEPS] if not SMOKE else [50, 500, 1000, STEPS]
SEED = 0


class TinyRegressor:
    """1 -> H tanh -> 1 linear, MSE."""

    def __init__(self, n_hidden, seed):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1.0, (1, n_hidden))
        self.b1 = rng.normal(0, 0.5, n_hidden)
        self.W2 = rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, 1))
        self.b2 = np.zeros(1)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x):
        a1 = np.tanh(x @ self.W1 + self.b1)
        return a1 @ self.W2 + self.b2, (x, a1)

    def loss_grads(self, x, y):
        n = len(x)
        pred, (xc, a1) = self.forward(x)
        err = pred - y
        loss = float(np.mean(err**2))
        d = 2 * err / n
        dW2 = a1.T @ d
        db2 = d.sum(axis=0)
        dz1 = (d @ self.W2.T) * (1 - a1**2)
        dW1 = xc.T @ dz1
        db1 = dz1.sum(axis=0)
        return loss, [dW1, db1, dW2, db2]

    def predict(self, x):
        return self.forward(x)[0]

    def hidden(self, x):
        return np.tanh(x @ self.W1 + self.b1)


def main():
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0, 1, N_POINTS)[:, None]
    y = np.sin(2 * np.pi * x)
    perm = rng.permutation(N_POINTS)
    tr, te = perm[: int(0.7 * N_POINTS)], perm[int(0.7 * N_POINTS):]

    net = TinyRegressor(N_HIDDEN, SEED)
    opt = Adam(net.params(), lr=LR)
    grid = np.linspace(0, 1, 600)[:, None]

    fits, feats = {}, {}
    losses_tr, losses_te, steps_seen = [], [], []
    for step in range(1, STEPS + 1):
        loss, grads = net.loss_grads(x[tr], y[tr])
        opt.step(net.params(), grads)
        if step in SNAP:
            fits[step] = net.predict(grid).ravel()
            feats[step] = net.hidden(grid)
        if step % 50 == 0:
            steps_seen.append(step)
            losses_tr.append(loss)
            losses_te.append(float(np.mean((net.predict(x[te]) - y[te]) ** 2)))

    # --- figure 1: the fit at four steps
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    for ax, step in zip(axes.ravel(), SNAP):
        ax.scatter(x[tr], y[tr], s=6, c="tab:blue", alpha=0.5, label="train")
        ax.scatter(x[te], y[te], s=6, c="tab:orange", alpha=0.5, label="test (unseen)")
        ax.plot(grid, fits[step], "k-", lw=1.5, label="network")
        mse_te = float(np.mean((net.predict(x[te]) - y[te]) ** 2)) if step == STEPS else None
        ax.set_title(f"step {step}", fontsize=10)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Learning y = sin(2 pi x) from 70% of the points")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sine_fit_epochs.png"), dpi=150)

    # --- figure 2: hidden features, early vs late
    early, late = SNAP[0], SNAP[-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, step in zip(axes, [early, late]):
        ax.plot(grid, feats[step], lw=0.9)
        ax.set_title(f"hidden activations, step {step}", fontsize=10)
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("activation of each of the 16 neurons")
    fig.suptitle("Each neuron becomes a local detector — nobody told it where")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sine_features.png"), dpi=150)

    # --- figure 3: loss
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps_seen, losses_tr, label="train MSE")
    ax.plot(steps_seen, losses_te, label="test MSE (unseen points)")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle("Gradual learning: train and test improve together")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sine_loss.png"), dpi=150)

    print(f"A done: final train MSE {losses_tr[-1]:.2e}, test MSE {losses_te[-1]:.2e}")
    with open(os.path.join(OUT, "SUMMARY.md"), "a") as f:
        f.write(f"\n## Experiment A (sine)\n\n- final train MSE {losses_tr[-1]:.3e}, test MSE {losses_te[-1]:.3e}\n")


if __name__ == "__main__":
    main()
