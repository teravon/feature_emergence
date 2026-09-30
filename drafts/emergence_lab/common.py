"""Shared pieces for the emergence lab.

A tiny numpy MLP (one hidden tanh layer + softmax), an Adam optimizer, and
Perceived Information in bits. The PI formula is the same one used by
src/feature_emergence/profiling_and_attack.py:information in the repository,
reimplemented here without TensorFlow so the lab runs in the light .venv:

    PI = H(K) + sum_k p(k) * mean_{i: y_i = k} log2 p_hat(k | x_i)

PI is 0 at chance, H(K) at perfect prediction, and negative when the model
does worse than chance on the chosen labelling.
"""
import numpy as np


def pi_metric(probs, labels, num_classes):
    """Perceived Information (bits) of predicted probs against labels."""
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64) + 1e-36
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    p_k = counts / counts.sum()
    nz = p_k > 0
    acc = float(-np.sum(p_k[nz] * np.log2(p_k[nz])))  # H(K)
    for k in range(num_classes):
        idx = labels == k
        if idx.any():
            acc += p_k[k] * float(np.log2(probs[idx, k]).mean())
    return acc


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads, decay=0.0, decay_idx=(0, 2)):
        """Adam update, then AdamW-style decoupled weight decay.

        decay applies p *= (1 - lr * decay) to params[i] for i in decay_idx
        (by default the two weight matrices, not the biases). Coupling L2
        into the loss does not work here: Adam normalizes the penalty away.
        """
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1**self.t)
            vh = self.v[i] / (1 - self.b2**self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
            if decay and i in decay_idx:
                p *= 1.0 - self.lr * decay


class MLP:
    """One hidden tanh layer, softmax output. Pure numpy classifier."""

    def __init__(self, n_in, n_hidden, n_out, seed=0, init_scale=1.0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, init_scale / np.sqrt(n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, init_scale / np.sqrt(n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        a1 = np.tanh(X @ self.W1 + self.b1)
        logits = a1 @ self.W2 + self.b2
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True), (X, a1)

    def loss_grads(self, X, y):
        """Mean cross-entropy; returns (loss, grads). Weight decay is handled
        by the optimizer (decoupled, AdamW-style), not here."""
        n = len(X)
        probs, (Xc, a1) = self.forward(X)
        ce = -np.log(probs[np.arange(n), y] + 1e-36).mean()
        dlogits = probs
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        dW2 = a1.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dz1 = (dlogits @ self.W2.T) * (1 - a1**2)
        dW1 = Xc.T @ dz1
        db1 = dz1.sum(axis=0)
        return ce, [dW1, db1, dW2, db2]

    def hidden(self, X):
        return np.tanh(X @ self.W1 + self.b1)

    def predict_proba(self, X):
        return self.forward(X)[0]


def pca2(A):
    """Project rows of A onto their first two principal components."""
    A = A - A.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    return A @ vt[:2].T


def onehot_pairs(p):
    """All (a, b) pairs with a, b in 0..p-1, one-hot encoded (concatenated).

    Labels are (a + b) mod p.
    """
    pairs = [(i, j) for i in range(p) for j in range(p)]
    X = np.zeros((p * p, 2 * p))
    for row, (i, j) in enumerate(pairs):
        X[row, i] = 1.0
        X[row, p + j] = 1.0
    y = np.array([(i + j) % p for i, j in pairs], dtype=np.int64)
    return X, y
