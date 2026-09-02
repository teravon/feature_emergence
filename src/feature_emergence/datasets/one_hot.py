"""Dependency-free one-hot encoding.

Drop-in replacement for `tensorflow.keras.utils.to_categorical` so that data
loading and exploration do not require a TensorFlow installation. TensorFlow
is only needed for model training/evaluation (`profiling_and_attack.py`).
"""

import numpy as np


def to_categorical(y, num_classes):
    """One-hot encode integer labels, matching the Keras behavior and dtype."""
    y = np.asarray(y, dtype=np.int64).ravel()
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out
