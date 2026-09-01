# Chapter 2 — The datasets

We work with three power-trace datasets, each from a different AES
implementation. Together they cover the two most common real-world scenarios:
**software** and **hardware** AES, with and without a **masking**
countermeasure.

A quick summary (full structural details are in
[`data/README.md`](../data/README.md)):

| Dataset | Implementation | Masking | Traces (profiling / attack) | Samples per trace |
|---------|----------------|---------|------------------------------|-------------------|
| **ASCADr** | software (ATMega8515) | yes | 200,000 / 10,000 | 2,000 |
| **ESHARD** | hardware (FPGA) | yes | 90,000 / 10,000 | 1,400 |
| **CHES_CTF** | software | no | 30,000 / 10,000 | 15,000 |

## Why three different datasets?

Side-channel measurements are extremely implementation-specific. A model that
works on one chip may completely fail on another just because the *leakage
profile* differs — the "shape" and position of the useful signal in the trace
changes between hardware and software, between devices, and with the presence
or absence of countermeasures.

By testing on three genuinely different datasets, the original study makes its
conclusions more credible than any single-dataset experiment could.

## What is inside each file

All datasets share the same HDF5 structure. For example `eshard.h5`:

```text
Profiling_traces/
  traces      (90000, 1400)  float32   ← power measurements
  metadata    (90000,)                 ← records with the crypto values
Attack_traces/
  traces      (10000, 1400)  float32
  metadata    (10000,)
```

The `metadata` array is your ground truth. Each record contains:

| Field | Meaning |
|-------|---------|
| `plaintext` | the 16 input bytes fed to AES |
| `key` | the 16 key bytes used for the encryption |
| `masks` | random masks (only in the masked datasets) |
| `ciphertext` | the 16 output bytes (CHES_CTF only) |

With `plaintext` and `key` we can compute the leakage label for every trace:

```python
label = sbox[plaintext[byte] ^ key[byte]]        # identity model
hw    = popcount(sbox[plaintext[byte] ^ key[byte]])  # Hamming weight model
```

## Profiling vs attack: a critical split

Every dataset is divided into two disjoint sets:

- **Profiling set**: traces where *everything* is known (plaintext, key,
  masks). Used to **train** the model — the deep learning equivalent of
  "profiling the device".
- **Attack set**: traces where the key is *known to the evaluator* but treated
  as *unknown to the attacker*. Used to **test** whether the trained model can
  actually recover the key from fresh measurements.

This mirrors real life: an attacker first trains a model on a device they own,
then uses that same model against a target device of the same type.

![Example profiling power traces per dataset](../outputs/figures/03_traces_overview.png)

## How to load a dataset

The package provides ready-made loaders. With the pip-installed package:

```python
from feature_emergence.utils import load_dataset

ds = load_dataset("eshard", traces_dim=1400, leakage_model="HW")
ds.x_profiling      # (90000, 1400) power traces
ds.profiling_labels # (90000,) leakage labels
```

In [Chapter 3](03_exploring_the_data.md) we look inside these arrays and start
visualizing them.

Next: [Chapter 3 — Exploring the data](03_exploring_the_data.md).