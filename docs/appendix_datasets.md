# Datasets Guide

This directory holds the extracted (pre-processed) power-consumption trace
datasets used in the fork of
[*"You have to be Realistic: On Investigating Feature Emergence in
Deep Learning-based Side-channel Analysis"*](https://github.com/Sengim/feature_emergence).

> All files here are **large binary artifacts and are excluded from git**.
> They are downloaded/extracted with `scripts/download_files.sh` (or
> `download_files.ps1`) from the Zenodo record
> <https://zenodo.org/records/15410792>. Only this README is tracked.

---

## File inventory

| File | Size | Traces (profiling / attack) | Samples per trace | Trace dtype |
|------|------|-----------------------------|-------------------|-------------|
| `eshard.h5` | 538 MB | 90,000 / 10,000 | 1,400 | `float32` |
| `ASCADr/ascad-variable_70k-90k_20.h5` | 481 MB | 200,000 / 10,000 | 2,000 | `int8` |
| `ches_ctf/ches_ctf_nopoi_window_20.h5` | 1.2 GB | 30,000 / 10,000 | 15,000 | `float16` |

All files share the same **HDF5 layout** (two top-level groups):

```
<file>.h5
├── Profiling_traces/
│   ├── traces      (N_profiling × n_samples)
│   └── metadata    (N_profiling × 1)  structured records
└── Attack_traces/
    ├── traces      (N_attack × n_samples)
    └── metadata    (N_attack × 1)     structured records
```

- **`traces`**: a 2-D array. Each row is one power-consumption trace, i.e. a
  time series of `n_samples` instantaneous power measurements (1st AES round
  window). Axis 0 = trace index, axis 1 = time sample.
- **`metadata`**: a structured NumPy array (compound dtype). Each element is a
  record holding the cryptographic values associated with that trace.

---

## 1. `eshard.h5` — ESHARD (masked AES, EM measurements)

**Board/target:** masked AES-128 software implementation on an **ARM
Cortex-M4** (32-bit), measured via **electromagnetic emissions**. This is the
trimmed public version of the ESHARD-AES128 dataset: first-order Boolean
masking, covering the first encryption round without shuffling.

**Structure:**

```
Profiling_traces/
  traces    (90000, 1400)   float32
  metadata  (90000,)        [('plaintext', 'u1', (16,)), ('key', 'u1', (16,)), ('masks', 'u1', (2,))]
Attack_traces/
  traces    (10000, 1400)   float32
  metadata  (10000,)        [('plaintext', 'u1', (16,)), ('key', 'u1', (16,)), ('masks', 'u1', (2,))]
```

**Metadata columns:**

| Field | Type | Description |
|-------|------|-------------|
| `plaintext` | `uint8[16]` | 16 AES plaintext bytes fed to the S-box |
| `key` | `uint8[16]` | 16 AES key bytes (fixed per dataset) |
| `masks` | `uint8[2]` | Random masks used by the masking countermeasure |

**Notes:**
- Labels used in the paper: `SBox[plaintext ^ key]` for the identity (ID)
  model — 256 classes — or its Hamming weight (HW) — 9 classes. Rows are
  labeled with `target_byte`.
- The loader class is `ReadEshard` in
  `src/feature_emergence/datasets/load_eshard.py`.

---

## 2. `ASCADr/ascad-variable_70k-90k_20.h5` — ASCAD variable key (software AES)

**Board/target:** AES-128 **software** implementation on an ATMega8515
running with a **variable key** and a Boolean-masking countermeasure
(the ASCAD "variable key" variant, re-windowed at 2,000 samples).

**Structure:**

```
Profiling_traces/
  traces    (200000, 2000)  int8
  metadata  (200000,)       [('plaintext', '<f8', (16,)), ('key', '<f8', (16,)), ('masks', '<f8', (18,))]
Attack_traces/
  traces    (10000, 2000)   int8
  metadata  (10000,)        [('plaintext', '<f8', (16,)), ('key', '<f8', (16,)), ('masks', '<f8', (18,))]
```

**Metadata columns:**

| Field | Type | Description |
|-------|------|-------------|
| `plaintext` | `float64[16]` | 16 AES plaintext bytes (stored as floats) |
| `key` | `float64[16]` | 16 AES key bytes (stored as floats; fixed key `00112233445566778899AABBCCDDEEFF`) |
| `masks` | `float64[18]` | 18 mask bytes of the masking scheme; indices `16` and `17` hold the extra masks used to build the higher-order shares |

**Notes:**
- The 18 mask bytes feed the share construction in the loader:
  - masks `[0..15]` → per-byte masks for round-1 shares
  - masks `[16]`, `[17]` → additional masks for the remaining shared
    intermediates (`shares 5/6`) used in the patching experiments.
- Traces are stored as signed `int8` (measured power, windowed). During
  profiling the model scales them (e.g. `StandardScaler` / min–max).
- The loader class is `ReadASCADr` in
  `src/feature_emergence/datasets/load_ascadr.py`.

---

## 3. `ches_ctf/ches_ctf_nopoi_window_20.h5` — CHES CTF 2018

**Board/target:** AES-128 **software** on an **ARM Cortex-M4** (32-bit) from
the CHES 2018 Capture-the-Flag side-channel challenge. The implementation is
protected with first-order Boolean masking, but the mask values are **not
included** in the metadata. Traces have been re-windowed
(`nopoi_window_20`) to 15,000 samples around the interesting region.

**Structure:**

```
Profiling_traces/
  traces    (30000, 15000)  float16
  metadata  (30000,)        [('plaintext', '<f8', (16,)), ('ciphertext', '<f8', (16,)), ('key', '<f8', (16,))]
Attack_traces/
  traces    (10000, 15000)  float16
  metadata  (10000,)        [('plaintext', 'u1', (16,)), ('ciphertext', 'u1', (16,)), ('key', 'u1', (16,))]
```

**Metadata columns:**

| Field | Type | Description |
|-------|------|-------------|
| `plaintext` | `float64`/`uint8[16]` | 16 AES plaintext bytes |
| `ciphertext` | `float64`/`uint8[16]` | 16 resulting AES ciphertext bytes (no mask fields) |
| `key` | `float64`/`uint8[16]` | 16 AES key bytes (fixed key `175cf2997a8583413c77dfac7e6c59d8`) |

**Notes:**
- No `masks` field: the mask values are not provided, so analyses work without
  mask information even though the implementation is masked.
- Note the type inconsistency between groups (profiling metadata is
  `float64`, attack metadata is `uint8`); the loader normalizes this.
- The loader class is `ReadCHESCTF` in
  `src/feature_emergence/datasets/load_chesctf.py`.

---

## Loading in code

Preferred — via the package (paths resolve to this directory automatically):

```python
from feature_emergence.utils import load_dataset

# ESHARD: 1,400 samples, HW leakage model
ds = load_dataset("eshard", traces_dim=1400, leakage_model="HW")

# ASCADr: 2,000 samples, identity leakage model
ds = load_dataset("ascad-variable", traces_dim=2000, leakage_model="ID")

# CHES_CTF: 15,000 samples, HW leakage model
ds = load_dataset("ches_ctf", traces_dim=15000, leakage_model="HW")

ds.x_profiling      # profiling traces
ds.profiling_labels # Sbox/leakage labels
ds.profiling_keys, ds.profiling_plaintexts, ds.profiling_masks
```

Raw — direct access with `h5py`:

```python
import h5py
f = h5py.File("data/ASCADr/ascad-variable_70k-90k_20.h5", "r")
prof_traces = f["Profiling_traces/traces"][:]
prof_meta   = f["Profiling_traces/metadata"][:]
keys        = prof_meta["key"]   #  (200000, 16)
plaintexts  = prof_meta["plaintext"]
masks       = prof_meta["masks"]
```

---

## Key bytes used for labeling

| Dataset | Fixed key (hex) | Target byte in paper |
|---------|-----------------|----------------------|
| eshard | `077BDA0FAB1E5501EAD0150AB1E020FE` | 2 |
| ASCADr | `00112233445566778899AABBCCDDEEFF` | 2 |
| CHES_CTF | `175cf2997a8583413c77dfac7e6c59d8` | 2 |