# Chapter 2 — From plaintext to intermediate value

[Chapter 1](01_secrets_that_leak.md) left the attacker with a power trace and a
known plaintext. The leakage is not related to the raw key bytes directly. It
is related to the *intermediate values* computed during the cipher.

AES works in rounds. In the first round each plaintext byte is combined with
one key byte and passed through a lookup table called the **S-box**:

```mermaid
flowchart LR
    P["plaintext byte<br/>(known)"] --> X["XOR"]
    K["key byte<br/>(secret)"] --> X
    X --> S["S-box<br/>(public table)"]
    S --> O["intermediate value<br/>= Sbox[plaintext ⊕ key]<br/>(this leaks in the power)"]
```

The S-box is a fixed, public 256-entry table — identical on every AES chip in
the world. For a slower walk through one AES round and the full table, see
the companion page **[AES-128 and the S-box](assets/pages/aes_and_sbox.html)**.

The plaintext is known to the attacker; the key byte is what we want. Because
a byte has only 256 possible values, we can enumerate all candidates for the
key byte and check which one explains the observed power. Two common ways of
labelling the traces — *leakage models* — are:

- **Identity (ID):** the label is the full 8-bit S-box output (256 classes).
- **Hamming Weight (HW):** the label is the number of `1` bits in that output
  (9 classes), reflecting the fact that power consumption grows with the
  number of bits that flip inside the chip.

Concretely, for a known plaintext byte `p` and a candidate key byte `g`:

```text
ID label  = Sbox[p ⊕ g]           ∈ {0, 1, …, 255}
HW label  = popcount(Sbox[p ⊕ g]) ∈ {0, 1, …, 8}
```

(`popcount` counts the `1` bits of a byte.)

For the first ASCADr attack trace used later in these chapters (`p = 0x8e`)
and the true key byte `k = 0x22`, the correct candidate `g = 0x22` gives:

```text
ID label  = Sbox[0x8e ⊕ 0x22] = 145
HW label  = popcount(145)     = 3    (145 = 0b10010001)
```

Attacking therefore means relating the trace to one of these intermediate
labels — not to the key string itself.

**What this does not say.** ID and HW are models the attacker chooses, not
truths about the chip — the silicon knows nothing of Hamming weights; it just
moves bits and dissipates power. And the first-round S-box output is one
possible target among several; it is the standard one because it is where the
secret key first meets the known plaintext.

On unprotected devices that relation can already be strong. On the devices we
study it is not, because of **masking**.
