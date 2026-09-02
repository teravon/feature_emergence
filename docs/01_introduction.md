# Chapter 1 — Introduction

## Cryptography in the real world

AES (Advanced Encryption Standard) is the most widely used cipher on the
planet. It takes 16 bytes of input (the **plaintext**) and a secret 16-byte
**key**, and produces 16 bytes of **ciphertext**. For example, with

```text
plaintext  = c549dd73446310a073b9f9c6bdc7e66d
key        = 175cf2997a8583413c77dfac7e6c59d8
```

AES-128 produces

```text
ciphertext = e70e50e7a0829e5f6691ac64dde13056
```

(This is not a made-up example: it is a real encryption taken from one of the
datasets we will explore in [Chapter 2](02_the_data.md). You can verify it
with any AES calculator.)

AES is designed to be secure against classical analysis: if you don't know
the key, recovering it by examining plaintexts and ciphertexts is, for all
practical purposes, infeasible. That is the theory.

In the physical world, however, an algorithm runs on a **device** — a smart
card, a microcontroller, a chip. A device consumes **power**, emits
**electromagnetic radiation**, and takes a certain **amount of time** for each
operation. All of these physical quantities depend on the data being
processed — including the secret key.

An attack that exploits these unintentional physical emissions is called a
**side-channel attack**. The attacker does not need to break the mathematics
of AES: they only need to measure something the device *leaks*.

## What is a power trace?

Every time the device performs an encryption, we can record its power
consumption over time with an oscilloscope. That recording is a **power
trace** — a long list of numbers, one power measurement per time step.

```mermaid
flowchart LR
    D["device<br/>running AES"] -->|"power consumption"| O["oscilloscope"]
    O --> T["power trace<br/>= thousands of samples"]
    P["known plaintext"] -.-> LOG["dataset:<br/>trace + plaintext stored together"]
    T --> LOG
```

Here is a real trace from one of our datasets — one full AES encryption,
recorded as 15,000 power measurements:

![One power trace: a single AES encryption](assets/figures/02_single_trace.png)

To the eye it looks like noise. But different intermediate values inside the
chip cause slightly different power consumption, and if we record **many**
traces together with their known plaintexts, statistics can find where those
differences hide. That is the whole game: **find which time samples leak
information about the key**, then use that leakage to recover the key.

## The leakage model

The leakage is related to the *intermediate values* computed during the
cipher, not to the raw key bytes directly. AES works in rounds, and in the
first round each plaintext byte is combined with one key byte and passed
through a lookup table called the **S-box**:

```mermaid
flowchart LR
    P["plaintext byte<br/>(known)"] --> X["XOR"]
    K["key byte<br/>(secret)"] --> X
    X --> S["S-box<br/>(public table)"]
    S --> O["intermediate value<br/>= Sbox[plaintext ⊕ key]<br/>(this leaks in the power)"]
```

The plaintext is known to the attacker; the key byte is what we want. Because
a byte has only 256 possible values, we can enumerate all candidates for the
key byte and check which one explains the observed power. Two common ways of
labelling the traces — *leakage models* — are:

- **Identity (ID):** the label is the full 8-bit S-box output (256 classes).
- **Hamming Weight (HW):** the label is the number of `1` bits in that output
  (9 classes), reflecting the fact that power consumption grows with the
  number of bits that flip inside the chip.

## Why deep learning?

Classical side-channel analysis builds a statistical model of the leakage by
hand — for example, comparing traces against a hypothetical power model and
scoring key guesses. This requires a good understanding of *where* the
leakage is.

Deep learning instead **learns the leakage directly from the traces**: the
network is given raw traces and their labels, and it learns to map a trace to
the likely intermediate value. No expert feature selection is needed — the
network figures out which time samples are informative.

That brings us to the central question of the whole project:

> **When, during training, does the network actually learn the features that
> make it leak the key?**

That question is the heart of this project, and we will build up to it step by
step. First, we need to look at the data we will be playing with.

← [Index](README.md) · Next: [Chapter 2 — The data](02_the_data.md) →
