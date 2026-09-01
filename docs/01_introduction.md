# Chapter 1 — Introduction

## Cryptography in the real world

Cryptographic algorithms such as AES are designed to be secure against
classical analysis: if you don't know the secret key, recovering it by
examining the inputs and outputs of the algorithm is (for all practical
purposes) infeasible. That is the theory.

In the physical world, however, an algorithm runs on a **device** — a smart
card, a microcontroller, an FPGA. A device consumes **power**, emits
**electromagnetic radiation**, and takes a certain **amount of time** for each
operation. All of these physical quantities are functions of the data being
processed, including the secret key.

An attack that exploits these unintentional physical emissions is called a
**side-channel attack**. Crucially, the attacker does not need to break the
mathematics of AES: they only need to measure something the device `leaks`.

## What is a power trace?

Every time the device performs an AES encryption, we can record its power
consumption over time with an oscilloscope. That recording is a **power
trace** — a time series of voltage/power samples.

```
power
  ▲        ╭──╮            ╭──╮
  │   ╭────╯  ╰──╮   ╭────╯  ╰──╮
  │───╯          ╰───╯          ╰───► time
  │              │        │
  │          round 1    round 2
```

Different intermediate values inside the chip cause slightly different power
consumption. If we record many traces together with their known cryptographic
inputs (plaintexts), we can statistically relate the power variations at each
time sample to the secret values that produced them. This is the whole game:
**find which time samples leak information about the key**, and then use that
leakage to recover the key.

## The leakage model

The amount of leakage is related to the *intermediate values* computed during
the cipher, not to the raw key bytes directly. For the first AES round, the
typical target is the **S-box output**:

```
state = Sbox[ plaintext ⊕ key ]
```

where `⊕` is the XOR of two bytes. `plaintext` is known to the attacker; `key`
is what we want. For a fixed key byte, the number of different states is small
(256 values, or 9 Hamming-weight classes), which makes statistics tractable.

Two common "leakage models" — ways of labelling the traces — are:

- **Identity (ID):** the label is the full 8-bit S-box output (256 classes).
- **Hamming Weight (HW):** the label is the number of set bits of that output
  (9 classes), reflecting the intuition that power consumption grows with the
  number of `1`s in a value.

## Why deep learning?

Classical side-channel analysis builds a statistical model of the leakage —
e.g. comparing traces against a Hypothetical Power Model and scoring key
guesses. This requires a good understanding of *where* the leakage is.

Deep learning approaches instead **learn the leakage directly from the
traces**: they are given raw (or lightly pre-processed) traces and the
corresponding labels, and they learn to map a trace to the likely intermediate
value. No expert feature selection is needed — the network figures out which
time samples are informative.

That brings us to the central question of the whole project:

> **When, during training, does the network actually learn the features that
> make it leak the key?**

That is the topic of [Chapter 5](05_feature_emergence.md). First, we need to
look at the data we will be playing with.

Next: [Chapter 2 — The datasets](02_the_datasets.md).