# Chapter 4 — Models and training

Now we use the profiling data to train a classifier that predicts the leakage
label of a trace.

## The task as a classification problem

For each trace we want the network to output a probability distribution over
the possible intermediate values:

- ASCADr uses the **identity** model → 256 classes
- ESHARD and CHES_CTF use the **Hamming weight** model → 9 classes

```
[ trace: 2000 samples ] → (MLP/CNN) → [ probabilities over 256 values ]
```

Training is standard supervised learning: we minimize categorical
cross-entropy on the profiling split.

## Two architectures

The study uses two families of networks:

- **MLP** (multi-layer perceptron): flat, dense layers. Each input sample is a
  weight — good at finding linear-ish combinations of trace samples, very
  interpretable in terms of "which samples matter".
- **CNN** (convolutional network): 1-D convolutions over the trace. Learns
  local patterns ("kernels") across time, which is powerful for capturing
  leakage shapes, at the cost of being less transparent.

### The MLP used for ASCADr

```text
Input (2000 samples)
 → Dense(100, elu) × 6
 → Dense(256, softmax)
```

### The CNN used for ESHARD / CHES_CTF (roughly)

```text
Input (N samples × 1)
 → Conv1D(16) → BN → AvgPool
 → Conv1D(32) → BN → AvgPool
 → ... → Dense(40/100) → softmax
```

> **[figure: 04_architectures.png — little schematic of MLP vs CNN]**

## Checkpoints: training progress without re-training

Training these networks takes hours on GPU. To make the study reproducible on a
laptop, the authors published **checkpoints**: the network weights saved after
*every training epoch*, from epoch 1 to 100, for every model.

```
models/
├── mlp_ascadr_01.weights.h5 ... mlp_ascadr_100.weights.h5
└── cnn_eshard_01.weights.h5 ... cnn_eshard_100.weights.h5
```

This is the key trick that lets [Chapter 5](05_feature_emergence.md) *watch*
how features appear over time: we load the model as it was at epoch 5, epoch
20, epoch 50, ... and measure what it knows at each moment.

## How well does a model attack?

Success is measured with the **Guessing Entropy (GE)**: given the model's
predictions on attack traces, we rank all 256 possible keys. When the correct
key reaches rank 1, the attack succeeded. GE = 1 "as early as possible" is the
goal.

Next: [Chapter 5 — When do features emerge?](05_feature_emergence.md).