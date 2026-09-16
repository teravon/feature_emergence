# Chapter 6 — Training, generalization, and checkpoints

[Chapter 5](05_the_network.md) defined the network as a function from a
trace to a distribution over intermediate values, and showed where a finished
MLP listens. This chapter is how those weights get set, why the profiling /
attack split matters, how inputs are scaled, and why every training epoch is
saved.

## Training: adjusting the numbers

Training is supervised learning:

1. Show the network a profiling trace (the question) together with its
   leakage label `Sbox[plaintext ⊕ key]` (the correct answer).
2. Compare the network's 256 probabilities against the correct answer. The
   mismatch is summarized in one number, the **loss**.
3. Nudge every one of the 276,456 numbers a little in the direction that
   reduces the loss (the calculus behind this — *gradient descent* — is
   automatic in TensorFlow).
4. Repeat with the next trace.

The loss used here is **categorical cross-entropy**, which in practice is
just `−log(probability of the true class)`, averaged over traces:

```text
true class: 145
model assigns p(145) = 0.5    →  penalty −log(0.5)  = 0.69
model assigns p(145) = 0.001  →  penalty −log(0.001) = 6.9
```

Confident and right is nearly free; confident and wrong costs ten times
more. That asymmetry is what pushes the network, epoch after epoch, to
put probability mass on the values that actually occur.

One **epoch** is one full pass through the profiling set. Models were trained
for 100 epochs — sweeps over the profiling set in batches of 400 — and the
published CHES_CTF run extends to 200.

## Generalization: the split from Chapter 4, revisited

[Chapter 4](04_the_datasets.md) divided every dataset into a profiling set
(everything known) and an attack set (key hidden). The division guards
against a specific failure mode: **overfitting**. A network with 276,456
free parameters is perfectly capable of *memorizing* its training traces —
the way one memorizes an answered exam rather than the subject. A memorizer
scores beautifully on the profiling set and is useless on anything else.

So the network is never judged on the data it trained on. All evaluation in
these chapters — the attack of the next chapter, and the training-time
analysis after it — happens on the attack set: traces the model has never
seen, recorded under different plaintexts and fresh random masks. If the
model ranks the correct key first *there*, it learned something real about
the device, not the answer sheet.

## Scaling the inputs

The datasets arrive in different units — ASCADr as raw `int8` ADC codes,
CHES_CTF as `float16` — and neural networks train poorly when inputs have
arbitrary offsets and scales. Before training, every sample of every trace is
therefore **standardized**: for each time sample, subtract the mean and
divide by the standard deviation, both measured on the profiling set:

```text
scaled sample = (raw sample − mean) / std        →  mostly within −3 … +3
```

After scaling, every dataset speaks the same language: mean 0, standard
deviation 1. The attack set is scaled with the *profiling* statistics — the
attack data itself is never used to fit anything.

## Checkpoints: the weights of every epoch

Training these networks takes hours on a GPU — this project has none. What
makes the whole study reproducible on a laptop are the **checkpoints**
published with the paper: the weight file saved *after each training epoch*
— `..._01.weights.h5` through `..._100.weights.h5` for most families, and up
to `..._200` for the longer CHES_CTF run. The checkpoints live on
[Zenodo](https://zenodo.org/records/15410792), in the same record as the
datasets, and the same script — `scripts/download_files.sh` — fetches both
archives. Like the datasets, they are large binaries excluded from git; the
full inventory is in the [appendix](appendix_datasets.md).

```text
models/
├── mlp_ascadr_01.weights.h5
├── mlp_ascadr_02.weights.h5
├── ...
└── mlp_ascadr_100.weights.h5
```

Loading checkpoint *k* into a freshly built network revives the model exactly
as it stood after epoch *k*. Nothing is retrained. The next chapter uses the
finished (epoch-100) model as an attack tool.

## The exploration notebook

Layer-by-layer summaries, the weight inventory and heatmap read from a
checkpoint, and the epoch-100 load are worked through in the
**[model notebook](notebooks/02_the_models.ipynb)**. It needs TensorFlow, so
run it in the Docker environment described in the [index](README.md).

The open question: the model outputs probabilities over *intermediate
values* — it never says "key byte 0x2b". How do those probabilities, over
many traces, single out one key byte?

← Previous: [Chapter 5 — The network](05_the_network.md) · [Index](README.md) · Next: [Chapter 7 — Scoring the attack](07_scoring_the_attack.md) →
