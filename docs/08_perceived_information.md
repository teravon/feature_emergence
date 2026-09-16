# Chapter 8 — Perceived Information

[Chapter 7](07_scoring_the_attack.md) scored a finished model with guessing
entropy. The checkpoints of [Chapter 6](06_training_and_checkpoints.md) also
store every earlier epoch. This chapter measures how informative the model’s
softmax outputs are at each of those epochs, as a single number in bits:
**Perceived Information (PI)**.

## GE and PI



## Definition



## PI vs training epoch



## Profiling set and attack set



## ASCADr



## Next

When PI moves sharply across epochs, training has crossed a transition. That
is the start of Act III.

← Previous: [Chapter 7 — Scoring the attack](07_scoring_the_attack.md) · [Index](README.md) · Next: [Appendix — Datasets guide](appendix_datasets.md) →
