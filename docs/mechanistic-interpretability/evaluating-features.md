# Evaluating Features

A featurizer always finds *something*: a low reconstruction error is easy to get, and it says
nothing about whether the features mean anything. This page covers the measures in
`featurizer_metrics.hpp` (the core of SAEBench, Karvonen et al.), the baselines that make them
interpretable, and what they showed when every featurizer in the library was trained on the
same protein-model layer.

**The rule of this page: no measure without a baseline.**
- Train the same featurizer on a **randomly initialized copy** of the model
  (`NullModelBaseline`). Anything that also shows up there isn't evidence of what the trained
  model learned.
- For a concept, compare with a **linear probe** on the raw activations, and with the best
  **single neuron**.

SAEBench's authors find that these proxy measures don't reliably predict how useful a
featurizer is downstream. Treat them as diagnostics, not scores to maximize.

## Reconstruction

```cpp
#include "pulsatrix/featurizer_metrics.hpp"

ReconstructionMetrics r = EvaluateReconstruction(sae, held_out);
```

| Field | Meaning | Read it as |
|---|---|---|
| `explained_variance` | Share of the activations' variance the reconstructions keep | Higher is better, but a random model's activations are often *easier* to compress |
| `cosine` | Mean cosine between input and reconstruction | |
| `norm_ratio` | Mean `‖x̂‖ / ‖x‖` | Below 1 means codes are shrunk (typical of L1) |
| `l0` | Active features per input | The sparsity you actually got |
| `dead_fraction` | Features that never fired on these inputs | Wasted capacity |
| `dense_fraction` | Features firing on more than 10% of inputs | Usually biases, not concepts |

For a transcoder, `EvaluatePrediction(t, inputs, targets)` measures prediction of the MLP's
output instead.

## Loss recovered

The most direct reconstruction test: put the reconstructions back into the model and see how
much of its loss survives.

```cpp
// Position 4: the residual stream after block 4 (0 is the embeddings).
LossRecovered lr = MeasureLossRecovered(sae, /*position=*/4, [&](const HiddenStateHook& hook) {
    model->set_hidden_state_hook(hook);
    const double loss = HeldOutLoss(*model);   // your evaluation loss
    model->set_hidden_state_hook({});
    return loss;
});
// lr.clean, lr.spliced, lr.ablated; lr.recovered = (ablated - spliced) / (ablated - clean)
```

- **1** means splicing costs nothing; **0** means it is as bad as zeroing the position.
- **NaN** means zeroing the position doesn't raise the loss, so there is nothing to recover.
  This happens on random models.
- `SpliceHook` and `AblationHook` build the two replacements. Their `rows` mask limits them to
  the positions the featurizer was trained on, for example residues but not `<cls>` and `<eos>`.
- For a transcoder, `MeasureMlpLossRecovered` replaces one MLP's output instead. Zeroing a single
  MLP usually costs much less than zeroing the residual stream, so the two numbers aren't
  comparable.

## Absorption

**Feature absorption** (Chanin et al.): a feature that stands for a concept goes silent where a
more specific feature fires instead and carries the concept's direction. The classic example is
a "starts with S" feature that skips "short", because a "short" feature took over.

```cpp
AbsorptionResult a = FeatureAbsorption(sae, held_out, labels);   // labels: 0/1 per input
// a.probe_f1, a.main_features, a.main_f1, a.absorption_rate, a.absorbing_features
```

How it is measured:
1. A logistic probe finds the concept's direction.
2. The **main features** are those that each raise F1 by at least 0.03.
3. A held-out positive the probe finds counts as absorbed when no main feature fires on it, and
   other features aligned with the probe carry at least 40% of its projection onto the probe.

Only trust absorption where the probe itself finds the concept well (F1 at least 0.5), and
compare the rate with the random model's. It needs single-direction features, so block-sparse
featurizers and transcoders are refused.

## Concept matching

To ask "is there a feature for concept X?", score every feature against labels by its best F1
over activation thresholds: `EncodeSparse` then `MatchConcept` (`protein_concepts.hpp`). Report
the best feature's F1 beside the best **single neuron's** and the best feature of a
**random-model** featurizer.

## Comparing featurizers on ESM-2

Every featurizer was trained by `pulsatrix_probe_esm --sae --featurizer ...`
([how to run it](featurizers.md#on-a-protein-model-pulsatrix_probe_esm)) on the same data:
- **Model and layer:** ESM-2 8M, the residual stream after layer 4.
- **Data:** residues of 2,000 Swiss-Prot proteins, each dimension standardized. Scores are on
  held-out proteins.
- **Budget:** 2,560 dictionary directions and about 16 active dimensions per residue. For the
  block-sparse featurizers that is 640 blocks of 4 with 4 active.

| | L1 SAE | TopK | BatchTopK | Matryoshka | JumpReLU (λ 0.005) | BSF Vanilla | BSF Grassmannian | BSF GroupLasso |
|---|---|---|---|---|---|---|---|---|
| L0 (features or blocks) | 17.0 | 16.0 | 16.5 | 16.5 | 12.6 | 4.0 | 4.0 | 4.1 |
| Explained variance | 0.66 | **0.78** | **0.78** | 0.77 | 0.75 | 0.66 | 0.65 | 0.54 |
| Loss recovered | 0.90 | **0.96** | 0.95 | 0.90 | 0.95 | 0.91 | 0.92 | 0.90 |
| Helix: best feature's F1 | 0.16 | 0.33 | 0.33 | **0.48** | 0.37 | 0.40 | 0.17 | 0.43 |
| Transmembrane: best feature's F1 | 0.52 | 0.63 | 0.69 | **0.78** | 0.73 | 0.60 | 0.75 | 0.47 |
| Helix absorbed (random model) | 42% (10%) | 25% (15%) | 27% (5%) | **22%** (15%) | 29% (8%) | — | — | — |
| Transmembrane absorbed (random model) | 34% (21%) | 30% (17%) | 19% (2%) | **9%** (4%) | 34% (11%) | — | — | — |

For reference, the best single neurons reach F1 0.50 for helix and 0.53 for transmembrane.

**What this says:**
- **Use TopK or BatchTopK for reconstruction.** They keep the most variance and the most of
  the model's loss at this sparsity. L1 loses on every measure at the same L0.
- **Use Matryoshka for clean concept features.** It absorbs least and has the best single
  features for helix and transmembrane. It pays in reconstruction (loss recovered 0.90 vs
  0.96), as its authors report.
- **JumpReLU matches TopK's loss recovered with fewer features** (12.6 vs 16), but absorbs
  slightly more (29% vs 25% for helix).
- **Block-sparse featurizers don't pay off at this layer.** At the same 16 active dimensions they
  reconstruct worse than TopK. Their blocks use about 1.5 of their 4 dimensions, so few protein
  concepts here look like surfaces of more than one dimension. They describe an input in about
  5% fewer bits than TopK, not the large factor the paper finds on image models.
- **Single neurons win for frequent concepts.** Helices and strands cover about a third of all
  residues, and no single sparse feature recalls them as well as the best neuron does.

**What the baselines showed:**
- **A random model's SAE explains *more* variance**: 0.95 for TopK on the random ESM-2, against
  0.78 on the trained one, with 85% of its features dead. Random representations are easy to
  compress, so explained variance alone says nothing about learning.
- **Loss recovered needs a position that matters.** Zeroing layer 4 of the random model doesn't
  raise its loss, so `recovered` is NaN there.
- **Some concept features are in the random model too.** Disulfide bonds (a "cysteine" feature
  goes a long way) and coiled coils (their sequence repeats) appear in a random model's SAE. The
  random-model absorption rates in the table are the floor for each figure.

The per-concept tables, with every concept, are in the
[Protein language models guide](../protein-models/index.md) and the roadmap's
[FEAT notes](../roadmap/index.md).

## Transcoders on ESM-2

`pulsatrix_probe_esm --featurizer transcoder` (or `skip-transcoder`) trains on the MLP that
writes into layer 4, at k = 16 with 2,560 features:

| | Transcoder | Skip transcoder |
|---|---|---|
| Explained variance of the MLP's output (random model) | 0.67 (0.96) | 0.81 (0.997) |
| Dead features | 25% | 8% |
| Loss recovered with the MLP replaced | 0.59 | 0.72 |

- **The skip connection helps on every measure.**
- **A random model's MLP is almost linear,** so transcoders explain nearly all of it: explained
  variance means nothing here either.
- **Against the TopK SAE's features, the skip transcoder's are mixed.** It is better for helix,
  disulfide bonds and signal peptides, and worse for strands and transmembrane. That is not the
  clear win Paulo et al. report for language models, where the measure was automated
  interpretability scores, not concept F1.

## References

- Karvonen et al., "SAEBench", arXiv 2503.09532.
- Chanin et al., "A is for Absorption", arXiv 2409.14507.
- Simon and Zou, "InterPLM", for SAE concept matching on protein models.
