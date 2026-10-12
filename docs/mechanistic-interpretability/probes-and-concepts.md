# Probes and Concepts

These tools test for concepts you can name. A **probe** asks whether a layer holds a concept you
have labels for. **TCAV** asks whether a concept, given only by examples, pushes the model
towards a class. Both need a control, and this page shows which.

To find the concepts a layer uses without naming them first, train a
[featurizer](featurizers.md). To see where a layer's own channels are in an input, use
[Concept Relevance Propagation](../interpretability/lrp.md#concepts-concept-relevance-propagation).

The examples collect activations with the hooks described in the
[overview](index.md#reading-and-changing-activations).

## Probing a layer for a concept

A linear probe tests whether a concept can be read out of a layer linearly. `LinearProbe` is a
`LinearModule(activation_dim, 1)` trained with `BCEWithLogitsLoss` on `(activation, label)`
pairs.

```cpp
#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/linear_probe.hpp"

// activation_batch: (N, 64) activations; label_batch: (N, 1), values in {0, 1}.
LinearProbe probe(/*activation_dim=*/64, &backend);
AdamOptimizer optimizer(0.01f, &backend);
for (int step = 0; step < 200; ++step) probe.train_step(activation_batch, label_batch, optimizer);
const float accuracy = probe.accuracy(held_out_activations, held_out_labels);
```

- **High held-out accuracy** means the concept is linearly readable.
- **Chance accuracy** means it isn't, at least not linearly.
- **A probe reads; it doesn't show the model *uses* the information.**
- **High accuracy can come from the probe rather than the representation.** Run a control
  (below).

### Controls: the representation or the probe?

`probe_controls.hpp` trains a multi-class probe, `TrainLinearProbe()` (standardized features,
AdamW, cross-entropy, scored on held-out rows), and gives it two kinds of control.

**Hewitt and Liang's control task** (EMNLP 2019) gives each token type, such as each word, a
random label and keeps it fixed. The labels are drawn from the real labels' distribution. A
probe can only learn that by memorizing types, so **selectivity**, task accuracy minus control
accuracy, is what the representation adds beyond memorization.

```cpp
#include "pulsatrix/probe_controls.hpp"

// features: rows x dim; labels: one class per row; token_types: which word each row is.
const ProbeSelectivity s = MeasureSelectivity(features, dim, labels, /*classes=*/2, token_types,
                                              train, test, &backend);
// s.task.accuracy, s.control.accuracy, s.selectivity, s.train_types
```

The test shows the difference on two representations of the same 100 token types. In the
structured one, each type's label is a linear function of an 8-dimensional vector. The one-hot
one only names the type:

| Representation | Task accuracy | Control accuracy | Selectivity |
|---|---|---|---|
| Structured (8 dimensions) | 0.96 | 0.61 | +0.35 |
| One-hot type (100 dimensions) | 0.96 | 0.98 | −0.02 |

Both read the task equally well. Only the structured representation does it without being able
to memorize arbitrary labels.

`ControlTaskLabels()` makes the control labels on their own, for example for a `LinearProbe`.

**When there are few types, compare against a baseline representation.** With no more types
than dimensions, a linear probe learns any control task. On 20 amino acids it scores 1.0 at
every layer of a protein model. `CompareToBaseline()` trains the same probe on the
representation and on a baseline, with the same rows and split, and reports the gain.

Good baselines:

- one-hot local sequence, `SequenceWindowFeatures()`;
- the same layer of a randomly initialized model;
- the raw input.

```cpp
const std::vector<float> window = SequenceWindowFeatures(sequences, /*window=*/3);
const ProbeGain g = CompareToBaseline(layer, hidden, window, 20 * 7, labels, 2, train, test, &backend);
// g.gain, g.balanced_gain, g.auc_gain (two classes)
```

The [Protein language models guide](../protein-models/index.md) uses this. Disulfide probes show
no gain over local sequence, because they find cysteines. Secondary structure gains 0.20 to
0.25 balanced accuracy.

Recipe: [Sparse autoencoder + linear probe](../recipes/mechanistic-interpretability/sparse_autoencoder_probe.md).

## Concepts (TCAV)

TCAV (Kim et al., 2018) asks whether a human concept, given only by examples (striped images,
say), pushes a model towards a class. It works in three steps:

1. **A concept activation vector (CAV).** `TrainCav` fits a linear classifier that separates a
   layer's activations on the concept's examples from its activations on random examples. Its
   normal, pointing towards the concept, is the CAV.
2. **Sensitivity.** Each input's sensitivity is the class logit's gradient at that layer,
   dotted with the CAV.
3. **The TCAV score.** It's the fraction of the class's inputs with positive sensitivity.

```cpp
#include "pulsatrix/tcav.hpp"

ExplainerContext ctx({&conv1, &relu1, &conv2, &relu2, &flatten, &fc});
// Node i + 1 is module i's output: 4 is relu2's.
const TcavResult r = RunTcav(ctx, /*layer=*/4, /*target=*/zebra, zebra_images, striped_images,
                             random_image_sets, &backend);
// r.mean_score, r.significant, r.test.p_value, r.cav_accuracies
```

**One score means little.** A CAV trained against one random set can point anywhere that
separates the two sets, and a meaningless direction scores anywhere from 0 to 1. `RunTcav`
follows Kim et al.:

- It trains a CAV against each of several random sets (at least 3; Kim et al. use many more).
- It does the same with random sets in place of the concept.
- A two-sided t-test (`StudentTTest`) compares the two lists of scores.

A concept is significant at `alpha` (0.05; divide it by the number of concepts, layers and
classes you test). A score near 0 is as significant as one near 1: the concept pushes away from
the class.

In the test's network, class 0's logit is 2 h0 − h1. Concepts that raise h0 score 1, and those
that raise h1 score 0. Both are significant. A concept that raises only h2, which the logit
ignores, isn't, even though its CAVs separate it at over 85%.

Things to know:

- **Use plenty of concept examples.** Every CAV of a concept shares its examples, so their
  sampling noise tilts every CAV the same way. In the test, an ignored concept passed the test
  in 8 of 40 trials with 30 concept examples, where about 2 are expected by chance. With 300
  examples it passed in none.
- **Check the CAV's accuracy.** `cav_accuracies` are measured on held-out examples. Near 0.5,
  the layer doesn't separate the concept and the CAV means nothing.
- **Logistic regression or difference of means.** `CavOptions::method` chooses. Logistic
  regression is Kim et al.'s. `DifferenceOfMeans` needs no fitting and doesn't tilt towards
  dimensions that separate the two samples only by chance.
- **Models.** The models are `ExplainerContext` chains. TCAV needs the gradient at an inner
  layer, which the language models don't expose yet.

TCAV starts from concepts people name and asks whether the model uses them.
[Concept Relevance Propagation](../interpretability/lrp.md#concepts-concept-relevance-propagation)
goes the other way: it starts from the concepts a layer has learned (its channels) and shows
where each one is in an input and what it responds to.

Full API reference: [Doxygen: Mechanistic Interpretability](../api/group__mech__interp.html)
