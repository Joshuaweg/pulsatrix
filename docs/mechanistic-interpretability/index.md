# Mechanistic Interpretability

[Interpretability](../interpretability/index.md) tells you which parts of an *input* mattered
for a prediction. Mechanistic interpretability asks how the *network* computes it:

- what its layers represent;
- which directions in its activations stand for concepts;
- which of its weights do the work;
- how information flows from the prompt to the answer.

This section covers the tools pulsatrix has for that, from reading activations to full
attribution graphs of a language model's prediction.

## Which tool should I use?

| Your question | Tool | Page |
|---|---|---|
| Is concept X linearly readable from layer L? | `LinearProbe`, `TrainLinearProbe()`, with `MeasureSelectivity()` or `CompareToBaseline()` as the control | [Below](#probing-a-layer-for-a-concept) |
| How similar are two layers, two models, or a model before and after fine-tuning? | `LinearCka`, `LinearCkaMatrix` | [Below](#comparing-representations-cka) |
| Which inputs does a layer treat as alike? Do two models agree on it? | `NearestNeighbors`, `MutualKnnAlignment`, `MutualNearestNeighbors`, `MeasureHubness` | [Below](#nearest-neighbors) |
| Is cosine similarity meaningful in this layer? How many dimensions does it use? | `Anisotropy`, `MeasureEffectiveRank`, `TwoNNDimension` | [Below](#the-shape-of-a-layer) |
| Which directions in a layer stand for concepts, found without labels? | Sparse autoencoders: `TopKSparseAutoencoder`, `JumpReLUSparseAutoencoder`, `SparseAutoencoder` | [Finding features](featurizers.md) |
| Which concepts take more than one dimension, such as an angle or a position? | `BlockSparseFeaturizer` | [Finding features](featurizers.md#block-sparse-featurizers) |
| What does an MLP compute, step by step? | `Transcoder` | [Finding features](featurizers.md#transcoders) |
| Are my features any good? | `EvaluateReconstruction`, `MeasureLossRecovered`, `FeatureAbsorption`, with baselines | [Evaluating features](evaluating-features.md) |
| Can a direction push the model's behavior, and on how many inputs does it work? | `SteeringHook`, `MeasureSteering` | [Steering](steering.md) |
| What did fine-tuning change in a model? | `Crosscoder`, `MeasureLatentScaling`, `pulsatrix_diff_lm` | [Model diffing](model-diffing.md) |
| Which pieces of the weights does each input use? | `ParameterDecomposition` (SPD, VPD) | [Parameter decomposition](parameter-decomposition.md) |
| Which features carry a language model's prediction, layer to layer? | `TraceCircuit`, `pulsatrix_explain_text --transcoders` | [Attribution graphs](attribution-graphs.md) |
| Which layers does a small network's output depend on? | `ExplainerContext::build_circuit_graph()` | [Attribution graphs](attribution-graphs.md#ablation-circuit-graphs-for-small-networks) |
| What happens if I overwrite one activation? | `ExplainerContext::forward_pass_with_patch()` | [Below](#reading-and-changing-activations) |
| What would the model predict if layer L were its last? | `ExplainerContext::logit_lens()` | [Below](#reading-and-changing-activations) |
| How do I sample outcomes in proportion to a reward? | GFlowNets | [GFlowNets](gflownets.md) |

## A typical workflow

For a researcher starting on a model, the pieces fit together like this:

1. **Collect activations** from the layer you care about, with a hook (below).
2. **Probe** for the concepts you already have labels for. A probe tells you whether the
   information is there.
3. **Train a featurizer** (a TopK or BatchTopK SAE is a good default) to find the directions the
   model uses without being told what to look for.
4. **Evaluate it against baselines**: the same featurizer on a randomly initialized copy of the
   model, and a linear probe. A feature that also shows up in a random model says nothing about
   what the trained one learned.
5. **Test causally.**
   - Splice the reconstructions back in and see how much of the loss you keep (loss
     recovered).
   - Steer along a direction, with a random-direction control.
   - For one prediction, trace an attribution graph.

Each page reports what these steps found on real models (ESM-2 protein models, SmolLM2,
Gemma 3), including results that didn't work out. Read those before relying on a method.

## Building blocks

### Reading and changing activations

Language models expose two hooks, used throughout this section:

- **`HiddenStateHook`** on `CausalLM` and `EncoderLM`. It is called with each residual-stream
  position during `forward()`: 0 for the embeddings, i for the output of block i. It returns the
  tensor the model continues with. Return it unchanged to only read.
- **`MlpHook`** on each `TransformerBlock` and `EncoderBlock`. It sees an MLP's input (after the
  block's norm) and its output, and returns what is added to the residual stream.

```cpp
#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/tokenizer_json.hpp"

using namespace pulsatrix;

CPUBackend backend;
std::unique_ptr<CausalLM> model = LoadCausalLM("models/SmolLM2-135M", &backend);
const TextTokenizer tok = LoadTokenizerJson("models/SmolLM2-135M/tokenizer.json");
const int64_t d = model->config().hidden_size;

// corpus: your texts, a std::vector<std::string>.
// Collect the residual stream after block 6, one row per token.
std::vector<float> rows;
model->set_hidden_state_hook([&](int64_t position, const Tensor& h) {
    if (position == 6) {
        const std::vector<float> v = h.to_host_vector();   // (1, tokens, d)
        rows.insert(rows.end(), v.begin(), v.end());
    }
    return h;                                              // unchanged: only read
});
for (const std::string& text : corpus) {
    const std::vector<int64_t> ids = tok.encode(text).ids;
    const std::vector<float> ids_f(ids.begin(), ids.end());
    (void)model->forward(Tensor(Shape({1, static_cast<int64_t>(ids.size())}), &backend, ids_f));
}
model->set_hidden_state_hook({});
const Tensor activations(Shape({static_cast<int64_t>(rows.size()) / d, d}), &backend, rows);
```

For small networks built from layers, `ExplainerContext` caches every layer's output and offers
the classic tools:
- `activation_snapshot()`: a copy of one forward pass's activations that stays valid after
  later passes.
- `forward_pass_with_patch()`: activation patching, overwriting one activation.
- `logit_lens()`: what the model would predict if a given layer were the last.
- `attention_weights()`.

### Probing a layer for a concept

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

#### Controls: the representation or the probe?

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

### Comparing representations (CKA)

Centered kernel alignment (CKA) measures how similar two representations of the same inputs
are: one layer of two models, two layers of one model, a model before and after fine-tuning, or
the same layer computed on the CPU and the GPU. Run the same n inputs through both, collect one
row per input from each, and compare:

```cpp
#include "pulsatrix/representation_similarity.hpp"

// layer_a: (n, 768) activations from one model; layer_b: (n, 1024) from another, same n inputs.
const double similarity = LinearCka(layer_a, layer_b);
```

- **1** means one representation is a rotation and uniform scaling of the other, plus an offset.
  The two layers don't need the same width, and their neurons don't need to line up.
- **Near 0** means they are unrelated.
- Dimensions after the first are flattened, so a convolution's (n, channels, height, width)
  activations are n vectors. The tensors can be on any device; CKA is computed in double
  precision on the host.

`LinearCkaMatrix(layers_a, layers_b)` compares every layer of one model with every layer of
another, reusing each layer's work across its row or column. Draw the result as a heatmap:

```cpp
#include "pulsatrix/viz/svg.hpp"

const CkaMatrix m = LinearCkaMatrix(layers_a, layers_b);
HeatmapDocument doc;
doc.title = "CKA, model A layers × model B layers";
doc.rows = m.rows;
doc.cols = m.cols;
doc.values.assign(m.values.begin(), m.values.end());
std::ofstream("cka.svg") << RenderHeatmapSvg(doc);
```

Things to know:

- **Use the default, debiased estimator.** The original estimator is pulled towards 1 when
  there are few inputs for the width: two unrelated random representations of 50 inputs with
  1,000 features each score 0.95 with it and −0.02 debiased. Pass `CkaEstimator::Biased` to
  reproduce numbers from papers that used it. The debiased value can be slightly negative, and
  needs at least 4 inputs.
- **Compare against an untrained model.** Two layers can look similar because of the
  architecture or the input, not what was learned. In the tests, the second hidden layers of two
  small networks trained from different seeds score 0.68; one of them against the same network
  untrained scores 0.10. Run the comparison on a `NullModelBaseline` too (below).
- **Outliers move it.** A few inputs with very large activations can dominate, and CKA can be
  pushed up or down on purpose (Davari et al., 2022). Check that a result holds on a second set
  of inputs.
- A representation that is constant over the inputs has no CKA: the result is NaN.

The implementation matches the reference code of Kornblith et al. (ICML 2019) to 12 decimal
places for both estimators.

### Nearest neighbors

`NearestNeighbors()` finds, for each input, the k inputs a layer represents most alike, by cosine
similarity (the default) or Euclidean distance:

```cpp
#include "pulsatrix/nearest_neighbors.hpp"

// embeddings: (n, width), a row per input.
NeighborOptions options;
options.k = 5;
options.center = true;  // subtract the mean first; see below
const Neighbors near = NearestNeighbors(embeddings, options);
for (int64_t r = 0; r < near.k; ++r) {
    std::cout << near.index(0, r) << " " << near.score(0, r) << "\n";  // input 0's neighbors, nearest first
}
```

- **Search another set** with `NearestNeighbors(queries, points, options)`, for example new
  inputs against a labelled reference set.
- **Match items across two sets** with `MutualNearestNeighbors(a, b, options)`: the pairs
  (i, j) where each is among the other's k nearest.
- **Compare two models** with `MutualKnnAlignment(a, b, options)`: for each input, the
  fraction of its k neighbors under one representation that are also its neighbors under the
  other, averaged. The two can have different widths. With the default options it is the
  mutual-kNN alignment of Huh et al.'s "Platonic Representation Hypothesis" (2024), and it
  matches their code exactly. Unrelated representations score about k / (n − 1) by chance, so
  read a score against that, not against 0. It complements CKA (above): CKA compares the whole
  geometry, mutual kNN only who is close to whom.

The search is exact, in double precision on the host. It costs about n² × width multiply-adds, so
it suits up to tens of thousands of inputs. Ties keep the lower index first.

Things to know:

- **Center cosine similarities.** Activations often share a large common component, so every
  pair's cosine is high and the ranking follows that component. `center = true` subtracts the
  points' mean first. It doesn't change Euclidean distances.
- **Check for hubs.** In high dimensions a few points become almost everyone's neighbor, and
  many become nobody's (Radovanović et al., 2010). Neighbor lists, and scores built on them,
  then say more about the hubs than about the inputs. `MeasureHubness(neighbors, n)` reports
  how often each point is a neighbor, and the skewness of those counts: near 0 is even, well
  above 1 means hubs. For 300 random Gaussian points and k = 5:

    | Dimension | Skewness | The biggest hub is in | Never a neighbor |
    |---|---|---|---|
    | 3 | −0.04 | 3% of lists | 2% of points |
    | 30 | 2.05 | 12% of lists | 21% of points |
    | 300 | 3.52 | 20% of lists | 36% of points |

    Centering also reduces hubness for cosine similarity (Suzuki et al., 2013).

### The shape of a layer

Three numbers describe a layer's representations of a set of inputs. Each is cheap enough to
compute for every layer.

```cpp
#include "pulsatrix/representation_geometry.hpp"

// layer: (n, width), a row per input.
const double raw = Anisotropy(layer);                     // mean cosine over all pairs
const double centered = Anisotropy(layer, /*center=*/true);
const EffectiveRank rank = MeasureEffectiveRank(layer);  // .effective_rank, .participation_ratio
const TwoNNResult id = TwoNNDimension(layer);            // .dimension, .mle_dimension
```

- **Anisotropy** is the mean cosine similarity between different inputs. Spread-out
  directions score about 0. Language models' layers often score 0.5 or more (Ethayarajh,
  2019): they share a few large directions, so any two inputs look alike by cosine. If raw
  anisotropy is high and centered anisotropy is near 0, the similarity was the shared
  direction. Center before comparing by cosine (`NeighborOptions::center`, above). The test
  data shows this: rows spread evenly score 0.001, shifting them all along one axis raises that
  to 0.63, and centering brings it back to −0.003.
- **Effective rank** (Roy and Vetterli, 2007) is how many directions the inputs spread along.
  It's the exponential of the entropy of the normalized singular values, between 1 and the
  smaller of n and the width. The participation ratio is a coarser count that the largest
  directions dominate more. Both center by default, because a shared direction would otherwise
  take up part of the count.
- **TwoNN intrinsic dimension** (Facco et al., 2017) is how many dimensions the inputs vary in
  locally, which can be far below the width. It uses only each point's two nearest neighbors.
  `dimension` is the paper's fit and matches scikit-dimension and DADApy. `mle_dimension` is
  DADApy's maximum-likelihood version. Exact repeated rows are dropped first and counted in
  `duplicates`. In the test, points filling a 2- and a 5-dimensional cube, written in 40
  dimensions, give 1.91 and 4.53.

Things to know about TwoNN:

- **Low with too few points.** It reads low when the dimension is high for the number of
  points: tens of dimensions need thousands.
- **Noise raises it.** It measures the nearest neighbors' scale, so noise on a low-dimensional
  surface adds dimensions.
- **Cost.** The search is exact, at n² × width.

### Baselines

`NullModelBaseline` re-initializes a model randomly, so the same probe, featurizer or
explanation can run on it ([Interpretability](../interpretability/index.md#the-null-model-baseline)).
It is the single most useful check in this section. In the ESM-2 results on the following
pages, a random model's SAE explains *more* variance than the trained model's, and some
"concept features" appear in it too.

## Command-line tools

| Tool | What it does | Page |
|---|---|---|
| `pulsatrix_probe_esm` | Probes, SAEs, transcoders and block-sparse featurizers on an ESM-2 layer, with concept matching, absorption, loss recovered and random-model baselines | [Finding features](featurizers.md#on-a-protein-model-pulsatrix_probe_esm), [Protein language models](../protein-models/index.md) |
| `pulsatrix_steer_esm` | A steering vector on ESM-2, with the reliability report | [Steering](steering.md) |
| `pulsatrix_diff_lm` | A crosscoder between a base language model and its fine-tune, with Latent Scaling | [Model diffing](model-diffing.md) |
| `pulsatrix_spd_toy` | Parameter decomposition (SPD, VPD) of a toy model of superposition | [Parameter decomposition](parameter-decomposition.md) |
| `pulsatrix_explain_text --graph` | An attribution graph for a prompt, from AttnLRP or (with `--transcoders`) transcoder features, for Neuronpedia's viewer | [Attribution graphs](attribution-graphs.md) |

Each prints its usage with `--help`. Each source file (in `tools/`) starts with a comment that
explains every option.

## Recipes

- [Sparse autoencoder + linear probe](../recipes/mechanistic-interpretability/sparse_autoencoder_probe.md)
- [GFlowNet on HyperGrid](../recipes/mechanistic-interpretability/gflownet_hypergrid.md)

Full API reference: [Doxygen: Mechanistic Interpretability](../api/group__mech__interp.html)
