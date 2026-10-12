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
| What does a layer look like in 2D or 3D? Which directions vary most? | `FitPca`, `PcaProject`, `PcaSplitHalfOverlap`, `RenderProjectionSvg` | [Below](#pca-projections) |
| Does a concept, given by examples, push the model towards a class? | `RunTcav`, `TrainCav` | [Below](#concepts-tcav) |
| Which of a layer's channels does an LRP explanation pass through, and where in the input is each? | `ChannelRelevance`, `ConditionalHeatmaps`, `ConceptReferenceSamples` (CRP) | [LRP](../interpretability/lrp.md#concepts-concept-relevance-propagation) |
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

### PCA projections

`FitPca()` finds the directions along which a layer's representations of a set of inputs vary
most. `PcaProject()` places inputs on them, to plot, or to pass fewer dimensions on to another
method.

```cpp
#include <fstream>

#include "pulsatrix/pca.hpp"
#include "pulsatrix/viz/svg.hpp"

// layer: (n, width), a row per input; words and tags: one per input.
const Pca pca = FitPca(layer, /*components=*/3);
const Tensor scores = PcaProject(pca, layer);  // (n, 3)
const ProjectionDocument doc = MakeProjectionDocument(pca, scores, /*labels=*/words, /*groups=*/tags);
std::ofstream("layer.svg") << RenderProjectionSvg(doc);          // PC1 against PC2
std::ofstream("layer13.svg") << RenderProjectionSvg(doc, 0, 2);  // PC1 against PC3
std::ofstream("layer.json") << ToJson(doc);                     // for pulsatrix_svg or HTML
```

It matches scikit-learn's `PCA` to 1e-10:

- The data are centered.
- Variances divide by n − 1.
- Each component's largest entry is positive, so the same data always give the same picture.

It runs in double precision, through whichever is smaller of the covariance and the Gram
matrix.

- **Plots.** `RenderProjectionSvg` draws two axes on one scale, so distances on the page are
  distances in the projection. Each axis is captioned with its share of the variance.
  - Points are colored by group or by value, and show their label on hover.
  - `RenderProjectionHtml` is the zoomable version. With three components it shows the three
    pairs side by side, the views of a 3D projection.
  - The `pulsatrix.projection.v1` document (`ToJson`) also lets any outside tool draw it.
- **Steering directions.** `PcaComponent(pca, i)` is component i as a unit vector for
  `SteeringHook`. One standard deviation along it is `sqrt(pca.explained_variance[i])`, a
  natural unit for the coefficient. Then test whether it does anything with `MeasureSteering`
  ([Steering](steering.md)).
- **Fewer dimensions.** `PcaReconstruct` maps coordinates back. `PcaProject(pca, x,
  /*whiten=*/true)` gives every axis variance 1.

Things to know:

- **Check that the picture is stable.** Components with close variances can swap or rotate
  between two samples of the same inputs. `PcaSplitHalfOverlap(layer, k)` fits each half of
  the inputs separately and compares the two k-dimensional spans: 1 means the same span, near
  k / width means unrelated. In the test, points on a plane give 0.9995, the same plane under
  noise as large as it gives 0.92, and noise alone gives 0.09.
- **Most variance isn't most meaning.** In anisotropic layers the first component is often the
  common direction or a few outlier dimensions (see [the shape of a layer](#the-shape-of-a-layer)).
  Look at the explained variance ratios, and treat a direction as a hypothesis to test.
- **`SubspaceOverlap(a, b, k)`** compares any two PCAs of the same width, for example the same
  layer in two models, or before and after fine-tuning.

### Concepts (TCAV)

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
