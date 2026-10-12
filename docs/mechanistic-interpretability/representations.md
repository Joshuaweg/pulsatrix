# Analyzing Representations

These tools describe what a layer does with a set of inputs, without labels: how it compares
with another layer or model, which inputs it treats as alike, how many dimensions it uses, and
what it looks like in two or three dimensions.

Each takes a matrix with one row per input, `(n, width)`, collected with the hooks described in
the [overview](index.md#reading-and-changing-activations). Two representations being compared
must come from the same n inputs, in the same order.

| Your question | Tool | Section |
|---|---|---|
| How similar are two layers, two models, or a model before and after fine-tuning? | `LinearCka`, `LinearCkaMatrix` | [CKA](#comparing-representations-cka) |
| Does a model order the inputs as a brain recording or a model of the categories does? | `RsaPermutationTest` | [RSA](#rsa-and-svcca) |
| Which directions do two layers share? | `Svcca`, `CanonicalCorrelations` | [SVCCA](#rsa-and-svcca) |
| Which inputs does a layer treat as alike? Do two models agree? | `NearestNeighbors`, `MutualKnnAlignment` | [Nearest neighbors](#nearest-neighbors) |
| Is cosine similarity meaningful here? How many dimensions are used? | `Anisotropy`, `MeasureEffectiveRank`, `TwoNNDimension` | [The shape of a layer](#the-shape-of-a-layer) |
| What does a layer look like in 2D or 3D? | `FitPca`, `RenderProjectionSvg` | [PCA](#pca-projections) |
| How do I run UMAP or t-SNE on a layer? | `ExportEmbeddings`, `ReadNpy` | [UMAP and t-SNE](#umap-and-t-sne) |

For clusters, loops and voids, see [Topology of representations](topology.md). To compare two representations' topology next to CKA, use [RTD and `CompareRepresentations()`](topology.md#comparing-two-representations-topology-rtd).

**Compare against an untrained model.** Every number on this page can come from the architecture
or the inputs rather than what was learned. Compute it on a
[`NullModelBaseline`](index.md#baselines) too.

## Comparing representations (CKA)

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
  untrained scores 0.10. Run the comparison on a `NullModelBaseline` too ([Baselines](index.md#baselines)).
- **Outliers move it.** A few inputs with very large activations can dominate, and CKA can be
  pushed up or down on purpose (Davari et al., 2022). Check that a result holds on a second set
  of inputs.
- A representation that is constant over the inputs has no CKA: the result is NaN.

The implementation matches the reference code of Kornblith et al. (ICML 2019) to 12 decimal
places for both estimators.

## RSA and SVCCA

Two older ways to compare representations are in the same header. Use CKA unless you need one
of them.

**RSA** (representational similarity analysis) describes a representation by how far apart it
puts every pair of inputs, its representational dissimilarity matrix (RDM), and correlates two
RDMs. Because only the RDM is compared, the other side can be anything that orders the same
inputs: another model, a brain recording, or a hand-made matrix that says which inputs share a
category.

```cpp
#include "pulsatrix/representation_similarity.hpp"

const Rdm model = RepresentationalDissimilarity(layer, RdmMetric::Correlation);  // 1 − Pearson
const Rdm categories = ...;  // e.g. 0 within a category, 1 across, in the same pair order
const RsaTestResult r = RsaPermutationTest(model, categories, RdmComparison::Spearman, 999);
// r.similarity: Spearman's correlation of the two RDMs; r.p_value: how often relabeled inputs
// agree as well.
```

- **The metrics** are correlation distance (the RSA default), Euclidean and cosine. RDMs are
  stored as scipy's `pdist` stores them, pairs (0, 1), (0, 2), ..., so they can go to or come
  from Python unchanged; `Rdm::at(i, j)` reads one.
- **Compare with Spearman** (the default) when the two RDMs are on different scales, which is
  almost always; only the order of the distances counts. `Rsa(x, y)` does both steps.
- **Test with `RsaPermutationTest()`, not a correlation's p-value.** The distances in an RDM
  aren't independent (one input's row moves n − 1 of them), so the usual p-value is too small.
  The test relabels the inputs of one RDM instead. In the tests, two noisy views of four
  categories agree at 0.83 with p = 1/200, the smallest 199 relabelings can give, and unrelated
  representations reach p < 0.05 in 3 of 40 trials, about the one in 20 they should.

**SVCCA** keeps each representation's top singular directions, 99% of the variance by default,
and finds the pairs of directions, one in each, that correlate most. The mean of those canonical
correlations is the similarity.

```cpp
const SvccaResult r = Svcca(layer_a, layer_b);  // r.similarity, r.correlations (largest first)
const SvccaResult fixed = Svcca(layer_a, layer_b, SvccaOptions{0.99, 20});  // 20 directions each
```

Things to know:

- **It is biased upwards, more than CKA.** Canonical correlations are fitted to the inputs. Two
  unrelated 20-dimensional representations score 0.50 with 60 inputs, against −0.02 for
  debiased CKA, and still 0.09 with 2,000. Use many more inputs than kept directions.
- **It refuses too few inputs.** When the kept directions of both add up to the number of inputs
  or more, some correlations are 1 whatever the data, and `Svcca()` throws rather than report
  them. Ask for fewer directions with `SvccaOptions::directions`.
- **It ignores how a layer weights its directions.** CCA is invariant to any invertible linear
  map: stretching one axis of a layer ten times and shearing another leaves SVCCA at 1, and
  drops CKA to 0.48 in the tests. Kornblith et al. found CKA better at matching corresponding layers for this reason.
- `CanonicalCorrelations(x, y)` is plain CCA, on every direction with variance.

RSA matches independent numpy implementations of scipy's `pdist` and `spearmanr` to 12 decimal
places, and SVCCA and CCA one written with numpy's SVD and QR to 9.

## Nearest neighbors

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

## The shape of a layer

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

## PCA projections

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

## UMAP and t-SNE

pulsatrix doesn't run UMAP or t-SNE. It writes a layer's representations in the formats the
tools that do run them read, and reads their coordinates back for its own plots.

```cpp
#include "pulsatrix/embedding_export.hpp"

// layer: (n, width); words and tags: one per input. The directory must exist.
ExportEmbeddings("layer6", layer, {{"word", words}, {"tag", tags}});
// layer6/vectors.npy, layer6/vectors.tsv, layer6/metadata.tsv
```

- **In Python:** `np.load("layer6/vectors.npy")` goes straight into `umap.UMAP().fit_transform`,
  `openTSNE.TSNE().fit` or scikit-learn's `TSNE`. `WriteNpy` and `SerializeNpy` write any tensor
  the same way, float32 with its shape.
- **In the browser, with nothing to install:** load `vectors.tsv` and `metadata.tsv` into the
  Embedding Projector (projector.tensorflow.org), which runs UMAP, t-SNE and PCA and colors
  points by any metadata column.
- **Back in pulsatrix:** save the coordinates with `np.save`, read them with `ReadNpy`, and make
  a projection document, which the [PCA views](#pca-projections) draw:

```cpp
const Tensor xy = ReadNpy("layer6/umap.npy", &backend);  // (n, 2), from Python
const ProjectionDocument doc = MakeProjectionDocument(xy, "umap", words, tags);
std::ofstream("layer6_umap.svg") << RenderProjectionSvg(doc);
```

Things to know:

- **These plots keep neighbors, not distances.** The gaps between clusters, the clusters' sizes
  and their densities carry little meaning, and shapes can come from the settings alone
  (perplexity, `n_neighbors`, `min_dist`). The views say so under every plot whose method is
  `umap`, `tsne`, `t-sne` or `opentsne`. Ask questions about distances of PCA, and about
  neighbors of [nearest neighbors](#nearest-neighbors), which measures them directly.
- **Run them more than once.** Both start from random positions. A cluster that appears in one
  run and not another, or under one perplexity and not another, isn't evidence of anything.
- **Reduce first if the layer is wide.** Running PCA to about 50 dimensions first (`FitPca`,
  `PcaProject`, then export the scores) is standard for t-SNE, and makes both faster.
- **The reader takes what numpy writes for float arrays:** float32 or float64, C or Fortran
  order, format versions 1 to 3. It rejects anything else, including pickled objects, with an
  error rather than a guess, and is fuzzed (`tools/fuzz/npy_fuzz.cpp`).

Full API reference: [Doxygen: Mechanistic Interpretability](../api/group__mech__interp.html)
