# Topology of Representations

Persistent homology measures the shape of a set of points: its clusters, the loops they form,
and the voids they enclose, and how long each lasts as the scale grows. Applied to a layer's
activations, it asks questions the tools in [Analyzing representations](representations.md)
can't: does a layer arrange the days of the week on a circle, and does that circle survive
noise and a random-network control?

pulsatrix builds this as a pipeline, one stage at a time:

```
PointCloud → distances → filtration → persistence diagram → numbers and plots
```

This page grows with each stage. So far it covers the point cloud, the distances, and the first diagram: H0.

## From activations to a point cloud

`PointCloud` gathers a layer's activations, batch by batch, into one set of points. Each point
remembers the sample and position it came from, so a loop found later can be traced back to the
inputs that form it.

```cpp
#include "pulsatrix/tda/point_cloud.hpp"

PointCloud cloud({PointPooling::PerPosition});  // one point per token
model->set_hidden_state_hook([&](int64_t position, const Tensor& h) {
    if (position == 6) cloud.add(h);             // (1, tokens, width)
    return h;
});
for (const std::string& text : corpus) (void)model->forward(Encode(text));
// cloud.size() points of cloud.dimension() == width.
// cloud.origins()[i].sample is the text, .position the token.
```

How a sample becomes points is the one choice to make:

| `PointPooling` | Points per sample | Use it for |
|---|---|---|
| `Flatten` (the default) | One, every dimension after the batch flattened | An MLP layer, a classifier's logits |
| `PerPosition` | One per token or pixel | The geometry of token representations |
| `Mean`, `Max` | One, pooled over the positions | A summary of an encoder's whole input |
| `Last` | One, the last position | A causal language model's summary of the whole input |

- **The feature axis.** For every pooling but `Flatten`, `PointCloudOptions::feature_axis` says
  which axis holds the features. The default, the last axis, suits a sequence model's
  `(batch, tokens, width)`. Use `1` for a convolution's `(batch, channels, height, width)`;
  its pixels then become the positions, in row-major order.
- **Batches of different lengths.** The first batch fixes the points' dimension. Later batches
  can have any number of positions, so texts of different lengths go in one at a time. Sample
  numbers keep counting across batches.
- **Snapshots.** `cloud.add(snapshot, node)` adds one node of an `ExplainerContext` snapshot.
  `MakePointCloud(snapshots, node)` builds a cloud from several.
- **Checks.** A batch with a NaN or infinity, or of the wrong dimension, throws and adds nothing.
  `Mean` sums in double precision.

`cloud.ToTensor(&backend)` gives the points as an `(n, dimension)` tensor for the stages that
follow.

## Distances, with an error bar

Persistent homology sees the points only through their pairwise distances. `PairwiseDistances()`
computes them as |x|² + |y|² − 2 x·y, with the dot products from one GEMM on the cloud's backend
(CPU, HIP or CUDA):

```cpp
#include "pulsatrix/tda/distance_matrix.hpp"

const DistanceMatrix d = PairwiseDistances(cloud, &backend);
// d.values: the n (n − 1) / 2 distances, in scipy's pdist order; d.at(i, j) reads one.
// d.error_bound: every stored distance is within this of the exact one.
```

**Why the error bound.** In float32 that formula cancels for points close together, relative to
their distance from the origin. The result can be wrong in every digit, and since persistent
homology is all about small distances (when do two points join?), that matters. So every
`DistanceMatrix` carries a certified bound: |stored − exact| ≤ `error_bound` for every pair.

- **It's rigorous.** It comes from the standard error analysis of a float32 dot product
  (at most γ_d ‖x‖ ‖y‖, whatever order the GEMM sums in), plus the double-precision steps and
  the final rounding. It assumes the GEMM accumulates in float32, not TF32 or bf16.
- **It carries through.** Persistence diagrams are stable: a diagram computed from these
  distances is within `error_bound`, in bottleneck distance, of the exact one. Later stages add
  their own errors to it.
- **When it's large, center.** Points sharing a large offset, as in an anisotropic layer, make
  the bound large. `DistanceOptions::center` subtracts the mean first, which leaves the
  distances unchanged. In the tests, points spread by 0.01 per coordinate, sitting 1,000 from the
  origin, get a bound bigger than a tenth of their distances. Centered, it shrinks over a hundredfold, to under a
  thousandth.
- **Or compute exactly.** `DistanceMethod::Exact` sums squared differences in double on the
  host. Its bound is the float32 rounding of the result, but it costs n² × d on the CPU.

Other options:

- `normalize` scales every point to unit length first. The distance is then √(2 − 2 cos), a
  metric that orders pairs as cosine similarity does.
- `block_rows` sets how many rows each GEMM computes. The device holds the points and one block
  at a time; by default a block has at most 2²⁴ entries.

Storing n (n − 1) / 2 float32 values takes 50 MB for 5,000 points. Persistent homology above H0
can't use many more points than that anyway; landmarks (coming) choose which ones to keep.

## Clusters across scales (H0)

Grow a ball around every point. At scale 0 each point is its own component. As the scale grows,
two components merge when the shortest edge between them appears, until one is left. H0
persistence records those merges: one pair (born at 0, dies at the merge scale) per merge, and
one pair that never dies.

That is single-linkage clustering, and the merge scales are the edge lengths of a minimum
spanning tree. `ComputeH0()` finds the tree exactly with Prim's algorithm, in O(n²) time:

```cpp
#include "pulsatrix/tda/h0.hpp"

const H0Result h0 = ComputeH0(PairwiseDistances(cloud, &backend));
// h0.diagram.pairs: (0, merge scale) per merge, then (0, ∞). h0.diagram.error_bound is the
// distances' bound: the diagram is within it of the exact one, in bottleneck distance.
// h0.edges: the tree's n − 1 edges, shortest first; edge k is the k-th merge.
const std::vector<int64_t> clusters = SingleLinkageClusters(h0.edges, cloud.size(), /*threshold=*/0.5);
```

- **Long-lived pairs are clusters.** A pair that dies late means a group that stays apart from
  the rest over a wide range of scales. `SingleLinkageClusters()` cuts the tree at a scale and
  labels the groups. Each edge's endpoints, through `cloud.origins()`, name the inputs a merge
  joins.
- **Duplicate points** give zero-length edges. They stay in `edges` but not in the diagram,
  which leaves out zero-persistence pairs as Ripser does.
- **Ties** in length break by the lower point index, so the tree is the same on every platform.
  Its lengths, which are all the diagram uses, are the same for every minimum tree.

**H0 alone says little.** It is a function of the minimum spanning tree, so it sees roughly what
nearest-neighbor distances see. Many recent papers on the topology of language models use only
H0. Before trusting such a result, check it against plain
[nearest-neighbor statistics](representations.md#nearest-neighbors). Loops and voids (H1, H2)
are where persistent homology sees something new; they come next.

Full API reference: [Doxygen: Topological Data Analysis](../api/group__tda.html)
