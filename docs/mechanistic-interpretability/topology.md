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

This page grows with each stage. So far it covers the first.

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

Full API reference: [Doxygen: Topological Data Analysis](../api/group__tda.html)
