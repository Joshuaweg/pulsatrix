# Gradient-Based Explainers

These explainers look inside a pulsatrix network to show which input features drove one
prediction. They read its gradients and cached activations. Use them when the model is built
from pulsatrix `Module`s. For a black-box model, use the
[model-agnostic explainers](model-agnostic.md) instead.

All of them run through `ExplainerContext` (`explainer_context.hpp`). It wraps your network's
modules, given as a `std::vector<Module*>` in forward order. It runs traced forward, backward
and relevance passes and exposes per-node activations and gradients, so each explainer doesn't
need its own graph-walking code.

## What's inside

- **`Saliency`** (`saliency.hpp`): the raw gradient of a target output with respect to the input.
- **`IntegratedGradients`** (`integrated_gradients.hpp`): gradients averaged along a straight
  path from a baseline input to your input. This avoids saliency's blind spot where the
  output is flat (saturated).
- **`GradCAM`** (`grad_cam.hpp`): gradient-weighted class activation mapping. It produces a
  coarse heatmap over the last `Conv2DModule`'s feature maps.
- **`FindCounterfactual`** (`counterfactual.hpp`): the nearest input that reaches a different
  class or an output range (Wachter et al. 2017), with immutable features, limits, and integer
  and one-hot features. It answers "what would have to change?" `FindDiverseCounterfactuals`
  finds several that take different routes (DiCE), and `Plausibility` scores how typical a
  counterfactual is of real data.
- **`ComputeAttributionStability(runs)`** (`explainer_stability.hpp`): measures how much
  repeated runs of any explainer vary on the same input.

[LRP](lrp.md) (Layer-wise Relevance Propagation) also runs through `ExplainerContext`, but
propagates relevance with per-layer rules instead of gradients. It has its own page.

Every explainer returns an `Attribution` and requires a rank-2 `(N, num_classes)` network output.

Full API reference: [Doxygen: Gradient-Based Explainers](../api/group__interpretability__dl.html)

## When to use which

| Method | Cost | Good for | Watch out for |
|---|---|---|---|
| `Saliency` | 1 forward + 1 backward | A quick first look | Noisy; reads zero where the output saturates |
| `IntegratedGradients` | `steps` forward + backward passes | Attributions that add up to `f(input) - f(baseline)` | You must choose a baseline |
| `GradCAM` | 1 forward + 1 backward | "Where in the image?" for CNNs | Coarse: feature-map resolution, not pixels |
| `FindDiverseCounterfactuals` | `count × max_steps` forward + backward passes | Several different ways to change the outcome | The diversity and proximity weights trade off against each other |
| `FindCounterfactual` | Up to `rounds × steps` forward + backward passes | "What is the smallest change that flips this?" | On images it finds adversarial noise, not a meaningful change |
| [`LRP`](lrp.md) | 1 forward + 1 relevance pass | Per-feature relevance that (approximately) sums to the output score | The rule choice changes the result |

## How to implement

The snippets below share this setup: a small two-layer classifier.

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"

using namespace pulsatrix;

CPUBackend backend;
LinearModule linear1(2, 4, &backend);
ReluModule relu(&backend);
LinearModule linear2(4, 2, &backend);
// ... train or load weights ...

ExplainerContext ctx({&linear1, &relu, &linear2});  // the layers, in forward order
Tensor input(Shape({1, 2}), &backend, {1.0f, 0.0f});  // a batch of one example
```

### Saliency

```cpp
#include "pulsatrix/saliency.hpp"

Saliency saliency;
Attribution result = saliency.explain(ctx, input, /*target_index=*/0, &backend);
// result.values: d(output[target_index]) / d(input), same shape as input
```

**What's happening:** `explain()` runs `ctx.forward_pass(input)`. It then seeds a one-hot
vector at `target_index` for every row in the batch and calls `ctx.backward_pass(seed)`. The
gradient that comes back is the saliency map.

Recipe: [Saliency and Integrated Gradients](../recipes/interpretability/saliency_and_integrated_gradients.md).

### Integrated Gradients

```cpp
#include "pulsatrix/integrated_gradients.hpp"

Tensor baseline(Shape({1, 2}), &backend, {0.0f, 0.0f});  // an "uninformative" reference input

IntegratedGradients ig;
Attribution result = ig.explain(ctx, input, baseline, /*target_index=*/0, /*steps=*/200, &backend);
// result.values: the path-integrated attribution, same shape as input
```

**What's happening:** `explain()` runs `Saliency` at `steps` evenly spaced points between
`baseline` and `input`. It averages those gradients and multiplies by `(input - baseline)`.
The attributions then sum to `f(input) - f(baseline)`, up to an approximation error that
shrinks as `steps` grows (50–300 is typical). Because it collects gradients along the whole
path, it doesn't go blind where the output is flat at the input itself.

Recipe: [Saliency and Integrated Gradients](../recipes/interpretability/saliency_and_integrated_gradients.md).

### Grad-CAM

Grad-CAM needs a network with at least one `Conv2DModule`. This example uses the built-in
MNIST classifier, `MnistConvNet` (Conv2D → ReLU → Flatten → Linear).

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/grad_cam.hpp"
#include "pulsatrix/mnist_classifier_example.hpp"

using namespace pulsatrix;

CPUBackend backend;
MnistConvNet net(&backend);
// ... train the network ...

ExplainerContext ctx(net.modules());
std::vector<float> pixels(28 * 28, 0.0f);  // one 28x28 image, values in [0, 1]
Tensor image(Shape({1, 1, 28, 28}), &backend, pixels);

GradCAM grad_cam;
Attribution result = grad_cam.explain(ctx, image, /*target_index=*/3, &backend);
// result.values: (N, H, W) class activation map, at the conv layer's feature-map size
```

**What's happening:** `explain()` runs forward and backward like `Saliency`. It then finds the
*last* `OpType::Conv` node in the graph and reads that node's activation `A` and gradient. It
averages the gradient over each channel's spatial positions to get a weight `αₖ` per channel.
The map is `ReLU(Σₖ αₖ · Aᵏ)`.

Recipe: [Grad-CAM walkthrough](../recipes/interpretability/grad_cam_walkthrough.md).

### Counterfactuals

```cpp
#include "pulsatrix/counterfactual.hpp"
#include "pulsatrix/viz/svg.hpp"

CounterfactualConstraints c;
c.scale = MedianAbsoluteDeviation(background);  // a change of one MAD costs 1
c.immutable = {0};                                // feature 0 (say, age) may not change
CounterfactualResult cf = FindCounterfactual(ctx, input, CounterfactualTarget::ToClass(1), c);
// cf.valid, cf.counterfactual, cf.distance, cf.num_changed, cf.output_before, cf.output_after

std::string svg = RenderCounterfactualSvg(
    ToCounterfactualDocument(input, cf, {"age", "income"}, c.scale, "class approved"));
```

**What's happening:** the search minimizes `lambda × loss + Σⱼ |x'ⱼ − xⱼ| / scaleⱼ`. The loss is
a hinge that is zero once the target class leads every other output by `margin` (or the output
is inside the target range). Each step is a gradient step on the loss followed by
soft-thresholding toward the input, so features the prediction doesn't need stay exactly
unchanged and the result is sparse. If a round of `steps_per_round` steps ends without a valid
point, `lambda` grows by `lambda_growth` and the search continues. Integer features are rounded
and one-hot groups snapped at the end, and `valid` is judged after that. On a linear model the
result is the L1-optimal change, which the tests check.

**Several routes.** One counterfactual hides the others: raising income and paying down debt may
both work.

```cpp
DiverseCounterfactualOptions opts;
opts.count = 4;
DiverseCounterfactualResult set = FindDiverseCounterfactuals(ctx, input, CounterfactualTarget::ToClass(1), c, opts);
// set.validity, set.diversity, set.count_diversity; set.counterfactuals[i] as above
float typical = Plausibility(set.counterfactuals[0].counterfactual, background, c.scale, /*k=*/5);

std::vector<CounterfactualDocument> docs;
for (const auto& cf : set.counterfactuals) docs.push_back(ToCounterfactualDocument(input, cf, names, c.scale));
std::string svg = RenderCounterfactualSetSvg(docs);
```

The counterfactuals are optimized together: each one's target hinge and distance, minus a
diversity term, the log-determinant of the kernel `1 / (1 + distance)` between them, which pushes
them apart. Afterwards each one's unneeded changes are undone, smallest first, as DiCE does.
`diversity` is their mean pairwise distance and `count_diversity` the mean fraction of features
that differ between pairs. `Plausibility` is the mean scaled distance to the `k` nearest
background instances: a valid counterfactual far from all real data may describe nobody.

Test reference: `tests/counterfactual_test.cpp`.

## Before you explain

- Call `model.set_training(false)` first if the model has BatchNorm or Dropout. Otherwise each
  sample's gradients, and so its explanation, depend on the rest of its batch.
- To save or plot a result, see [Visualization](../visualization/documents.md#json-documents). A
  Grad-CAM map goes through `ToSaliencyHeatmap()`, then `ToHeatmapDocument()` for JSON or
  `RenderHeatmapSvg()` for an SVG figure.

## Recipes

- [Saliency and Integrated Gradients](../recipes/interpretability/saliency_and_integrated_gradients.md)
- [Grad-CAM walkthrough](../recipes/interpretability/grad_cam_walkthrough.md)
