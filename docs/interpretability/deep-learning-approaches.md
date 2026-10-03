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

## Recipes

- [Saliency and Integrated Gradients](../recipes/interpretability/saliency_and_integrated_gradients.md)
- [Grad-CAM walkthrough](../recipes/interpretability/grad_cam_walkthrough.md)
