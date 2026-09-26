# Ad-hoc Interpretability — Deep Learning Approaches

These explainers use gradients or cached activations from the model's real
`ComputationGraph`, via `ExplainerContext` (`explainer_context.hpp`) — the shared harness
that wraps a `Module`'s forward/backward pass and exposes per-node activations/gradients
without each explainer needing its own graph-walking code.

## What's inside

- **`Saliency`** (`saliency.hpp`) — raw gradient of a target output w.r.t. the input.
- **`IntegratedGradients`** (`integrated_gradients.hpp`) — path-integrated gradient from a
  baseline to the input, addressing saliency's gradient-saturation blind spot.
- **`GradCAM`** (`grad_cam.hpp`) — gradient-weighted class activation mapping over the last
  `Conv2DModule`'s feature maps.

Full API reference: [Doxygen: Deep Learning Approaches](../api/group__interpretability__dl.html)

LRP (Layer-wise Relevance Propagation) is the other deep-learning-native explanation method
in pulsatrix, but it isn't a separate class here — every LRP-capable layer implements
`Module::propagate_relevance()` directly (see
[Deep Learning Modules and Layers](../deep-learning/index.md)), configured via
`LRPRuleConfig` (`lrp_rule_config.hpp`).

## How to implement

### Saliency

```cpp
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/saliency.hpp"

using namespace pulsatrix;

ExplainerContext ctx(&my_network, &backend);
Tensor input(Shape({1, 2}), &backend, {1.0f, 0.0f});

Saliency saliency;
Attribution result = saliency.explain(ctx, input, /*target_index=*/0, &backend);
// result.values: d(output[target_index]) / d(input)
```

**What's happening:** `explain()` runs `ctx.forward_pass(input)`, seeds a one-hot vector at
`target_index` for every row in the batch, and calls `ctx.backward_pass(seed)` — the
gradient that comes back *is* the saliency map. Requires a rank-2 `(N, num_classes)` network
output.

Recipe: [Saliency and Integrated Gradients](../recipes/interpretability/saliency_and_integrated_gradients.md).

### Grad-CAM

```cpp
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/grad_cam.hpp"

using namespace pulsatrix;

ExplainerContext ctx(&my_conv_network, &backend);  // must contain at least one Conv2DModule
Tensor input(Shape({1, 1, 28, 28}), &backend, /* ... */ {});

GradCAM grad_cam;
Attribution result = grad_cam.explain(ctx, input, /*target_index=*/3, &backend);
// result.values: N x H x W class activation map, sized to the target conv layer's feature maps
```

**What's happening:** `explain()` runs forward + backward exactly like `Saliency`, then finds
the *last* `OpType::Conv` node in the graph (`ctx.graph().nodes_by_op_type(OpType::Conv)`),
reads its cached activation and gradient, averages the gradient spatially per channel to get
per-channel weights α꜀ₖ, and computes `ReLU(Σ αₖ · Aᵏ)`.

Recipe: [Grad-CAM walkthrough](../recipes/interpretability/grad_cam_walkthrough.md).

## Recipes

- [Saliency and Integrated Gradients](../recipes/interpretability/saliency_and_integrated_gradients.md)
- [Grad-CAM walkthrough](../recipes/interpretability/grad_cam_walkthrough.md)
