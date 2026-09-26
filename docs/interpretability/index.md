# Ad-hoc Interpretability

"Ad-hoc" here means *post-hoc*: these are explanation methods applied to an already-trained
model from the outside, as opposed to the LRP (Layer-wise Relevance Propagation) rule every
layer in [Deep Learning Modules and Layers](../deep-learning/index.md) carries as part of
its own `propagate_relevance()` implementation. Pulsatrix ships both, because they answer
different questions: LRP explains *this specific forward pass, exactly*, using the model's
real computation graph; the explainers here approximate an explanation using only a model's
input/output behavior or its gradients, which generalizes to any model but trades off exact
faithfulness for that generality.

This section splits into two families:

- **[Model-Agnostic](model-agnostic.md)** — perturb-and-observe methods that only need a
  callable model (no gradients, no internal access): `KernelSHAP`, `LIME`, `PDP`, backed by
  `WeightedLinearRegression`.
- **[Deep Learning Approaches](deep-learning-approaches.md)** — methods that use gradients
  or cached activations directly: `Saliency`, `IntegratedGradients`, `GradCAM`.

Full API reference: [Doxygen: Ad-hoc Interpretability](../api/group__interpretability.html)

## Shared infrastructure

Both families return the same result type, `Attribution` (`include/pulsatrix/attribution.hpp`)
— values, method name, and metadata together, so downstream code doesn't need a different
code path per explainer. `ExplainerContext` (`include/pulsatrix/explainer_context.hpp`) wires
a `Module`/`ComputationGraph` pair into whichever explainer needs gradient or activation
access, keeping that plumbing out of each explainer's own implementation.

## Recipes

See the recipe lists on [Model-Agnostic](model-agnostic.md#recipes) and
[Deep Learning Approaches](deep-learning-approaches.md#recipes).
