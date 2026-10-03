# Interpretability

These are post-hoc explanation methods: you apply them to an already-trained model to see
which input features drove a prediction. Use them to debug a model, check that it relies on
sensible features, or show a user why it made a decision.

This section has three parts:

- **[Layer-wise Relevance Propagation (LRP)](lrp.md)**: passes the prediction's score back
  through the network, layer by layer, using a rule chosen for each layer. Every pulsatrix
  layer implements an LRP rule, so any network you build can be explained with it.
- **[Model-Agnostic Explainers](model-agnostic.md)**: `KernelSHAP`, `LIME` and `PDP`. They only
  call the model, perturbing the input and watching the output. Use them for any model,
  including ones not built with pulsatrix.
- **[Gradient-Based Explainers](deep-learning-approaches.md)**: `Saliency`,
  `IntegratedGradients` and `GradCAM`. They read a pulsatrix network's gradients and
  activations, so they are faster and see inside the model.

Full API reference: [Doxygen: Interpretability](../api/group__interpretability.html)

## Shared infrastructure

- **`Attribution`** (`include/pulsatrix/attribution.hpp`): the result type every explainer
  returns, LRP included. It holds the attribution `values`, the `method` name and a `metadata`
  map, so downstream code needs only one code path for all explainers.
- **`ExplainerContext`** (`include/pulsatrix/explainer_context.hpp`): wraps your network's
  modules, given as a `std::vector<Module*>` in forward order. It runs traced forward,
  backward and relevance passes and exposes per-node activations and gradients. LRP and
  the gradient-based explainers take it as their first argument.

## Recipes

New here? Start with the
[Saliency and Integrated Gradients recipe](../recipes/interpretability/saliency_and_integrated_gradients.md).
See the full recipe lists on [LRP](lrp.md#see-also),
[Model-Agnostic Explainers](model-agnostic.md#recipes) and
[Gradient-Based Explainers](deep-learning-approaches.md#recipes).
