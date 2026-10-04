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

## Checking an explanation

An explanation can look plausible and still not reflect the model. `explanation_metrics.hpp`
measures this, with metrics from the Quantus families. They work with any explainer:

- **Faithfulness: `DeletionCurve`, `InsertionCurve`.** Remove the most relevant features first
  (deletion) or restore them first (insertion), and record the prediction. A faithful explanation
  makes the score fall fast (a low deletion AUC) or rise fast (a high insertion AUC).
- **ROAD imputation (`Imputation::NoisyLinear`, Rong et al. 2022).** Use it for images. Filling
  removed pixels with a constant leaks which pixels were removed through the shape of the hole;
  ROAD fills them from their neighbors, plus a little noise.
- **Randomization: `ModelParameterRandomizationTest`** (Adebayo et al., "Sanity Checks for
  Saliency Maps"). It randomizes the model's layers from the output down and re-explains. An
  explanation that barely changes isn't explaining the model. The model is restored afterwards.
- **Complexity: `Sparseness` (Gini index) and `Complexity` (entropy).** These measure how
  concentrated the relevance is.

```cpp
#include "pulsatrix/explanation_metrics.hpp"

PredictFn predict = [&](const Tensor& x) { return model.forward(x); };
PerturbationOptions options;
options.target = 3;  // the class being explained
options.imputation = Imputation::NoisyLinear();
float deletion_auc = DeletionCurve(predict, image, attribution, options).auc;

ExplainFn explain = [&](const Tensor& x) { return LRP().explain(ctx, x, 3, &backend); };
RandomizationResult sanity = ModelParameterRandomizationTest(model, explain, image, /*seed=*/0);
// sanity.similarity[i]: rank correlation with the original after randomizing layers[0..i]
```

## Recipes

New here? Start with the
[Saliency and Integrated Gradients recipe](../recipes/interpretability/saliency_and_integrated_gradients.md).
See the full recipe lists on [LRP](lrp.md#see-also),
[Model-Agnostic Explainers](model-agnostic.md#recipes) and
[Gradient-Based Explainers](deep-learning-approaches.md#recipes).
