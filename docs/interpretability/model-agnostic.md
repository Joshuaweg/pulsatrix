# Ad-hoc Interpretability — Model-Agnostic

These explainers treat the model as a black box: they only call it as
`std::function<Tensor(const Tensor&)>`, never touch its `ComputationGraph` or gradients, and
so work with any callable — including non-pulsatrix models wrapped in a lambda. That's what
"model-agnostic" means by construction here, not just in intent.

## What's inside

- **`KernelSHAP`** (`kernel_shap.hpp`) — Shapley value approximation via full 2ⁿ coalition
  enumeration + SHAP-kernel-weighted linear regression (exact at this codebase's small
  feature counts, not a sampling approximation).
- **`LIME`** (`lime.hpp`) — fits a locality-weighted linear surrogate around one input by
  Gaussian perturbation.
- **`PDP`** (`pdp.hpp`) — Partial Dependence Plots: how the prediction changes as one
  feature varies, averaged over the rest.
- **`fit_weighted_linear_regression`** (`weighted_linear_regression.hpp`) — the shared ridge
  regression solver both `KernelSHAP` and `LIME` reduce to.

Full API reference: [Doxygen: Model-Agnostic](../api/group__interpretability__agnostic.html)

## How to implement

### KernelSHAP

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/kernel_shap.hpp"

using namespace pulsatrix;

CPUBackend backend;
auto predict = [&](const Tensor& x) { return my_network.forward(x); };

Tensor input(Shape({3}), &backend, {1.0f, 0.5f, -1.0f});
Tensor baseline(Shape({3}), &backend, {0.0f, 0.0f, 0.0f});  // "feature absent" reference

KernelSHAP shap;
Attribution result = shap.explain(predict, input, baseline, /*target_index=*/0, &backend);
// result.values: per-feature Shapley values, same shape as input
```

**What's happening:** `explain()` enumerates every coalition of "present"/"baseline" features
(2ⁿ − 2 of them, excluding the empty and full coalitions, which are pinned exactly), weights
each by the SHAP kernel π(z′), and fits the reduced (n−1)-dimensional regression problem the
efficiency axiom produces via `fit_weighted_linear_regression`. Capped at `n ≤ 20` features —
full enumeration, not a sampling scheme, so it doesn't scale past that.

Test reference: `tests/kernel_shap_test.cpp`. Recipe:
[KernelSHAP basics](../recipes/interpretability/kernel_shap_basics.md).

### LIME

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lime.hpp"

using namespace pulsatrix;

CPUBackend backend;
auto predict = [&](const Tensor& x) { return my_network.forward(x); };

Tensor input(Shape({3}), &backend, {1.0f, 0.5f, -1.0f});

LIME lime;
Attribution result = lime.explain(predict, input, /*target_index=*/0,
                                   /*num_samples=*/500, /*sigma=*/0.25f,
                                   /*l2_lambda=*/0.01f, /*seed=*/42, &backend);
// result.values: local linear coefficients, same shape as input
```

**What's happening:** `explain()` draws `num_samples` Gaussian perturbations of `input`
(std. dev. `sigma`), weights each by an exponential locality kernel of the same `sigma`, and
fits a ridge-regularized local linear model `g(z) = f(x) + wᵀ(z − x)` — so `w` (the
attribution) is the model's local sensitivity around `x`, not a global explanation.

Test reference: `tests/lime_test.cpp`. Recipe:
[LIME basics](../recipes/interpretability/lime_basics.md).

## Recipes

- [KernelSHAP basics](../recipes/interpretability/kernel_shap_basics.md)
- [LIME basics](../recipes/interpretability/lime_basics.md)
