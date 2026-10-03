# Model-Agnostic Explainers

These explainers treat the model as a black box. They only call it as a
`std::function<Tensor(const Tensor&)>`: change the input, watch the output. They never touch
its `ComputationGraph` or gradients, so they work with any callable, including non-pulsatrix
models wrapped in a lambda.

## What's inside

- **`KernelSHAP`** (`kernel_shap.hpp`): computes Shapley values, each feature's fair share of
  the prediction. It enumerates all 2ⁿ feature coalitions and fits a SHAP-kernel-weighted
  linear regression. The result is exact (no sampling), so it is limited to 20 features.
- **`LIME`** (`lime.hpp`): fits a simple linear model around one input, using random Gaussian
  perturbations weighted by how close they stay to that input.
- **`PDP`** (`pdp.hpp`): a Partial Dependence Plot. It shows how the prediction changes as one
  feature varies, averaged over a set of background inputs.
- **`fit_weighted_linear_regression`** (`weighted_linear_regression.hpp`): the shared ridge
  regression solver that `KernelSHAP` and `LIME` both use. It returns one coefficient per
  feature and fits no intercept.

Full API reference: [Doxygen: Model-Agnostic](../api/group__interpretability__agnostic.html)

## When to use which

| Method | Scope | Good for | Watch out for |
|---|---|---|---|
| `KernelSHAP` | One input | Exact additive attributions that sum to `f(input) - f(baseline)` | 2ⁿ model calls; at most 20 features |
| `LIME` | One input | A fast local linear approximation | Random: fix `seed`; results depend on `sigma` |
| `PDP` | Whole dataset | The average effect of one feature on the output | Hides interactions between features |

## How to implement

The snippets below share this setup. Any callable that maps a `Tensor` to a `Tensor` works as
`predict`; here it wraps a single pulsatrix layer.

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"

using namespace pulsatrix;

CPUBackend backend;
LinearModule model(3, 2, &backend);  // stand-in for your model: 3 features in, 2 scores out
auto predict = [&](const Tensor& x) { return model.forward(x); };

Tensor input(Shape({1, 3}), &backend, {1.0f, 0.5f, -1.0f});
```

`target_index` is a flat index into the model's output: which output value to explain.

### KernelSHAP

```cpp
#include "pulsatrix/kernel_shap.hpp"

Tensor baseline(Shape({1, 3}), &backend, {0.0f, 0.0f, 0.0f});  // the "feature absent" reference

KernelSHAP shap;
Attribution result = shap.explain(predict, input, baseline, /*target_index=*/0, &backend);
// result.values: per-feature Shapley values, same shape as input
```

**What's happening:** a coalition is a set of features kept at their `input` values, with the
rest set to `baseline`. `explain()` evaluates every coalition. The empty and full coalitions are
pinned exactly, and the other 2ⁿ − 2 are weighted by the SHAP kernel π(z′). It then solves the
resulting weighted regression with `fit_weighted_linear_regression`. The input must have
between 1 and 20 features, or `explain()` throws `std::invalid_argument`.

Test reference: `tests/kernel_shap_test.cpp`. Recipe:
[KernelSHAP basics](../recipes/interpretability/kernel_shap_basics.md).

### LIME

```cpp
#include "pulsatrix/lime.hpp"

LIME lime;
Attribution result = lime.explain(predict, input, /*target_index=*/0,
                                  /*num_samples=*/500, /*sigma=*/0.25f,
                                  /*l2_lambda=*/0.01f, /*seed=*/42, &backend);
// result.values: local linear coefficients, same shape as input
```

**What's happening:** `explain()` draws `num_samples` Gaussian perturbations of `input` with
standard deviation `sigma`. It weights each one by a Gaussian locality kernel of the same
width. It then fits a ridge-regularized linear model `g(z) = f(x) + wᵀ(z − x)`. The coefficients
`w` are the attribution: the model's local sensitivity around `x`, not a global explanation.

Test reference: `tests/lime_test.cpp`. Recipe:
[LIME basics](../recipes/interpretability/lime_basics.md).

### PDP

```cpp
#include "pulsatrix/pdp.hpp"

std::vector<Tensor> background = {  // reference inputs to average over, all the same shape
    Tensor(Shape({1, 3}), &backend, {0.0f, 1.0f, -0.5f}),
    Tensor(Shape({1, 3}), &backend, {0.5f, -1.0f, 0.0f}),
};

PDP pdp;
Attribution curve = pdp.explain(predict, background, /*feature_index=*/0, /*target_index=*/0,
                                /*grid_min=*/-2.0f, /*grid_max=*/2.0f, /*grid_size=*/21, &backend);
// curve.values: (grid_size,) average prediction as feature 0 sweeps from grid_min to grid_max
```

**What's happening:** for each of `grid_size` evenly spaced values, `explain()` sets
`feature_index` to that value in every background input, runs the model, and averages the
`target_index` output. Unlike the other explainers, this is a global view of the model, not an
explanation of one input. `background` must not be empty.

Test reference: `tests/pdp_test.cpp`.

## Recipes

- [KernelSHAP basics](../recipes/interpretability/kernel_shap_basics.md)
- [LIME basics](../recipes/interpretability/lime_basics.md)
