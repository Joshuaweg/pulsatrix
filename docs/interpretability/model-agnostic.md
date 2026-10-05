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
- **ICE** (`ice.hpp`): individual conditional expectation. `ComputeIce` draws one curve per
  input instead of PDP's single average, with centered (`centered()`) and derivative
  (`derivative()`) forms. `ComputePartialDependence2D` varies two features at once, and
  `FeatureGrid` builds a grid the way scikit-learn does. `ComputeAle` gives accumulated local
  effects, PDP's replacement when features are correlated.
- **Local sensitivity** (`sensitivity.hpp`): `ComputeLocalSensitivity` moves each feature, alone,
  to a low and a high value and records the output, the data of a tornado chart. The bounds come
  from `DeltaBounds` (±δ), `ScaledDeltaBounds` (± some standard deviations of a background set)
  or `RangeBounds` (a background set's percentiles). `Occlusion` slides a window over the input,
  as Captum's `Occlusion` does.
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
| ICE | Each input, and their average | Whether the feature's effect differs between inputs (an interaction) | Reads the model at unrealistic inputs when features are correlated |
| 2-D partial dependence | Whole dataset | How two features interact | grid_x × grid_y × background model calls |
| ALE | Whole dataset | A feature's average effect when it is correlated with others | Bins with few instances are noisy |
| Local sensitivity | One input | How far each feature, alone, can move the output | One feature at a time: misses joint effects |
| `Occlusion` | One input | Which regions of an image or spans of a sequence the output depends on | The baseline value is a choice; results change with it and with the window |

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

### ICE

```cpp
#include "pulsatrix/ice.hpp"
#include "pulsatrix/viz/svg.hpp"

std::vector<float> grid = FeatureGrid(background, /*feature_index=*/0, /*grid_size=*/20);
IceResult ice = ComputeIce(predict, background, /*feature_index=*/0, /*target_index=*/0, grid);
std::vector<float> average = ice.partial_dependence();  // the PDP curve
std::vector<float> centered = ice.centered();           // each curve minus its first value
std::vector<float> slopes = ice.derivative();           // each curve's slope

PartialDependenceDocument doc = ToPartialDependenceDocument(ice, "age", "risk");
PartialDependenceSvgOptions view;
view.style = IceStyle::Centered;
std::string svg = RenderPartialDependenceSvg(doc, view);
```

**What's happening:** `ComputeIce` sets feature 0 to each grid value in each background input,
one model call per pair, and keeps every curve. Their mean is exactly what `PDP` returns.
Curves that are parallel mean the feature acts the same way everywhere. When the centered curves
fan out, or the slopes differ at the same grid point, the feature interacts with another one.
`FeatureGrid` uses the feature's distinct values when there are fewer than `grid_size`, otherwise
an even grid between its 5th and 95th percentiles, as scikit-learn's `partial_dependence` does.

For two features, `ComputePartialDependence2D(predict, background, fx, fy, target, grid_x,
grid_y)` returns a grid of averages; `ToHeatmapDocument` turns it into a heatmap document for
`RenderHeatmapSvg`.

Test reference: `tests/ice_test.cpp`, checked against scikit-learn 1.9.1
(`tools/generate_ice_reference_values.py`).

### ALE

```cpp
AleResult ale = ComputeAle(predict, background, /*feature_index=*/0, /*target_index=*/0, /*num_bins=*/20);
// ale.edges: bin edges; ale.effects: the centered effect at each edge; ale.counts: per bin
std::string svg = RenderPartialDependenceSvg(ToPartialDependenceDocument(ale, "age", "risk"));
```

**What's happening:** PDP sets a feature to every grid value in every background input, so when
features are correlated it asks the model about inputs that never occur (a 20-year-old with 30
years of work experience). ALE cuts the feature's range into bins at its quantiles, moves each
input only to its own bin's two edges, averages those local differences per bin, and sums them
from the left. The curve is centered so its count-weighted mean is zero; read it as "how much
this value raises or lowers the prediction compared with the average". It matches PyALE 1.2.0
and R's ALEPlot, including their quantile and binning conventions.

### Local sensitivity and occlusion

```cpp
#include "pulsatrix/sensitivity.hpp"
#include "pulsatrix/viz/svg.hpp"

// Each feature from its 5th to its 95th percentile over the background, one at a time.
LocalSensitivityResult s = ComputeLocalSensitivity(predict, input, /*target_index=*/0,
                                                   RangeBounds(background));
for (const FeatureSensitivity& f : s.features) {
    // f.output_low, f.output_high, f.swing(), f.slope()
}
std::string tornado = RenderTornadoSvg(ToSensitivityDocument(s, {"age", "dose", "weight"}, "risk"));

// Occlusion of 3x3 patches of a (1, H, W) image, stride 1, replaced with 0.
Attribution occ = Occlusion(predict, image, /*target_index=*/7, /*window=*/{1, 3, 3},
                            /*strides=*/{1, 1, 1}, /*baseline=*/0.0f);
```

**What's happening:** `ComputeLocalSensitivity` makes 1 + 2 × features model calls. A gradient
describes the model in an infinitesimal neighborhood; this measures a finite one, so the two
disagree where the model saturates or has a kink, and `slope()` shows by how much. `Occlusion`
gives each element the drop in the output when a window covering it is replaced by `baseline`,
averaged over the windows that covered it. The last window in a dimension is cropped at the
edge, as in Captum. For a (1, H, W) image, `ToSaliencyHeatmap(occ)` and `ToHeatmapDocument`
draw it with `RenderHeatmapSvg`; a window that spans all channels gives every channel the same
value, so read one.

Test reference: `tests/sensitivity_test.cpp`, checked against Captum 0.9.0 and numpy
(`tools/generate_sensitivity_reference_values.py`).

## Recipes

- [KernelSHAP basics](../recipes/interpretability/kernel_shap_basics.md)
- [LIME basics](../recipes/interpretability/lime_basics.md)
