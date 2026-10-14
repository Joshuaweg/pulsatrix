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
  perturbations weighted by how close they stay to that input. Its `local_fidelity` says how well
  that linear model fits.
- **Surrogate tree** (`surrogate_tree.hpp`): `FitSurrogateTree` fits a shallow decision tree to
  the model's predictions, as rules a person can read, and measures its fidelity to the model on
  held-out inputs.
- **Anchors** (`anchors.hpp`): `FindAnchor` finds an if-then rule around one input that keeps the
  model's prediction with high estimated precision, and reports how much of the data it covers.
- **Prototypes** (`prototypes.hpp`): `FindPrototypes` returns the background examples nearest
  an input that the model classes the same way (its prototypes) and the nearest it classes
  differently (its contrasts). `ProtoDash` chooses weighted prototypes that summarize a dataset
  or a class.
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
- **Global sensitivity** (`global_sensitivity.hpp`): `ComputeMorris` screens features with
  Morris elementary effects (μ*, σ) and `ComputeSobol` estimates first- and total-order Sobol
  indices, both with bootstrap confidence intervals, over ranges you give for each feature.
  `AnalyzeMorris` and `AnalyzeSobol` analyze samples made elsewhere (for example by SALib).
- **`GrowingSpheresCounterfactual`** (`counterfactual.hpp`): a counterfactual for any prediction
  function, including models with no gradient such as trees, rules and thresholds. It takes the
  same targets and constraints as the gradient search in
  [Gradient-Based Explainers](deep-learning-approaches.md#counterfactuals).
- **`fit_weighted_linear_regression`** (`weighted_linear_regression.hpp`): the shared ridge
  regression solver that `KernelSHAP` and `LIME` both use. It returns one coefficient per
  feature and fits no intercept.

Full API reference: [Doxygen: Model-Agnostic](../api/group__interpretability__agnostic.html)

## When to use which

| Method | Scope | Good for | Watch out for |
|---|---|---|---|
| `KernelSHAP` | One input | Exact additive attributions that sum to `f(input) - f(baseline)` | 2ⁿ model calls; at most 20 features |
| `LIME` | One input | A fast local linear approximation | Random: fix `seed`; results depend on `sigma`; check `local_fidelity` |
| Anchors | One input | Which conditions keep this prediction, as a rule a person can check | Precision is estimated; coverage can be tiny; the rule is only as fine as the percentile cuts |
| Prototypes | One input | "Classed like these, unlike those": examples a person can check | Distances in mixed units need a scale; in high dimensions, an embedding |
| ProtoDash | A dataset or a class | A few weighted examples that stand for the whole | The kernel width sets what counts as alike |
| Surrogate tree | Whole dataset | The model's overall logic, as if/else rules | Only as good as its held-out fidelity; a readable tree can't imitate a complicated model |
| `PDP` | Whole dataset | The average effect of one feature on the output | Hides interactions between features |
| ICE | Each input, and their average | Whether the feature's effect differs between inputs (an interaction) | Reads the model at unrealistic inputs when features are correlated |
| 2-D partial dependence | Whole dataset | How two features interact | grid_x × grid_y × background model calls |
| ALE | Whole dataset | A feature's average effect when it is correlated with others | Bins with few instances are noisy |
| Local sensitivity | One input | How far each feature, alone, can move the output | One feature at a time: misses joint effects |
| Morris | The input space | Ranking many features cheaply, and spotting nonlinear or interacting ones | Screening, not a variance decomposition |
| Sobol | The input space | What share of the output's variance each feature explains, alone and with interactions | (features + 2) × N model calls; assumes independent features |
| `GrowingSpheresCounterfactual` | One input | "What would have to change?" for a model with no gradient | Samples in the input space; pass a scale when features have different units |
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

**Local fidelity.** `result.metadata["local_fidelity"]` is the locality-weighted R² of the
linear model on the perturbed samples, the `score` the reference LIME implementation reports.
Near 1, the coefficients describe the model around `x`. Near 0 or below, they don't, and the
model is not linear at this scale. At the bottom of a bowl, f(z) = |z|², every slope is zero, and
LIME's coefficients are noise; the score there is negative. It can go below 0 because the linear
model has no intercept: it passes through f(x), so samples that all sit on one side of f(x) fit
worse than their mean does. A smaller `sigma` usually fits better. An `Explanation` report (XAI-3)
warns below 0.8.

Test reference: `tests/lime_test.cpp`. Recipe:
[LIME basics](../recipes/interpretability/lime_basics.md).

### Global surrogate tree

```cpp
#include "pulsatrix/surrogate_tree.hpp"

SurrogateTreeOptions options;
options.max_depth = 2;      // three or four levels is the most a person reads
SurrogateTreeResult s = FitSurrogateTree(predict, background, options);
std::cout << DescribeSurrogateTree(s.tree, {"income", "debt", "age"});
std::cout << "held-out agreement: " << s.fidelity.test << "\n";
```

**What's happening:** `FitSurrogateTree` labels each background input with the model's class
(the index of its largest output) or, with `SurrogateTask::Regression`, one output's value. It
holds out 30% of the inputs, fits a CART decision tree to the rest, and measures the tree's
**fidelity** on both parts:
- for classes, the share of inputs where the tree and the model agree;
- for values, R².

The tree is grown exactly as scikit-learn's `DecisionTreeClassifier` and
`DecisionTreeRegressor` grow it, with Gini or squared-error splits. Ties between features go to
the lower index, where scikit-learn breaks them at random. `FeatureImportance` gives each
feature's share of the impurity decrease, as `feature_importances_` does.

On a model whose class is set by two thresholds, income > 0.2 and debt > −0.1, a depth-2 tree
finds both:

```
if income <= 0.2052:
  if debt <= -0.1038:
    class 0 (62 inputs, 100%)
  else:
    class 2 (95 inputs, 100%)
else:
  if debt <= -0.112:
    class 1 (49 inputs, 100%)
  else:
    class 3 (74 inputs, 100%)
```

It agrees with the model on 99% of the held-out inputs. Age never appears, and its importance is 0.

**Read the fidelity before the rules.** A tree that agrees with the model on 70% of new inputs
describes a different model the other 30% of the time. Below about 85% at a depth a person can
read, the rules aren't a description of the model (Craven and Shavlik, 1996). Fitted to
sin(4x₀) + cos(3x₁)·x₂, the held-out R² grows with depth:

| Depth | Leaves | Held-out R² |
|---|---|---|
| 1 | 2 | 0.24 |
| 2 | 4 | 0.46 |
| 3 | 8 | 0.57 |
| 4 | 15 | 0.78 |
| 8 | 61 | 0.84 |

No depth anyone could read reaches 0.85. Here PDP, ALE or Sobol indices tell you more than the
rules do. The planner (XAI-2) lists the held-out fidelity as the tree's own check.

`ToJson(s, names)` writes a `pulsatrix.surrogate_tree.v1` document, with every node, both
fidelities and the importances. `FitDecisionTree` fits the same tree to rows and targets you
already have.

Test reference: `tests/surrogate_tree_test.cpp`, checked node by node against scikit-learn 1.9.1
(`tools/generate_surrogate_tree_reference_values.py`).

#### The tree as Datalog rules

`ExtractRules` turns a classification tree into a Datalog program, one rule per leaf, that the
library's Datalog engine runs. Rules can then be combined with other knowledge, or explained
with the engine's own LRP:

```cpp
#include "pulsatrix/rule_extraction.hpp"

TreeRules rules = ExtractRules(s.tree, {"length", "width"}, {"a", "b", "c"});
for (const ExtractedRule& r : rules.rules) std::cout << FormatRule(r.rule) << "\n";
std::vector<int64_t> classes = ClassifyWithRules(rules, inputs);  // the engine's answer, input by input
```

For the tree "length ≤ 0.5? (length ≤ 0.25? a : (width ≤ 0.3? b : a)) : c" (the one in
`tests/rule_extraction_test.cpp`) this prints:

```
predict(S, a) :- sample(S), le(S, length, 0.25).
predict(S, b) :- sample(S), le(S, length, 0.5), gt(S, length, 0.25), le(S, width, 0.3).
predict(S, a) :- sample(S), le(S, length, 0.5), gt(S, length, 0.25), gt(S, width, 0.3).
predict(S, c) :- sample(S), gt(S, length, 0.5).
```

- **Facts:** `rules.Facts("s1", x)` writes the facts for one input: `sample(s1)`, and `le` or
  `gt` for every threshold the rules test. Datalog's constants are strings, so thresholds are
  written with 9 significant digits, in the rules and the facts alike.
- **Merging:** two tests in the same direction on one feature merge into the tighter one, as in
  the first rule, where length ≤ 0.5 and length ≤ 0.25 become length ≤ 0.25.
- **Each rule records:** the leaf's training support, and its confidence, the share of that
  support in the rule's class. `ToJson` writes them all as a `pulsatrix.tree_rules.v1` document.
- **Checked:** on a fitted depth-5 tree, the engine derives exactly one `predict` fact for each
  of 300 inputs, and it is always the tree's class.

A `SoftDecisionTree` (see [Knowledge distillation](../deep-learning/index.md#knowledge-distillation))
gives weighted rules instead:

```
predict(S, C) :- right(S, n0), left(S, n2), right(S, n5), leaf(l5, C).
```

- **The facts:** `right(s, n)` carries the probability of going right at node n, `left(s, n)` its
  complement, and `leaf(l, c)` leaf l's probability of class c.
- **The result:** over the real semiring, the engine sums the product of the weights over every
  derivation. So the derived `predict(s, c)` is exactly the tree's probability of class c
  (`ClassProbabilitiesFromRules`), within 1e-12 in the tests.

### Anchors

```cpp
#include "pulsatrix/anchors.hpp"

Anchor a = FindAnchor(predict, input, background);
std::cout << DescribeAnchor(a, {"income", "debt", "age"}) << "\n";
// IF income > -0.02925 AND debt > 0.07842 THEN class 1 (precision 1, coverage 0.258)
```

**What's happening:** an anchor (Ribeiro, Singh and Guestrin, AAAI 2018) is a rule that keeps the
prediction whatever the other features do. Every feature is cut at the background's quartiles,
and every half-line at a cut that contains the input is a candidate predicate: `x > cut` below
the input, `x <= cut` above it, as in the reference implementation. To test a rule,
`FindAnchor` samples inputs that satisfy it: each sample takes a random background row and gives
each anchored feature a background value inside its interval. A beam search adds one predicate at
a time. KL-LUCB picks the most precise candidates with few model calls. Each one is then sampled
until its precision is settled either way. Among the rules that reach `precision_threshold`
(0.95), the one that covers the most background data wins.

The rule above is for a model whose class is 1 when income > 0 and debt > 0. The input is
(0.9, 0.9, 0). The anchor found the model's own rule, cut at the medians (−0.03 and 0.08), and it
covers a quarter of the data. For an input with income −0.9 the anchor is
`IF income <= -0.02925 THEN class 0`: one low feature is enough. When no rule reaches the
threshold, `meets_threshold` is false and the result is the most precise rule found. For
example, when the class flips with sin(40 · Σx), the best rule reaches a precision of 0.54.

**Read the precision bound and the coverage, not just the rule.** Precision is estimated from
samples: `precision_lower` is its bound at confidence `1 - delta`. Coverage can be a sliver of the
data, and a rule that applies to 2% of inputs says little about the model.

**Stricter than the reference.** The reference implementation accepts a rule when the lower
bound is within `epsilon_stop = 0.05` of the threshold. On a model where `x0 > median` keeps the
class for 92.5% of inputs, that acceptance let sampling noise lift `x0 > median` plus an
irrelevant predicate just over 0.95. That rule covers more data, so it won. Over 50 seeds,
22 accepted rules had a true precision below 0.95. Pulsatrix's default is `epsilon_stop = 0`:
all 50 seeds then return the right one-predicate rule, at about 2.5 times the model calls. Set
0.05 to match the reference.

The samples draw each anchored feature independently, so a rule over correlated features is
tested on some inputs the data never has. The reference implementation does this too, when too
few rows satisfy every predicate at once. `ToJson` writes a `pulsatrix.anchor.v1` document.

Test reference: `tests/anchors_test.cpp`.

### Prototypes and ProtoDash

```cpp
#include "pulsatrix/prototypes.hpp"

PrototypeOptions options;
options.k = 3;
options.scale = MedianAbsoluteDeviation(background);   // tabular features in mixed units
PrototypeResult r = FindPrototypes(predict, input, background, options);
// r.prototypes: the nearest examples with the input's class; r.contrasts: the nearest without

ProtoDashOptions summary;
summary.count = 5;
ProtoDashResult p = ProtoDash(background, background, summary);   // p.indices, p.weights
```

**What's happening:** `FindPrototypes` classes every background example with the model, ranks
them by distance to the input, and keeps the k nearest on each side of the decision. Distances
are in the input space with a per-feature scale, or in any representation you give as `embed`,
such as a layer's activations.

`ProtoDash` (Gurumoorthy et al., ICDM 2019) chooses prototypes from a source set whose weighted
Gaussian-kernel mean matches a target set's, an MMD criterion. It is greedy:
- each step adds the point with the largest gradient of wᵀμ − ½wᵀKw;
- then it re-fits the chosen points' weights under w ≥ 0, by coordinate descent;
- it stops early when no point would improve the match.

The kernel width defaults to the median distance between source points. Use the same set for
both to summarize a dataset, or a class as the target to find what stands for that class. On
three clusters of 60, 30 and 10 points, three prototypes come one from each cluster, largest
first, with weights 0.6, 0.3 and 0.1.

Test reference: `tests/prototypes_test.cpp`.

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

### Counterfactuals without gradients

```cpp
#include "pulsatrix/counterfactual.hpp"

CounterfactualConstraints c;
c.scale = MedianAbsoluteDeviation(background);
CounterfactualResult cf = GrowingSpheresCounterfactual(predict, input, CounterfactualTarget::ToClass(1), c);
```

**What's happening:** growing spheres (Laugel et al. 2018) draws points uniformly in a ball around
the input, measured in units of each feature's scale. While the ball still contains a point that
reaches the target, it halves the radius; then it draws in ever larger shells until one does. It
keeps the nearest such point and puts features back to their original values, smallest change
first, as long as the point still reaches the target, so the answer is sparse. The result, the
constraints and the view (`RenderCounterfactualSvg`) are the same as for `FindCounterfactual`.

Test reference: `tests/counterfactual_test.cpp`.

### Global sensitivity: Morris and Sobol

```cpp
#include "pulsatrix/global_sensitivity.hpp"
#include "pulsatrix/viz/svg.hpp"

// Vary features 0, 1 and 2 over their ranges; every other feature keeps input's value.
GlobalSensitivityProblem problem{input, {0, 1, 2}, {0.0f, 0.0f, -1.0f}, {1.0f, 10.0f, 1.0f}};

MorrisResult morris = ComputeMorris(predict, problem, /*target_index=*/0);  // 20 trajectories
std::string scatter = RenderMorrisSvg(ToMorrisDocument(morris, {"a", "b", "c"}, "risk"));

SobolOptions opts;
opts.num_samples = 2048;  // the model runs 2048 * (3 + 2) times
SobolResult sobol = ComputeSobol(predict, problem, 0, opts);
std::string bars = RenderSobolSvg(ToSobolDocument(sobol, {"a", "b", "c"}, "risk"));
```

**What's happening:** Morris walks random one-feature-at-a-time paths over a grid of each
range. A feature's elementary effects are the output changes per step. μ* (their mean absolute
value) ranks importance, and σ (their spread) is large when the effect depends on where you are:
nonlinearity or interaction. Sobol splits the output's variance: the first-order index is the
share a feature explains alone, the total-order index includes its interactions, and the gap
between them is the interaction share. Both use a seeded generator that gives the same samples on
every platform. The analyzers reproduce SALib 1.6 (`morris.analyze`, and `sobol.analyze` without
second order) on the same samples; the confidence intervals are bootstraps with a different
generator, so they agree in size, not digit for digit.

Test reference: `tests/global_sensitivity_test.cpp`, checked against SALib 1.6.0
(`tools/generate_global_sensitivity_reference_values.py`) and the Ishigami function's analytic
Sobol indices.

## Recipes

- [KernelSHAP basics](../recipes/interpretability/kernel_shap_basics.md)
- [LIME basics](../recipes/interpretability/lime_basics.md)
