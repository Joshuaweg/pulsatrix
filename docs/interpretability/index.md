# Interpretability

These are post-hoc explanation methods: you apply them to an already-trained model to see
which input features drove a prediction. Use them to debug a model, check that it relies on
sensible features, or show a user why it made a decision.

This section has three parts:

- **[Layer-wise Relevance Propagation (LRP)](lrp.md)**: passes the prediction's score back
  through the network, layer by layer, using a rule chosen for each layer. Every pulsatrix
  layer implements an LRP rule, so any network you build can be explained with it.
- **[Model-Agnostic Explainers](model-agnostic.md)**: `KernelSHAP`, `LIME`, `PDP`, ICE, local
  sensitivity and `Occlusion`. They only call the model, perturbing the input and watching the
  output. Use them for any model,
  including ones not built with pulsatrix.
- **[Gradient-Based Explainers](deep-learning-approaches.md)**: `Saliency`,
  `IntegratedGradients`, `GradCAM` and counterfactuals (`FindCounterfactual`). They read a pulsatrix network's gradients and
  activations, so they are faster and see inside the model.

To explain a pretrained language model's predictions over text, see
**[Language Models](../language-models/index.md)**. It covers AttnLRP on Hugging Face models,
word-level scores and figures.

Full API reference: [Doxygen: Interpretability](../api/group__interpretability.html)

## Which explainer answers which question

Different questions need different explainers. Pulsatrix sorts them by the nine question
categories of Liao, Gruen and Miller's XAI question bank (CHI 2020):

| `XaiQuestion` | The question | Explainers |
|---|---|---|
| `Input` | What kind of data was the system trained on? | `NearestNeighbors` |
| `Output` | What kind of output does the system give? | none yet |
| `Performance` | How accurate/precise/reliable are the predictions? | none yet (KS-3) |
| `HowGlobal` | How does the system make predictions? | PDP, ALE, Morris, Sobol, TCAV, probes, featurizers, representation comparison |
| `Why` | Why/how is this instance given this prediction? | LRP, Integrated Gradients, Saliency, Grad-CAM, KernelSHAP, LIME, Occlusion |
| `WhyNot` | Why is this instance NOT predicted to be Q? | Contrastive LRP, counterfactuals |
| `WhatIf` | What would the system predict if this instance changes to…? | Local sensitivity, ICE, PDP, Occlusion, activation patching, steering |
| `HowToBeThat` | How should this instance change to get a different prediction Q? | Counterfactuals |
| `HowToStillBeThis` | What change is permitted for this instance to still get the same prediction? | none yet (Anchors, XAI-4) |

There are also four reasons to explain (Adadi and Berrada, 2018), as `XaiReason`: to
**justify** a decision, to **control** the model by finding its flaws, to **improve** it, and to
**discover** something new from it.

Every explainer declares the questions it answers, the main one first. The declaration also gives
its scope (one prediction or the whole model), the least model access it needs (black box,
weights, activations, gradients or LRP rules) and the inputs it takes. It also says:

- whether it returns an `Attribution`, so that the metrics below apply;
- whether its attributions are complete, adding up to the output;
- whether it needs the base model of a fine-tune;
- the most input features it takes.

The classes with an `explain()` method return their declaration from `capabilities()`. Every
explainer, functions included, is in `ExplainerCatalog()`:

```cpp
#include "pulsatrix/explainer_capabilities.hpp"

LRP::capabilities().answers(XaiQuestion::WhyNot);           // true: contrastive LRP
ExplainerCapabilitiesOf("GrowingSpheres").access;           // ModelAccess::BlackBox

for (const ExplainerCapabilities* e : ExplainersFor(XaiQuestion::HowToBeThat)) {
    std::cout << e->name << " in " << e->header << "\n";    // the three counterfactual searches
}

std::string json = WriteJson(ExplainerCatalogDocument());   // pulsatrix.explainer_catalog.v1
```

`ExplainersFor()` lists first the explainers whose main question it is. A question with no
explainer yet returns an empty list, not a poor substitute.

## Planning an explanation

`ExplanationPlanner` turns a question and a reason into a plan. A plan lists the explainers that
can answer the question for your model, best first, and the checks their results need before you
rely on them. It only reads the catalog above and runs nothing, so every explainer it names is
still called directly.

```cpp
#include "pulsatrix/explanation_planner.hpp"

ModelTraits traits;
traits.access = ModelAccess::Relevance;   // a pulsatrix network with an LRP rule at every layer
traits.input = InputKind::Image;
traits.can_reinitialize = true;           // for the randomization test and the null model
ExplanationPlan plan = ExplanationPlanner().plan(XaiQuestion::Why, XaiReason::Justify, traits);

plan.explainers.front().capabilities->name;   // "LRP"
plan.checks;      // faithfulness, stability and conservation (required), randomization (recommended)
plan.excluded;    // e.g. RelevanceGraph: "takes tokens input; the model takes image"
plan.complete();  // true: some explainer applies and every required check can run
std::cout << DescribePlan(plan);              // or WriteJson(ToJson(plan)): pulsatrix.explanation_plan.v1
```

`ModelTraits` says what you have:

- `access`: the most the model gives, from a prediction function (`BlackBox`) to an LRP rule at
  every layer (`Relevance`);
- `input`: the kind of input (`Any` if unknown);
- `can_reinitialize`: whether its weights can be redrawn and restored;
- `has_dataset`: whether there are many inputs or just one;
- `has_base_model`: whether the model it was fine-tuned from is also at hand;
- `features`: the number of features per input, if known.

An explainer that needs more than you have is **excluded**, with the reason. When the input kind
or the feature count is unknown, explainers made for one kind, or limited to a few features
(KernelSHAP, 20), stay in the plan with a **condition** instead.

**The ranking** compares, in order:

1. whether the question is the explainer's main one;
2. the reason's preference:
   - **justify:** complete explainers first, whose attributions add up to the output, then those
     that return an `Attribution` the faithfulness checks can test;
   - **control** and **improve:** global explainers first;
   - **discover:** global explainers first, then those that read the model's internals;
3. explainers made for your kind of input;
4. more model access before less.

Ties keep catalog order.

**The checks** follow the reason:

| Reason | Required | Recommended |
|---|---|---|
| Justify | faithfulness (`DeletionCurve`, `InsertionCurve`), stability, conservation for complete explainers | randomization, null model |
| Control | randomization for attributions, the null model for the rest, per-slice error | faithfulness, stability, conservation |
| Improve | aggregating the explanations over the dataset | faithfulness, stability, conservation, randomization, null model |
| Discover | the null model, for everything, and each explainer's own controls | faithfulness, stability, conservation, randomization |

The explainers' own controls are:

- control tasks for probes (`MeasureSelectivity`, `CompareToBaseline`);
- a significance test for persistent homology (`UniversalNullTest` and the rest);
- split halves for PCA;
- loss recovered for featurizers.

A required check that can't run makes the plan incomplete and adds a warning. For example,
Control requires per-slice error, which pulsatrix doesn't have yet (KS-3), so every Control plan
says so. A question with no explainer, such as Output, comes back empty, with a warning, not a
poor substitute.

From the command line, `pulsatrix_plan` prints the same plan. It exits 1 when the plan has a
warning:

```
$ pulsatrix_plan --question why --reason control --access relevance --input tokens --reinitialize --dataset
Question: why. Why/how is this instance given this prediction?
Reason: control. Control the model: find its flaws and failure modes.
Model: an LRP rule at every layer, tokens input, can be re-initialized, a dataset

Explainers, best first:
  1. RelevanceGraph (pulsatrix/relevance_graph.hpp: BuildRelevanceGraph)
     answers why as its main question; made for tokens input; needs an LRP rule at every layer
  2. CircuitTracing (pulsatrix/circuit_tracing.hpp: TraceCircuit, ScoreCircuit)
     answers why as its main question; made for tokens input; needs activations
  3. LRP (pulsatrix/lrp.hpp: LRP::explain)
     answers why as its main question; needs an LRP rule at every layer
  ...

Checks:
  [required] randomization: ModelParameterRandomizationTest (pulsatrix/explanation_metrics.hpp), for LRP, Saliency, ...
  [required] null_model: NullModelBaseline (pulsatrix/null_model_baseline.hpp), for RelevanceGraph, CircuitTracing, ...
  [required, can't run] per_slice_error
     can't run: pulsatrix has no per-slice error report yet (KS-3)
  ...

Warnings:
  control requires the per_slice_error check, which can't run: pulsatrix has no per-slice error report yet (KS-3)
```

The plan is only as precise as the catalog. Explainers declared for any input are offered for
every kind. For a language model, though, LRP, Saliency and Integrated Gradients run through an
`ExplainerContext`, a list of modules, rather than a `CausalLM`. The `CausalLM` explainers are the
token-specific ones at the top of this plan.

## Shared infrastructure

- **`Attribution`** (`include/pulsatrix/attribution.hpp`): the result type every explainer
  returns, LRP included. It holds the attribution `values`, the `method` name and a `metadata`
  map, so downstream code needs only one code path for all explainers.
- **`ExplainerContext`** (`include/pulsatrix/explainer_context.hpp`): wraps your network's
  modules, given as a `std::vector<Module*>` in forward order. It runs traced forward,
  backward and relevance passes and exposes per-node activations and gradients. LRP and
  the gradient-based explainers take it as their first argument.
- **Eval mode.** Call `model.set_training(false)` before explaining a model that has BatchNorm
  or Dropout. No explainer switches modes for you, and in training mode each sample's
  explanation depends on the rest of its batch.
- **Saving and plotting.** `ToAttributionDocument()` turns an `Attribution` into a
  `pulsatrix.attribution.v1` JSON document; `RenderBarChartSvg()` and `RenderWaterfallSvg()`
  draw it as an SVG figure, and the `pulsatrix_svg` tool does both from the command line. See
  [Visualization](../visualization/documents.md).

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

### The null-model baseline

Attribution maps, probes and sparse autoencoders all produce plausible-looking output on a
network with random weights ("The Dead Salmons of AI Interpretability", 2025). So every result
should be compared against the same analysis of a randomly re-initialized model.
`NullModelBaseline` does that in one call, for any analysis. It runs the analysis on the trained
model and on a re-initialized one, then restores the model exactly:

```cpp
#include "pulsatrix/null_model_baseline.hpp"

// Any analysis, any result type: here a probe's accuracy on the model's activations.
auto probe_accuracy = [&] { return TrainProbeAndScore(model, data); };
NullModelComparison<float> acc = NullModelBaseline(model, probe_accuracy, /*seed=*/0);
// acc.trained vs acc.null_model: an accuracy the null model also reaches isn't evidence

// An explanation: also reports how similar the null model's explanation is.
AttributionNullReport r = NullModelBaseline(model, explain, image, /*seed=*/0);
// r.rank_similarity close to 1: the explanation doesn't depend on what the model learned
```

Re-initialization replaces each parameter tensor with Gaussian noise at that tensor's own scale,
and leaves buffers such as BatchNorm's running statistics alone. `ParameterSnapshot` and
`ReinitializeParameters` are available separately.

## Reporting an explanation

`BuildExplanation` (`include/pulsatrix/explanation_report.hpp`) runs an attribution explainer,
then every check the plan asks for, and returns an `Explanation`. The report holds:

- the question and reason, the method and its rank in the plan;
- the attribution, its settings (the explainer's metadata, yours, and every check option) and two
  metrics, sparseness and complexity;
- one result per planned check: passed, failed, measured, or not run with the reason;
- the **caveats**, derived from the rest.

`supports_reason()` is false when any caveat is blocking.

```cpp
#include "pulsatrix/explanation_report.hpp"

ExplanationRequest request;
request.question = XaiQuestion::Why;
request.reason = XaiReason::Justify;
request.method = "KernelSHAP";
request.traits.input = InputKind::Tabular;
request.traits.can_reinitialize = true;
request.baseline = zeros;  // what KernelSHAP measured from, for the conservation check

PredictFn predict = [&](const Tensor& x) { return model.forward(x); };
ExplainFn explain = [&](const Tensor& x) { return KernelSHAP().explain(predict, x, zeros, 0, &backend); };
Explanation e = BuildExplanation(request, x, explain, predict, &model);
std::cout << DescribeExplanation(e);
```

**What each check does:**

| Check | How it runs | Passes when |
|---|---|---|
| faithfulness | Deletion and insertion curves, scored as insertion AUC minus deletion AUC, against 19 random orderings of the features | it beats them: p ≤ 0.05 |
| stability | Two more runs on the same input, and three on the input plus 1% noise (not for token or symbolic input) | every Spearman correlation of attribution magnitudes is ≥ 0.8 |
| conservation | The attributions' sum against LRP's seeded total, or f(input) − f(baseline) | the gap is ≤ 5% |
| randomization | `ModelParameterRandomizationTest` | the fully randomized model's explanation correlates ≤ 0.5 with the original |
| null_model | `NullModelBaseline` | the same threshold |
| aggregate | The mean absolute attribution over `request.dataset`, kept in `Explanation::aggregate` | (measured, no verdict) |

Every threshold is in `ExplanationCheckOptions`, and the report records the values used. A check
pulsatrix can't run yet, such as per-slice error (KS-3), comes back as not run.

**The caveats** follow from the report alone, so `DeriveCaveats()` gives the same caveats for a
stored report:

- **Blocking:**
  - a required check failed or didn't run;
  - the planner excluded the method for this model.
- **Warning:**
  - a recommended check failed;
  - a known failure mode applies to these settings or this input: a constant fill on images,
    a fill value substituted into token ids, or fewer than 50 Integrated Gradients steps.
- **Note:**
  - a recommended check that didn't run;
  - the method's known limitations: Integrated Gradients' and KernelSHAP's baseline, LRP's rule
    choice, LIME's sampling, Saliency's sensitivity, Grad-CAM's resolution, Occlusion's fill;
  - for Justify, that an attribution shows what the model relied on, not whether it should have.

On a 12-feature MLP the code above prints:

```
Explanation: KernelSHAP, for the why question. Why/how is this instance given this prediction?
Reason: justify. Justify a decision to the people it affects or who must sign it off.
Supports the reason: yes
Plan: ranked 1 for this question, reason and model

Largest attributions (kernel_shap, 12 features): [11] 0.116, [8] -0.112, [4] -0.0937, [7] 0.0756, [2] -0.0708
Metrics: sparseness 0.416, complexity 2.17
Settings: target_index=0, check.target=0, check.seed=0, check.imputation=constant:0, ...

Checks:
  [required] faithfulness: passed. Removing the features it ranks highest, and restoring them, moved the target output more than 19 of 19 random orderings did (insertion minus deletion AUC 0.47, p = 0.05).
  [required] stability: passed. Repeated runs were identical; with noise of 1% of the input's spread, the ranking correlated 1 with the original on average.
  [required] conservation: passed. The attributions add up to -0.0324 against -0.0324, f(input) - f(baseline): a gap of 0%.
  [recommended] randomization: failed. With the 2 layers randomized from the output down, the explanation's ranking correlates 0.51 with the original (0.559 after the top layer alone).

Caveats:
  [warning] Randomization check failed. With the 2 layers randomized from the output down, ...
  [note] An absent feature takes the baseline's value, so the Shapley values compare with the baseline and treat features as independent; correlated features split credit unpredictably (Kumar et al. 2020).
  [note] An attribution shows what the model relied on for this prediction, not whether it should have: passing the checks means the ranking matches the model, not the world.
```

The randomization warning is a real result, not a fault of the example. The input's values range
from 0.05 to 2.2, so any model's attributions grow with |x|, and a random network's ranking
still correlates 0.51 with the trained one's. Adebayo et al. found the same of gradient × input.
Justify only recommends the test, so the explanation still supports its reason. Asked to control
the model, the same explanation would be blocked.

`ToJson(e)` writes a `pulsatrix.explanation.v1` document. For explainers that don't return an
attribution, such as a probe or persistent homology, run their checks yourself and pass the
results to `ReportExplanation`. It matches them to the planned checks by name, marks the missing
ones as not run, and derives the same caveats.

## Recipes

New here? Start with the
[Saliency and Integrated Gradients recipe](../recipes/interpretability/saliency_and_integrated_gradients.md).
See the full recipe lists on [LRP](lrp.md#see-also),
[Model-Agnostic Explainers](model-agnostic.md#recipes) and
[Gradient-Based Explainers](deep-learning-approaches.md#recipes).
