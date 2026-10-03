# Layer-wise Relevance Propagation (LRP)

LRP explains one prediction by starting at the output and passing a "relevance" score
backward, layer by layer, until every input feature has a share. Unlike gradient-based
explainers, each layer uses its own propagation rule, chosen to fit what that layer computes.

In pulsatrix, LRP is built into the layer contract: every `Module` must implement
`propagate_relevance()`. The `LRP` class runs those rules over a whole network, so any network
built from pulsatrix layers can be explained without extra setup.

## Quick example

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp.hpp"
#include "pulsatrix/relu_module.hpp"

using namespace pulsatrix;

CPUBackend backend;
LinearModule fc1(2, 4, &backend);
ReluModule relu(&backend);
LinearModule fc2(4, 2, &backend);
// ... train or load weights ...

ExplainerContext ctx({&fc1, &relu, &fc2});  // the layers, in forward order
Tensor input(Shape({1, 2}), &backend, {1.0f, 0.5f});

// Which input features pushed class 1 up? Epsilon rule on every layer:
Attribution eps = LRP().explain(ctx, input, /*target_index=*/1, &backend);

// Or a Zennit-style composite that picks a rule per layer type:
Attribution plus = LRP::epsilon_plus().explain(ctx, input, 1, &backend);

// eps.values has the input's shape: one relevance score per input feature.
```

The network output must be rank 2, `(batch, num_classes)`.

From Python:

```python
import pulsatrix_py as px

ctx = px.ExplainerContext([fc1, relu, fc2])
x = px.Tensor.from_values([1, 2], [1.0, 0.5])
attr = px.LRP().explain(ctx, x, [1])                 # epsilon everywhere
attr = px.LRP.epsilon_plus().explain(ctx, x, [1])    # composite
attr = px.LRP(rule=px.LRPRule.Gamma, gamma=0.25).explain(ctx, x, [1])
```

## Rules

Pick a rule with `LRPRuleConfig` and pass it to `LRP(config)` to use it on every layer.

| `LRPRule` | What it does | Parameters |
|---|---|---|
| `Epsilon` (default) | Divides each contribution by the layer output plus a small stabilizer. | `epsilon` (default `1e-6`) |
| `Gamma` | Boosts positive contributions before dividing, which reduces noise. | `gamma` (default `0.25`) |
| `AlphaBeta` | Weights positive and negative contributions separately. `alpha - beta` must be 1. `alpha = 1, beta = 0` is ZPlus. | `alpha`, `beta` |
| `ZBox` | For the input layer when inputs are bounded, e.g. pixels in `[0, 1]`. | `low`, `high` |

The semantics follow Zennit 1.0.0. `Epsilon` works on every layer. `Gamma`, `AlphaBeta`
and `ZBox` are defined only for the affine layers, `LinearModule` and `Conv2DModule`.
Pass-through layers (`ReluModule`, `FlattenModule`, `DropoutModule`, `MaxPool2DModule`) accept
any rule and pass relevance through unchanged.

If you ask a layer for a rule it doesn't implement, `explain()` throws
`std::invalid_argument` before propagating anything. It never silently falls back to epsilon.

```cpp
LRPRuleConfig config;
config.rule = LRPRule::AlphaBeta;
config.alpha = 2.0f;
config.beta = 1.0f;
Attribution a = LRP(config).explain(ctx, input, 1, &backend);
```

### Bias handling

By default the epsilon rule leaves the bias out of the denominator. Under that default, linear
and convolution layers pass on all the relevance they receive. Zennit and LXT include the bias.
Set `epsilon_bias_in_denominator = true` to match them; the bias then absorbs part of the
relevance.

## Composites: a different rule per layer

A composite chooses a rule for each layer. Three of Zennit's presets are built in:

| Preset | `LinearModule` | `Conv2DModule` |
|---|---|---|
| `LRP::epsilon_plus()` | Epsilon | ZPlus |
| `LRP::epsilon_alpha2_beta1()` | Epsilon | AlphaBeta(2, 1) |
| `LRP::epsilon_gamma_box(low, high)` | Epsilon | ZBox on the first Conv2D, Gamma on the rest |

Every other layer gets the epsilon rule. The presets set `epsilon_bias_in_denominator = true`
so they match Zennit. As in Zennit, `epsilon_gamma_box` never applies ZBox to a `LinearModule`,
so on a network without convolutions it is epsilon everywhere.

To write your own, pass any function `(size_t layer_index, const Module&) -> LRPRuleConfig`:

```cpp
LRP custom([](size_t i, const Module&) {
    LRPRuleConfig c;
    if (i == 0) { c.rule = LRPRule::ZBox; c.low = 0.0f; c.high = 1.0f; }
    return c;
}, "zbox_then_epsilon");
```

## Targets, contrasts and seeds

`explain(ctx, input, target_index, backend)` explains one class. For more control, pass an
`LRPTarget`:

```cpp
// Why class 3 rather than class 8?
LRPTarget target{{3}, {8}};
Attribution a = LRP().explain(ctx, input, target, &backend);
```

- **`targets`**: one class index for the whole batch, or one per row.
- **`contrasts`** (optional): the explanation becomes `explain(target) - explain(contrast)`,
  which answers "why this class rather than that one?"
- **`seed`**: `LRPSeed::OutputValue` (default) starts from the logit itself, so the
  relevance sums to the output. `LRPSeed::OneHot` starts from 1, which is the convention
  Zennit and LXT use. Use `OneHot` when comparing against them.

The returned `Attribution` has `method == "lrp"`, the relevance in `values` (same shape as
the input), and `metadata` with the rule(s) used, the seed, and `relevance_out_sum` /
`relevance_in_sum`. Compare those two sums to see how much relevance was conserved.

## Which layers have which rule

Every layer implements `propagate_relevance()`. The rules beyond plain epsilon come from
the literature:

| Layer | Rule |
|---|---|
| `LinearModule`, `Conv2DModule` | Epsilon, Gamma, AlphaBeta, ZBox |
| `RNNModule`, `LSTMModule`, `GRUModule` | Arras et al. 2019 (relevance follows the signal path; gates pass none) |
| `SoftmaxModule`, `MultiHeadAttentionModule`, `TransformerBlock` | AttnLRP |
| `MambaModule` | MambaLRP |
| `RetNetModule` | AttnLRP-style rule over the retention scores |
| `RWKVModule` | MambaLRP-style rule adapted to the WKV quotient |

Rules that conserve relevance by construction have conservation tests. The AttnLRP rules for
softmax and attention don't conserve exactly; their tests measure and report the gap instead.

The VAE, GAN and diffusion building blocks (`Reparameterize`, `KLDivergenceLoss`,
`BCEWithLogitsLoss`, `NoiseSchedule`, `SinusoidalTimestepEmbedding`) are losses and sampling
steps, not `Module`s, so they have no LRP rule.

## Validation against Zennit and LXT

`tests/lrp_reference_test.cpp` checks `LRP::explain()` (one-hot seed) against values computed
by the reference libraries. The values were generated offline by
`tools/generate_lrp_reference_values.py` in the environment pinned by
`tools/lrp_reference.Dockerfile` (torch 2.14.1+cpu, zennit 1.0.0, lxt 2.1).

**Zennit**, using its `Gradient` attributor on an MLP and a two-conv CNN with nonzero biases and
mixed-sign inputs:

- uniform Epsilon (ε = 1e-6 and 0.25), ZPlus, AlphaBeta(2, 1) and Gamma(0.25)
- the `EpsilonPlus`, `EpsilonAlpha2Beta1` and `EpsilonGammaBox` presets

**LXT**, using its explicit AttnLRP rules on `MultiHeadAttentionModule` and `TransformerBlock`
(RoPE and QK-Norm off), each followed by Flatten → Linear.

The tolerance is `1e-4 · max(1, |ref|)` (float32 vs float32); the largest observed error is
under 1e-5. Matching needs `epsilon_bias_in_denominator = true` (see [Bias handling](#bias-handling)).

Known differences and gaps:

- LXT stabilizes with `z + ε` for every sign of `z`; pulsatrix and Zennit use `z + ε·sign(z)`.
  They differ only where `|z|` is on the order of ε.
- Not covered by the reference test: RoPE, QK-Norm, LayerNorm, the recurrent and state-space
  rules, and LXT's HF-patching (`lxt.efficient`) path.

## See also

- [Recipe: LRP on a trained MNIST classifier](../recipes/interpretability/mnist_lrp.md)
- [Recipe: tracing relevance through a Datalog derivation](../recipes/neuro-symbolic/datalog_lrp_bridge.md)
- [Customization](../customization/index.md#adding-a-new-layer): writing `propagate_relevance()` for your own layer
- [API reference: Layer-wise Relevance Propagation](../api/group__interpretability__lrp.html)
