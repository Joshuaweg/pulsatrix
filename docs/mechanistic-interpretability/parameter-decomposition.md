# Parameter Decomposition

Featurizers explain a network's **activations**. Parameter decomposition explains its
**weights**. The idea: a weight matrix does several separate jobs at once. Split it into pieces
so that every input needs only a few pieces to get the same output. Each piece is then one part
of the network's mechanism, and you can ask which inputs use which pieces.

pulsatrix implements two methods from Goodfire's line of work:
- **SPD**, stochastic parameter decomposition (Bushnaq, Braun and Sharkey);
- **VPD**, adversarial parameter decomposition (Bushnaq et al., 2026), its successor. Goodfire
  names it "VPD" for ad**V**ersarial **P**arameter **D**ecomposition. It isn't a variational
  method.

## The idea in three terms

- **Subcomponents.** Each weight matrix W is written as a sum of C rank-one pieces:
  `W ≈ Σ_c V[:, c] U[c, :]`, or `W ≈ V U`. Training asks the pieces to add back up to W, a
  loss called **faithfulness**.
- **Causal importance.** A small network gives each subcomponent a score `g` in [0, 1] for each
  input: how much that input needs it.
- **Stochastic masks.** Training scales each subcomponent by a random mask `m = g + (1 − g) r`,
  with r uniform in [0, 1]. An important subcomponent (g near 1) is kept whole. An unimportant
  one is scaled randomly, and the output must not change. So `g` can only stay low for
  subcomponents an input really doesn't use.

A penalty on `g` (**importance minimality**) pushes each input to use as few subcomponents as
possible. After training, a subcomponent is a mechanism, and `g` says which inputs use it.

## Quick start

Wrap each `LinearModule` you want to decompose in a `ComponentLinear`, and build the model from
the wrappers. This works for models assembled from layers you hold yourself, in containers like
`SequentialModule` and `ResidualModule` that keep pointers to them.

```cpp
#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/parameter_decomposition.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"

// The trained target: w1 (5 → 2), w2 (2 → 5), then a ReLU.
ComponentLinear c1(w1, /*components=*/20, &backend), c2(w2, 20, &backend);
SequentialModule model({&c1, &c2, &relu});

DecompositionOptions o;                        // SPD by default
o.importance_coefficient = 3e-3f;
o.p = 1.0f;
ParameterDecomposition d(model, {&c1, &c2}, &backend, o);
AdamOptimizer opt(1e-3f, &backend);
for (int step = 0; step < steps; ++step) {
    const DecompositionLoss loss = TrainDecomposition(d, NextBatch(), opt);
}

std::vector<std::vector<float>> ci = d.causal_importances(x);   // per layer, (N, C)
std::vector<ImportanceStats> stats = MeasureImportance(d, x);   // alive subcomponents, L0
ComponentAlignment a = AlignComponentsToRows(c1);               // MMCS and ML2R, for toy models
```

- **The target model is the model itself** with every `ComponentLinear` in target mode, so you
  don't pass it separately.
- **The output loss** compares the masked model's output with the target's: mean squared error,
  or `OutputDivergence::KlOnLogits` for logits.
- **Training takes long.** The reference toy runs use 40,000 steps of 4,096 inputs, with a
  learning rate decaying to zero (`set_learning_rate` on the optimizer).
- **No autograd needed.** `ComponentLinear::backward()` returns the masks' gradients, so the
  decomposition runs on the model's own backward pass.

## SPD and VPD

| | SPD | VPD |
|---|---|---|
| Faithfulness | `Σ_l ‖W − V U‖²` over the number of weights | The same, as the loss on the left-over weight Δ = W − V U |
| Reconstruction | Every layer masked; then one layer masked at a time | For each input, a random number k of the layers use their subcomponents and the rest their original weights. The left-over weight Δ gets its own random mask |
| Adversarial masks | — | Masks from persistent sources that ascend the loss (PGD), so the decomposition must hold for the worst masks, not only random ones |
| Importance minimality | `Σ g^p` | Adds a term that rewards subcomponents used on few inputs. The paper anneals p from 2 to 0.4 during training (`set_p`) |

Choose with `o.method = DecompositionMethod::SPD` or `VPD`.

**Reading the results:**
- **Alive subcomponents** (`ImportanceStats::alive`): those important (g > 0.1) on some input.
  Ideally there is one per real mechanism, and the rest die.
- **L0 of causal importance:** how many subcomponents an input uses on average.
- **MMCS and ML2R** (toy models). For each input feature, the best cosine between a subcomponent
  and the target weight's row, and that subcomponent's norm relative to the row. 1.0 and 1.0
  mean the feature has a subcomponent that carries its whole row. An ML2R slightly above 1, as
  in the results below, means the subcomponent overshoots a little and others cancel the excess.

## What we found: a toy model of superposition

`pulsatrix_spd_toy` trains the paper's first test case, then decomposes it: 5 sparse features
squeezed through 2 dimensions, `x̂ = ReLU(x W Wᵀ + b)`, with W and Wᵀ decomposed into 20
subcomponents each.

```bash
pulsatrix_spd_toy --method spd --steps 40000 --batch 4096
```

| | SPD | SPD paper | VPD (p annealed from 2 to 1) |
|---|---|---|---|
| MMCS / ML2R | 1.000 / 1.017 | 1.000 / 0.993 | 0.993 / 0.672 |
| Alive subcomponents per layer (of 20) | 5, 5 | 5 | 4, 5 |
| Features with exactly one important subcomponent per layer | 5 of 5 | | 5 of 5 |

- **SPD reproduces the paper.** Each feature gets one subcomponent per layer, carrying its whole
  weight row.
- **VPD, run with SPD's toy settings, merges two of the five features.** The VPD report has no
  toy results to tune against.
- **Faster settings for experiments.** A learning rate of 1e-2 and an importance coefficient of
  1e-2 reach one subcomponent per feature in 3,000 steps of 1,024.

`PULSATRIX_SPD_DUMP=1` makes the tool print every subcomponent's part of each feature's weight
row, with its causal importance. That makes problems visible. In development, the first toy
target had dropped one of its five features, so that feature's weight row was almost zero.
Measures relative to that row were then meaningless.

## Limitations

- **Language models can't be decomposed.** `ComponentLinear` must be swapped in for a
  `LinearModule`, and `CausalLM`'s blocks own theirs.
- **VPD's causal-importance network for language models** isn't here. VPD there uses one
  transformer over every layer's activations; pulsatrix uses SPD's small per-subcomponent
  networks for both methods.

## How it's checked

A PyTorch rendering (`tools/golden/make_spd_golden.py`) matches both methods through the loss,
every gradient and an Adam step:
- SPD's four losses;
- VPD's routing, Δ masks, adversarial masks, frequency term and straight-through clamp.

`ComponentLinear` matches finite differences. The implementations follow Goodfire's reference
code, github.com/goodfire-ai/param-decomp: tag `v1` for SPD, and `nano_param_decomp/run.py` for
VPD.

## References

- Bushnaq, Braun and Sharkey, "Stochastic Parameter Decomposition", arXiv 2506.20790.
- Bushnaq et al., "Interpreting Language Model Parameters", Goodfire, 2026
  (goodfire.ai/research/interpreting-lm-parameters).
- Braun et al., "Interpretability in Parameter Space" (APD), arXiv 2501.14926.
- Elhage et al., "Toy Models of Superposition", Transformer Circuits, 2022.
