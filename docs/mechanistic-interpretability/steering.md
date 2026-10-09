# Steering

**Steering** adds a direction to a model's residual stream during the forward pass, to push its
behavior one way: more positive sentiment, a different language, a hydrophobic residue. It is
the causal counterpart of a probe. A probe asks whether a concept can be *read* from a layer;
steering asks whether writing a direction *changes* what the model does.

It is also easy to fool yourself with. An average effect at one large coefficient can look like
success when any perturbation of that size would do the same, and many individual inputs may
move the wrong way. `MeasureSteering` reports what you need to tell the difference.

## Quick start

```cpp
#include "pulsatrix/steering.hpp"

// 1. A direction: the difference of means between activations with and without the concept.
//    with_concept, without_concept: row-major (N, d) activations from the same layer.
std::vector<float> v = DifferenceOfMeans(with_concept, without_concept, /*d=*/hidden);

// 2. Steer: add coefficient × v to the residual stream after block 4.
model->set_hidden_state_hook(SteeringHook(/*position=*/4, v, /*coefficient=*/1.0f));
// ... generate or score ...
model->set_hidden_state_hook({});

// 3. Measure how reliably it works.
SteeringReport r = MeasureSteering(
    [&](int64_t i, const HiddenStateHook& hook) {
        model->set_hidden_state_hook(hook);
        const double b = Behavior(*model, inputs[i]);   // a number steering should raise
        model->set_hidden_state_hook({});
        return b;
    },
    static_cast<int64_t>(inputs.size()), /*position=*/4, v);
```

- **The direction.** The difference of means is the default, and usually the better choice. A
  featurizer's decoder direction is the alternative (`FeaturizerDirection(sae, feature)`); on
  AxBench (Wu et al.) it usually steers worse.
- **The behavior** is any number steering should raise: a logit difference, a log-probability,
  a probe's score. `MeasureSteering` calls it once per input and coefficient.
- **Options** (`SteeringOptions`):
  - the coefficients to try. The default is −2, −1, 0, 1, 2, with 0 as the unsteered baseline.
    **Check that these are small for your model:** in the example below, ±2 was large enough to
    disrupt ESM-2 and fake a success. Choose a range well below the residual stream's norm.
  - how many random directions to use as the control (default 3);
  - a seed.
- `SteeringHook` takes an optional row mask, to steer only some positions.

## Reading the report

Following Tan et al., the report doesn't stop at the mean effect:

| Field | Meaning |
|---|---|
| `steerability` | Each input's least-squares slope of behavior against the coefficient |
| `mean_steerability`, `steerability_sd` | Their mean and spread over inputs |
| `anti_steerable_fraction` | The share of inputs with a negative slope: steering moves them the wrong way |
| `random_mean_steerability`, `random_max_abs_steerability` | The same for random directions of the same norm: their mean, and the largest magnitude |
| `over_random` | Mean steerability divided by `random_max_abs_steerability`. Below about 1, the direction does no better than noise |

A checklist before calling a steering vector successful:
1. `over_random` well above 1.
2. A small anti-steerable share.
3. Behavior that rises roughly linearly with the coefficient, in both directions. If it rises
   at both −1 and +1, the perturbation is breaking the model, not steering it.
4. Small coefficients relative to the residual stream's norm, so you stay where the model is
   linear.

## A worked example: a steering vector that fails its checks

`pulsatrix_steer_esm` steers ESM-2 8M's layer 4 toward transmembrane residues:

```bash
python3 tools/plm/fetch_swissprot_annotations.py data/swissprot
pulsatrix_steer_esm models/esm2_t6_8M_UR50D --annotations data/swissprot/annotated.jsonl --out steer/ \
    --layer 4 --max-coefficient 2
```

It writes the per-coefficient means (`steering.csv`) and each protein's slope
(`steerability.csv`). `--max-coefficient` sets the range, from −C to C; it defaults to 0.5.
The ESM-2 download is in [Finding features](featurizers.md#on-a-protein-model-pulsatrix_probe_esm).

The setup:
- **Direction:** the difference of means between transmembrane residues and the rest, over 800
  proteins. Its norm is 9.1, against a mean residual norm of 29. The alternative is the TopK SAE
  feature that best matches transmembrane residues (F1 0.64 in this run's SAE), scaled to the
  same norm.
- **Behavior:** the log-probability of a hydrophobic residue minus a polar one, at masked
  positions. It should rise.
- **Inputs:** 100 held-out soluble proteins.

| Coefficients | Direction | Mean steerability | Anti-steerable | `over_random` |
|---|---|---|---|---|
| ±2 | Difference of means | +0.087 | 18% | 0.29 |
| ±2 | SAE feature | +0.023 | 32% | 0.08 |
| ±0.5 | Difference of means | **−0.139** | **66%** | −1.41 |
| ±0.5 | SAE feature | +0.002 | 46% | 0.02 |
| ±0.25 | Difference of means | **−0.208** | **78%** | −1.96 |

- **At ±2 the average looks like success, and it isn't steering.** The added vector has norm 18,
  and random directions of the same norm move the behavior *more*. The behavior also rises at
  both −1 and +1, which no linear direction explains.
- **At small coefficients the difference of means steers the wrong way.** It makes soluble
  proteins' masked positions *less* hydrophobic, for about three proteins in four.
- **The SAE feature doesn't steer at all** at any scale.

A mean effect at ±2 alone would have reported a success. The per-input slopes, the
anti-steerable share and the random control show there is none. **A probe that reads a concept
doesn't give a direction that writes it.** For this reason the tool defaults to ±0.5.

## References

- Tan et al., "Analyzing the Generalization and Reliability of Steering Vectors", NeurIPS 2024.
- Wu et al., "AxBench", arXiv 2501.17148.
