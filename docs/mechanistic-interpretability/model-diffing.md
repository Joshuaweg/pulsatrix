# Model Diffing with Crosscoders

**What did fine-tuning change?** Comparing two models' weights directly says little, because
the change is spread over every parameter. **Model diffing** compares what the two models
*represent*. Train one dictionary of features on both models' activations at once, then ask
which features belong to only one of them.

The tool for that is a **crosscoder** (Lindsey et al.): one sparse dictionary read from and
written to several **sources** at once, such as the same layer in a base model and its
fine-tune, or several layers of one model. Each feature has a single code, and its own decoder
direction in each source. The decoders' norms say where a feature writes:
- about equal norms: a feature both models share;
- a near-zero norm in one model: a feature only the other model has.

## Quick start: diff a base model and its chat fine-tune

`pulsatrix_diff_lm` does the whole thing:
1. reads the same text through both models;
2. trains a crosscoder on their residual streams at one layer;
3. reports which features are specific to one model, and what they fire on.

```bash
hf download HuggingFaceTB/SmolLM2-135M --include "*.json" --include "*.safetensors" --local-dir models/SmolLM2-135M
hf download HuggingFaceTB/SmolLM2-135M-Instruct --include "*.json" --include "*.safetensors" --local-dir models/SmolLM2-135M-Instruct
python3 tools/explain/fetch_chat_sample.py data/chat      # 2,379 chat conversations in ChatML

pulsatrix_diff_lm models/SmolLM2-135M models/SmolLM2-135M-Instruct \
    --text data/chat/conversations.jsonl --out diff/ --layer 15 \
    --crosscoder btk --epochs 12 --lr 1e-3 --device hip
```

| Option | Default | Meaning |
|---|---|---|
| `--layer L` | the middle | Residual-stream position: 0 the embeddings, i the output of block i |
| `--crosscoder` | `btk` | `btk` (BatchTopK), `l1`, or `delta` (Delta-Crosscoder) |
| `--features N` | 8 × hidden | Dictionary size |
| `--k K` | 32 | Active features per token (BatchTopK) |
| `--l1 λ` | 1e-3 | Sparsity penalty (L1) |
| `--epochs`, `--batch`, `--lr` | 4, 1024, 3e-4 | Training |
| `--max-length N` | 256 | Tokens kept per text |
| `--lean N` | 50 | How many of the features leaning furthest toward each model get Latent Scaling |

**It prints**, on the held-out tenth of the texts:
- each model's explained variance;
- how many features fall in each class: base-only, fine-tune-only, shared;
- how many pass Latent Scaling's test for being specific to one model;
- how each group's activations split over chat-template tokens and the system, user and
  assistant turns;
- the top tokens of the most fine-tune-specific features.

`diff/latents.csv` has every feature's decoder norms, Δnorm, cosine and, where measured, ν
(defined below). A run of 12 epochs over 504K tokens took about 16 minutes on a Radeon 8060S.

## In C++

```cpp
#include "pulsatrix/crosscoder.hpp"

CrosscoderOptions o;                       // BatchTopK by default
o.k = 32;
Crosscoder c(/*sources=*/2, /*dim=*/576, /*features=*/4608, &backend, o);
// x is (N, 2 × 576): the base model's activations (source 0, "a"), then the fine-tune's
// (source 1, "b"), for the same tokens. Scale each model's activations so their mean squared
// norm is the dimension. sample: a few thousand such rows.
c.initialize_bias(sample);
for (const Tensor& batch : batches) {
    (void)TrainFeaturizer(c, batch, opt, /*unit_norm_decoder=*/false);   // decoder norms carry the result
}

std::vector<CrosscoderLatentStats> stats = CrosscoderLatents(c);   // per feature: norms, Δnorm, cosine
LatentClass cls = ClassifyLatent(stats[i]);                        // AOnly, BOnly, Shared, Other
// candidates: the feature indices to test, for example those with Δnorm above 0.9.
std::vector<LatentScaling> ls = MeasureLatentScaling(c, held_out, candidates);   // ν per feature
std::vector<double> ev = ExplainedVarianceBySource(c, held_out);
```

With `sources` > 2 and one model's layers side by side, the same class finds features that span
several layers.

## Reading the results

The API calls features **latents**, as the papers do. Source **a** is source 0 (the base model)
and **b** is source 1 (the fine-tune); swap the source arguments to ask the other way round.

**Decoder norms.**
- **Δnorm** (Minder et al.) is `½ (1 + (‖d_b‖ − ‖d_a‖) / max(‖d_a‖, ‖d_b‖))`. It is 0 for a
  feature only in model a, 1 for one only in model b, and ½ for an even split.
  `relative_norm`, `‖d_b‖ / (‖d_a‖ + ‖d_b‖)`, agrees at those three points.
- `ClassifyLatent` uses Minder et al.'s bins: below 0.1 **a-only**, above 0.9 **b-only**, 0.4 to
  0.6 **shared**, and **other** in between (0.1 to 0.4, 0.6 to 0.9).
- **The cosine** between a shared feature's two decoders says whether both models write it in the
  same direction.

**Latent Scaling.** Decoder norms can mislead: a feature can look one-model when it isn't, and
the other way round. Latent Scaling (Minder et al.) checks each candidate against the data. For
each model it fits a scale β: how much of the feature's b-side direction, times its code, best
explains that model's activations. **ν = β_a / β_b** is then how much model a uses the feature,
relative to model b. Near 0 means model a doesn't need it; near 1 means both do.

It is fitted against two targets:
- **ν_error**: against what the *other* features leave unexplained in each model. A high value
  means **complete shrinkage**: model a needs the feature too, but its a-side decoder was
  shrunk to almost nothing.
- **ν_reconstruction**: against the crosscoder's reconstruction. A high value means **latent
  decoupling**: other features write the same thing into model a.

A feature is **specific to model b** when ν_reconstruction < 0.5 and ν_error < 0.2.

**Run Latent Scaling, not only the bins.** In both experiments below, the bins missed what
Latent Scaling found.

## Variants

| Variant | Option | How it differs |
|---|---|---|
| L1 (Lindsey et al.) | `o.sparsity = CrosscoderSparsity::L1` | Penalty `λ Σ_i f_i Σ_s ‖d_i^s‖`: a sum of decoder norms, so a one-model feature stays cheap |
| BatchTopK (Minder et al.) | the default | The batch's `N·k` largest `f_i Σ_s ‖d_i^s‖` are kept; codes aren't shrunk; a threshold at inference |
| Delta-Crosscoder (Kassem et al.) | `o.delta = true` | Separate shared and "delta" feature sets, each with its own top-k; a loss that makes the delta features alone predict `x_fine-tune − x_base`. Meant for narrow fine-tunes |

## What we found

**Planted features** (`tests/crosscoder_test.cpp`). Two synthetic "models" share 16 features,
and each has 4 of its own.
- **When every feature is common** (8% of inputs), BatchTopK recovers all 24 and puts each in the
  right class.
- **In a narrow fine-tune**, the fine-tune adds 4 rare features (0.5% of inputs each).
  - The crosscoder learns them, but the bins call none of them fine-tune-only (mean Δnorm 0.76).
    Their base-model decoders keep about half their norm, even after 10 times more training.
  - **Latent Scaling calls all 4 fine-tune-specific**, and all 16 shared ones needed by both
    models.
  - The Delta-Crosscoder doesn't fix the norms. Its top-k must keep `N·k` delta activations per
    batch even when the fine-tune's features are rarer than that.

**SmolLM2-135M against SmolLM2-135M-Instruct**, with the command above (`--crosscoder btk`,
`l1 --l1 5e-4` or `delta`; 12 epochs at a learning rate of 1e-3 each): 504K tokens, layer 15 of
30, 4,608 features, scored on 50K held-out tokens. The Delta-Crosscoder used its defaults, 32
active delta features and 64 shared.

| | BatchTopK (k = 32) | L1 (λ = 5e-4) | Delta-Crosscoder |
|---|---|---|---|
| Explained variance (both models) | 0.93 | 0.93 | 0.96 |
| L0 | 32 | 46 | 96 |
| One-model features by the bins | 0 | 11 | 0 |
| Instruct-specific by Latent Scaling, of the 50 leaning furthest | 8 | 1 | 12 |
| Share of those features' activation on chat-template tokens (which are 22% of all tokens) | 86% | 99% | 84% |

- **What chat tuning added is about the chat template.** The Instruct-specific features fire on
  `<|im_start|>`, the role name, and the newline that closes a turn's header. Minder et al. found
  the same in Gemma 2.
- **The bins alone find almost nothing** for two models this close. Latent Scaling is what
  separates the specific features.
- **The Delta-Crosscoder finds the most,** but at three times the L0, which also explains its
  higher explained variance.

These are small runs: Minder et al. train on 100M tokens.

## How it's checked

All three variants match a PyTorch rendering (`tools/golden/make_crosscoder_golden.py`) through
the loss, every gradient and two Adam steps.

**Conventions:**
- The loss is the mean squared error over every source plus the penalty, on the same scale as
  the other featurizers; Lindsey et al. sum it.
- Every feature starts with the same decoder in every source, at norm 0.05 for L1 and 1 for
  BatchTopK (Minder et al.'s values).
- The Delta-Crosscoder paper has no public code, so it follows the paper's equations. Its
  separate sparsity term is left out, since the top-k already fixes the sparsity.

## References

- Lindsey et al., "Sparse Crosscoders for Cross-Layer Features and Model Diffing",
  Transformer Circuits, 2024.
- Minder, Dumas et al., "Overcoming Sparsity Artifacts in Crosscoders to Interpret
  Chat-Tuning", arXiv 2504.02922.
- Kassem et al., "Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning
  Regimes", arXiv 2603.04426.
