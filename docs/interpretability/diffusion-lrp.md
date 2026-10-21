# LRP for Diffusion Models (Design)

**Status: design only.** No code implements this page yet. The work is tracked as the
[DIFF epic](../roadmap/index.md#diff-diffusion-models-and-their-explanations), and the evidence
is in the [research notes](../roadmap/research-notes.md#diff-diffusion-models).

A diffusion model doesn't produce its output in one forward pass. It starts from noise (or from a
fully masked sequence) and runs the same network many times, each step refining the last. Every
explanation method published up to 2026-10 explains **one** of those steps: DAAM sums attention
maps, DifFRACT builds attribution graphs inside one Flux step, and DLIG runs Integrated Gradients
at one partially unmasked state. None follows relevance from the final output back through the
whole sampler. No LRP for diffusion has been published, and the AttnLRP author confirmed in
2025-08 that he hadn't tried it (LXT issue #36).

This page sets out three rules that do follow it:

- **D1** for continuous samplers (DDPM, DDIM)
- **D2** for masked diffusion language models (MDLM, LLaDA-style, DPLM)
- **D3** for the adaLN conditioning that diffusion transformers use

There are no reference numbers to match. Each rule therefore comes with the result that would
show it's wrong, and with a set of checks that don't need a reference.

## The common idea: seed once, unroll the sampler

The earlier objection was that diffusion "has no single conserved relevance seed per timestep".
That's true for one step taken alone. The way around it is to seed relevance **once**, on the
final sample, and treat the whole sampler as one deep network. A denoising step reads the
previous state and adds a correction, so it is a residual block. The rules already used for
residual connections, attention and recurrences then apply step by step.

What doesn't fit the network picture is collected into named **sinks** instead of being
dropped: injected noise, the timestep embedding, mask tokens, biases and the epsilon
stabilizer. The sinks and the input relevance together add up to the seed. The ledger of sinks
is part of every explanation: it shows how much of the output the explanation could trace and
where the rest went.

## D1: trajectory LRP for continuous samplers

A DDPM step computes

```
x_{t-1} = c1_t · x_t  −  c2_t · ε_θ(x_t, t, c)  +  σ_t · z_t
          (skip)         (network)                 (injected noise)
```

DDIM is the same step with `σ_t = 0`. The relevance arriving at `x_{t-1}` is split across the
three terms. The skip share goes straight to `x_t`. The network share runs through `ε_θ` with its
ordinary layer rules and lands on `x_t`, on the conditioning `c` and on the timestep embedding
(a constant, so a sink). The noise share is the **injected-noise relevance** of step `t`: how much
of the explained output that step's randomness decided. It is exactly zero under DDIM.

### The split: absolute-value ratio, not epsilon

In early steps the skip and network terms are large and nearly cancel, and early steps are where
class and layout get decided. When terms cancel, the epsilon rule's shares `R · h_k / Σh` grow
without bound. The one published signed split of this kind (Arras et al. 2019, LSTMs) needed
ε = 0.2 to be stable and then scored near random. Rewriting the step in terms of the predicted
clean sample `x̂_0` doesn't help: the cancellation just moves inside `x̂_0`.

D1 therefore uses the absolute-value ratio split from LRP for ResNets (Otsuki et al., ECCV 2024):

```
R_k = R · |h_k| / (Σ_j |h_j| + ε)        h_1 = c1·x_t,  h_2 = −c2·ε_θ,  h_3 = σ·z
```

It conserves relevance exactly, and every share is bounded however much the terms cancel. The
small ε only handles 0/0. It loses sign, so the signed fractions `h_k / Σh` are logged next to
it. The signed epsilon rule, with ε scaled to `Σ|h_k|`, stays available as an option, and it
reports the relevance it absorbs at each step.

The weakness is that the split follows magnitude, not contribution. Two large cancelling terms
each get about half, so the network's share in an early step says how big the terms were, not
what the network decided. The checks below are there to catch this.

### The seed: a scalar of x_0

Seed a scalar computed from the final sample, not the raw noise prediction. The one reported
attempt at gradient × input on DDPM (LXT issue #36) seeded the noise prediction. Its relevance
sign followed the sign of the predicted noise, and its maps changed from run to run. Three
seeds are offered:

- a region of `x_0`
- a reconstruction error `‖x_0 − x_ref‖²` rooted at a reference, as in LRP for autoencoders
  (Kobayashi et al. 2023), which conserves exactly at the seed
- a downstream classifier's logit on `x_0`, whose own LRP supplies the relevance

### The budget

```
R(seed) = R(x_T) + Σ_t [ R_noise,t + R_time,t + R_cond,t + R_bias,t + R_ε,t ] + R_readout-bias
```

Bias absorption and epsilon absorption are kept apart. AttnLRP warns that rules which push bias
relevance back into the inputs to force conservation become unstable in ways that are hard to
diagnose. Three more cases get explicit handling:

- **Classifier-free guidance** `(1+w)·ε(c) − w·ε(∅)` is a linear layer over two forward passes.
  Both passes are propagated.
- **Fixed positional embeddings** get their own sink.
- **Latent diffusion** adds the VAE decoder as a final block with its own rules.

### Memory

Only the trajectory is stored: each `x_t` and each `z_t` (or the exact RNG state). The backward
sweep re-runs step `t`'s forward pass and then its relevance pass, so memory stays at one step.
The forward pass must be deterministic for this to work. If it isn't, the recomputed pass won't
match the one being explained.

### What would show D1 is wrong

- Relevance on the conditioning doesn't concentrate in early steps. Theory ("critical windows",
  Li & Chen 2024) and DifFRACT's measurements (the text stream's share falls from 89.9% at the
  first Flux step to 5.4% at the last) both predict that it should.
- Explanations vary across sampler seeds as much as DAAM's authors feared when they called
  gradient attribution through the sampler intractable.

## D2: commitment-graph LRP for masked diffusion LMs

A masked diffusion LM starts from a fully masked sequence. At each step one bidirectional
transformer pass predicts every masked position, and the sampler commits some of them. In the
standard carry-over samplers (MDLM, LLaDA), committed tokens are never revised, and no hidden
state passes between steps. **The committed tokens are the only channel from one step to the
next**, so chaining relevance through them misses nothing.

### Within a step

Seed the logit of a token committed at step `k` and propagate with the AttnLRP rules. The
bidirectional mask changes where relevance can flow, not the rules. Relevance lands on three
kinds of input: prompt tokens, tokens committed at earlier steps, and `[MASK]` tokens.

### Across steps

Relevance that reaches a token committed at an earlier step `j` is not an endpoint. It is
re-seeded at step `j`, onto the logit that produced that token. The steps are walked in reverse.
Each step seeds every token committed there, with the relevance accumulated for it so far, in
**one** relevance pass: with the forward activations fixed, every rule in use (epsilon,
identity, uniform, softmax) is linear in the incoming relevance. Steps that commit nothing need
no pass. What reaches the prompt is the explanation. Relevance on `[MASK]` tokens has no earlier
step to go to, so it is the mask sink of that step.

Tokens committed in the same step can't attribute to each other, so the graph is a DAG layered by
step. The top seed is the summed log-probability of the tokens being explained, as in DLIG and
the attribution-contract paper. DLIG found that a fixed reference string correlated with infill
quality far better than the model's own tokens (r = 0.197 against 0.013), so a fixed-target
seed is offered as a check.

### The commit is "the last write"

The core assumption is that relevance passes unchanged through the discrete commit (an argmax,
or a sample). The attribution-contract paper supports it: for a masked diffusion model whose
unmasking order is replayed rather than recomputed, Integrated Gradients' completeness residual
falls from 97% at 1 step to 18% at 256 steps.

Not every sampler commits for good. DPLM's default sampler re-masks the lowest-confidence
positions at every step, including ones it has already decoded, and every other position takes
the new prediction. ReMDM also re-masks. So the re-seed point is defined as **the last step that
wrote the token at that position**. Under a carry-over sampler that's the commit step. Under a
remasking sampler, a revoked position holds `[MASK]` and feeds the mask sink until it's written
again.

### Limits

- **Which token, not when.** Which positions get committed at a step is decided by a top-k over
  every masked position, so relevance from the committed logit explains which token was chosen,
  not why it was committed then. The model often settles on an answer well before committing it.
  Commit times therefore date decisions late.
- **Greedy decoding.** Sampling can commit a low-probability token, so explanation runs use greedy
  decoding.
- **Leakage compounds.** Every hop leaks a little, so the absorbed relevance is reported per hop.
- **The mask sink is large and moves.** Mask tokens make up a third of attention-sink tokens in
  LLaDA and Dream, and sinks move between steps. Results are reported with and without sink
  positions.
- **DPLM zeroes mask embeddings.** ESM-style token dropout zeroes the `<mask>` embedding at
  inference, so input-times-relevance rules give exactly zero relevance there. DPLM's mask sink
  is measured at the first residual-stream input instead.
- **Model-specific bookkeeping.** Dream predicts position `j` from hidden state `j−1`, so its
  seed sits one position earlier. LLaDA's classifier-free guidance combines two passes linearly,
  like D1's.

### What would show D2 is wrong

- Its prompt relevance disagrees systematically with Integrated Gradients from prompt to output
  under the same frozen commit order.
- Deleting the most relevant prompt tokens (with the trajectory replayed) does no better than
  deleting random ones.
- Patching the highest-relevance tokens moves later commits no more than patching low-relevance
  ones. This is a causal check in the style of TimeROME-DLM.

## D3: adaLN and adaLN-Zero

A diffusion transformer block with adaLN-Zero computes

```
y = LN(x) · (1 + γ(c)) + β(c)
x_out = x + α(c) · Block(y)
```

The modulation vectors `γ`, `β` and `α` come from a small MLP on the conditioning (the timestep,
the class or the prompt). In a DiT the conditioning enters **only** here. So whether these
products give relevance to the conditioning decides whether the explanation can say anything
about the conditioning at all.

AttnLRP's uniform rule splits a product half to each factor. It assumes the product is zero when
both factors are, and `LN(x)·(1+γ)` isn't such a product. D3 therefore writes the modulation as a
three-term sum, `LN(x) + LN(x)·γ + β`, and routes it with the same absolute-ratio split as D1.
A switch then picks how the product term is treated:

| Component | Split mode (explain the conditioning) | Detach mode (explain the input only) |
|---|---|---|
| `LN(x)` term | to `x` through the LayerNorm identity rule | the same |
| `LN(x)·γ` term | half to `LN(x)`, half to `γ` → MLP → conditioning | all to `LN(x)` |
| `β` term | to `β` → MLP → conditioning | bias sink |
| `x + α·Block` | ratio split at the residual; `α·Block` split half and half | ratio split; `α` held constant |
| Sinks reported | timestep, class or prompt | bias only |

The precedents disagree on which mode is better. MambaLRP found that splitting a gate beats
detaching it (1.978 against 1.577 on SST-2). Arras et al. found that, for LSTM gates, giving
the gate nothing beats any split, which scored near random. adaLN differs from both, because its
modulating factor is side information rather than the input being explained. Both modes are
therefore implemented, and a planted test decides the default.

Only some targets need D3. MDLM-owt was trained with time conditioning switched off, so its
modulation vectors are constants. They fold exactly into the LayerNorm and projection weights
at load time, and the folded model needs no D3. SEDD-small (which really uses its noise level)
and image DiTs do need it.

### What would show D3 is wrong

On a class-conditional DiT trained so that the class controls only one region of the image,
split mode fails to send that region's relevance to the class embedding, or it scores near random
on perturbation tests. Detach mode then becomes the default.

## Checks that need no reference

Each check is cheaper than the next. A rule must pass all of them before any result is reported.

| Check | Expected |
|---|---|
| DDIM (`σ = 0`) | The noise sink is exactly zero |
| One step (`T = 1`) | Reduces to ordinary single-network LRP |
| fp64, biases zeroed, every factor detached | Equals autograd gradient × input |
| Uniform rule in fp64 without biases | Each factor gets half of gradient × input |
| MDLM-owt, detach mode | Matches LRP on the folded model exactly |
| Budget | Input relevance plus every sink equals the seed, to fp32 tolerance |
| Randomize only the denoiser | The network's share changes. The skip share alone can survive, so a map that survives is not a pass |
| Integrated Gradients through the sampler | Totals agree with the budget (IG is complete) |
| Planted tasks | A copy task for D2, a class-controls-one-region task for D1 and D3: relevance lands where the task put it |
| Faithfulness | Deletion and insertion beat random, gradient × input and attention baselines |

## Models

| Model | Size | Used for | Notes |
|---|---|---|---|
| `kuleshov-group/mdlm-owt` | 170M, Apache-2.0 | D2, first target | Carry-over sampler, no time input, adaLN folds away |
| `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` | 0.75B, Apache-2.0 | D2 | Stock Qwen3 with bidirectional attention |
| `airkingbd/dplm_150m`, `dplm_650m` | 149M, 652M | D2 under remasking | Same weight names as ESM-2; `.bin` files go through `tools/convert/pickle_to_safetensors.py`; no license tag on the cards |
| `louaaron/sedd-small` | 170M | D3 | Real noise-level conditioning |
| `diffusionfamily/diffugpt-m` | 355M | D2 against DLIG | The only model with published DLIG numbers |
| A small class-conditional DDPM or DiT trained here | small | D1, D3 | A planted task gives the ground truth no public checkpoint has |

LLaDA-8B and Dream-7B need over 30 GB for fp32 weights alone, so they're out of reach locally.
