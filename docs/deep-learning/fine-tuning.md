# Parameter-Efficient Fine-Tuning

Full fine-tuning trains every weight of a model (see the
[full fine-tuning recipe](../recipes/deep-learning/tagger_finetune.md)). Parameter-efficient
methods leave the pretrained weights frozen and train a small number of new parameters instead,
so a fine-tune is a few megabytes, many fine-tunes can share one base model, and the change the
fine-tuning made is small enough to inspect.

## LoRA

LoRA (Hu et al., 2021) adds a low-rank update to a Linear layer. The layer computes

```
y = x W + b + (alpha / r) x A B
```

with the pretrained W and b frozen. Only A, shaped `(in, r)`, and B, shaped `(r, out)`, train. B
starts at zero, so the adapted model starts out exactly as the base model. With rank r = 8 on a
576 → 576 projection, that is 9,216 trained values instead of 331,776.

In pulsatrix the adapter lives inside the `LinearModule` (`attach_lora()`). Every model's
projections are `LinearModule`s, so `ApplyLora()` finds them by name and attaches an adapter to
each. The model then trains, generates (with the KV cache) and explains exactly as before.

```cpp
#include "pulsatrix/lora.hpp"

std::unique_ptr<CausalLM> model = LoadCausalLM("SmolLM2-135M", &backend);

LoRAConfig lora;           // PEFT's defaults: rank 8, alpha 8, no dropout
lora.rank = 16;
lora.alpha = 32;
std::vector<std::string> adapted = ApplyLora(*model, lora);  // every projection, then freezes the rest

AdamWOptimizer opt(2e-4f, &backend, /*weight_decay=*/0.0f);  // about 10x a full fine-tune's rate
TokenCrossEntropyLoss loss(&backend);
for (/* each batch */;;) {
    opt.zero_grad(*model);
    const float value = loss.forward(model->forward(tokens), targets);
    (void)model->backward(loss.backward());
    opt.step(*model);  // frozen weights are skipped
}

SaveLoraAdapter(*model, "my-adapter", HuggingFaceMapping(model->config()));
```

`ApplyLora()` returns the names of the layers it adapted (`layers.0.mha.q_proj`, ...). Each
adapter's A and B appear in `named_parameters()` as `<layer>.lora_A.weight` and
`<layer>.lora_B.weight`, so parameter groups, gradient clipping and native checkpoints see
them like any other parameter.

### Which layers, and what learning rate

`LoRAConfig::target_modules` chooses the layers the way PEFT's `target_modules` does: a layer
matches a name if its own name is that name or ends with `.` and that name, so `{"q_proj",
"v_proj"}` selects every query and value projection. A name that matches nothing, or matches a
module that isn't a `LinearModule`, throws.

The default, an empty list, adapts every `LinearModule` except an output head named `lm_head`.
That is PEFT's `"all-linear"`. PEFT's own default for Llama-family models is only `q_proj` and
`v_proj`, but "LoRA Without Regret" (Schulman and Thinking Machines Lab, 2025) found that:

- LoRA on the MLP layers matters most. Attention-only LoRA does worse, even with as many
  trained parameters, and adding attention to MLP LoRA adds little.
- The best learning rate is consistently about **10 times** the best one for full fine-tuning
  of the same model and task.
- With the alpha / r scaling, the best learning rate barely depends on the rank. The paper uses
  alpha = 32.
- Small ranks match full fine-tuning on small datasets and in reinforcement learning. On
  datasets too large for the adapter's capacity, LoRA falls behind full fine-tuning.
- LoRA tolerates large batches less well than full fine-tuning, and a higher rank doesn't fix
  that.

### Initialization and reproducibility

A is initialized as PEFT initializes it, Kaiming-uniform with a = √5, which is uniform on
±1/√in, and B is zero. The values come from `LoRAConfig::seed` and each layer's position in the
model, through raw `mt19937_64` output, so they are the same on every platform. They are not
PyTorch's values: a run seeded the same way in PEFT starts from different A matrices.

`LoRAConfig::dropout` applies dropout to the adapter's input in training mode, as PEFT's
`lora_dropout` does. It defaults to 0.

### Merging, unmerging and turning adapters off

| Call | Effect |
|---|---|
| `MergeLora(model)` | W += (alpha / r) A B in every adapted layer, so inference runs at the base model's cost. While merged, A and B get no gradients. |
| `UnmergeLora(model)` | Subtracts the update again. W comes back up to float32 rounding, as in PEFT. |
| `SetLoraEnabled(model, false)` | Runs every adapted layer as its base layer, so one model gives both the base and the fine-tuned outputs. Not allowed while merged. |
| `RemoveLora(model)` | Removes the adapters. Merged ones stay in the weights (PEFT's `merge_and_unload()` is `MergeLora` then `RemoveLora`). |
| `LoraLayers(model)` | The adapted layers, by name. |

The same calls exist per layer on `LinearModule`: `merge_lora()`, `unmerge_lora()`,
`set_lora_enabled()`, `detach_lora()` and `lora()`.

### Explaining an adapted model: what did the fine-tuning add?

Every LRP rule, AttnLRP and the other explainers run on a LoRA model unchanged. An adapted layer
also splits its relevance in two parts:
- the part that reaches its input through the base weight W;
- the part that reaches it through the adapter (alpha / r) A B.

The two parts sum to the layer's input relevance. After a relevance pass, `LoraRelevanceSplit()`
reports each adapted layer's split. A layer with a large adapter share is one whose explanation
the fine-tuning changed. Within a layer, the adapter's `(N, in)` tensor shows which inputs and
which features changed.

```cpp
LoadLoraAdapter(*model, "my-adapter", HuggingFaceMapping(model->config()));
const Tensor logits = model->forward(tokens);
(void)model->propagate_relevance(target_relevance, LRPRuleConfig{});
for (const LoraLayerRelevance& layer : LoraRelevanceSplit(*model)) {
    std::cout << layer.name << ": base " << layer.base_total << ", adapter " << layer.adapter_total << "\n";
}
```

To read one layer directly, use `LoRAAdapter::last_base_relevance()` and
`last_adapter_relevance()`. Both are cleared by the next `forward()`.

**How the split is computed.** The layer is treated as two parallel paths. The rule runs on both
at once, as on a single Linear layer whose input is x repeated twice and whose weight is W and
(alpha / r) A B stacked. Every rule's denominator therefore sees both paths. What that means
depends on the rule:

- **Epsilon** (the default). Each path gets x_i w_ij R_j / z_j, where z is the whole layer's
  output. The two parts add up exactly to the relevance of the merged layer, so splitting changes
  nothing about the explanation itself.
- **Gamma and AlphaBeta (z⁺).** These rules favour positive weights. The positive part of the
  merged weight, (W + sAB)⁺, is not W⁺ + (sAB)⁺, so the split relevance differs from the merged
  layer's, sometimes by a lot. Both are conservative. They answer different questions: the split
  treats the fine-tune as a separate circuit added next to the base model, and the merged form
  treats the fine-tuned model as one new model.

`LRPRuleConfig::lora_relevance` chooses between them:
- `LoRARelevance::Split` (the default) splits the layer as above.
- `LoRARelevance::Merged` runs the rule on the adapted weight, so an unmerged adapter gives
  exactly the relevance a merged one does.

A merged adapter (`MergeLora()`) can't be split, because its update is already part of W. A
disabled adapter isn't split either. Neither reports a split.

To see what a fine-tune changed, compare the relevance with the adapters off
(`SetLoraEnabled(model, false)`) and on.

### PEFT adapter files

`SaveLoraAdapter()` writes the directory PEFT's `PeftModel.from_pretrained()` reads:

- `adapter_model.safetensors`, with keys such as
  `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`, in PyTorch's layout
  (`lora_A` is `(r, in)` and `lora_B` is `(out, r)`, the transposes of pulsatrix's).
- `adapter_config.json`, with `r`, `lora_alpha`, `lora_dropout` and the adapted modules as
  `target_modules`.

The mapping argument turns pulsatrix's layer names into the base checkpoint's: pass the manifest
the base model was loaded with, `HuggingFaceMapping()` for a `CausalLM` or `EsmMapping()` for an
`EncoderLM`. Without it the keys use pulsatrix's names.

`LoadLoraAdapter()` reads such a directory onto a loaded model, adapter by adapter, with the rank
from the tensors' shapes, and freezes the base model as `ApplyLora()` does. Everything is checked
before the model changes. It refuses what pulsatrix doesn't run, rather than loading it and
computing something else:

- DoRA, rsLoRA and QA-LoRA;
- `fan_in_fan_out` (GPT-2's Conv1D layout);
- trained biases;
- `modules_to_save`;
- per-layer `rank_pattern` or `alpha_pattern`.

**Checked against PEFT.** `tools/golden/make_lora_golden.py` makes an adapter with PEFT 0.21.0
on a tiny Qwen2 model, on every projection. Loaded into pulsatrix:

- the logits match PEFT's to 1e-4, on the CPU and through hipBLAS;
- the gradients of all 28 A and B matrices match PEFT's;
- merging and unmerging leave the logits unchanged;
- `SaveLoraAdapter()` writes back the same keys, shapes and values as PEFT's file.

`ApplyLora()`'s default selects the same layers as PEFT's `"all-linear"`.

### LoRA's limits

- Only `LinearModule` layers take adapters: not convolutions, embeddings or the tied
  language-model head.
- The models that report their layers to `ApplyLora()` (`named_children()`) are `CausalLM`,
  `EncoderLM`, the transformer and encoder blocks, attention, the MLP blocks,
  `SequentialModule` and `ResidualModule`. A custom container can override `named_children()`
  the same way.
- rsLoRA, LoRA+ and DoRA are roadmap item TRN-13.

## LoReFT

LoReFT (Wu et al., 2024, "ReFT: Representation Finetuning for Language Models") leaves every weight
of the model alone. It trains a small edit of the hidden states instead, at a few positions after
chosen layers. At each edited position, the residual stream h after a block becomes

```
h' = h + Rᵀ (W h + b − R h)
```

R is `(r, d)` with orthonormal rows. In the r-dimensional subspace that R spans, the edit replaces
the hidden state's coordinates R h with learned ones, W h + b. Outside the subspace, h doesn't
move. That makes the fine-tune an interpretable object in its own right: a few directions in
representation space and what is written into them. It is also very small, 2rd + r values per
layer.

```cpp
#include "pulsatrix/loreft.hpp"

LoReFTConfig reft;
reft.rank = 4;
reft.layers = {4, 8};            // empty: every layer
reft.first = 3;                  // edit the first 3 and the last 3 prompt positions:
reft.last = 3;                   // pyreft's "f3+l3"
std::vector<LoReFTIntervention*> edits = ApplyLoReFT(*model, reft);  // and freezes the model

for (/* each batch */;;) {
    opt.zero_grad(*model);
    const float value = loss.forward(model->forward(tokens), targets);
    (void)model->backward(loss.backward());
    opt.step(*model);
    OrthonormalizeLoReFT(*model);  // put R back on the orthonormal set
}
SaveLoReFT(*model, "reft.safetensors");
```

The interventions belong to the `CausalLM` (`set_intervention()`, `intervention()`). Their
parameters join `named_parameters()` as `interventions.<layer>.rotate_layer`,
`interventions.<layer>.learned_source.weight` and `interventions.<layer>.learned_source.bias`.
They run in `forward()`, `backward()` and the relevance passes. They also run in generation, which
edits the prompt's positions only, as pyreft does: tokens the model generates aren't edited.

**Keeping R orthonormal.** A gradient step moves R off the set of orthonormal matrices, so call
`OrthonormalizeLoReFT(model)` (or `orthonormalize()` on one intervention) after each optimizer
step. pyreft keeps R orthonormal with PyTorch's orthogonal parametrization instead. For the same
R, the edit, its gradient with respect to R and its relevance are identical, but a training run
takes a different path through parameter space than pyreft's.

**Initialization.** As in pyreft, R starts orthonormal and W and b start like an `nn.Linear`
layer, uniform on ±1/√d. Unlike LoRA, the edit is not zero at the start. The values come from
`LoReFTConfig::seed` through raw `mt19937_64` output.

### Explaining a LoReFT model

The edit is affine in h: h' = h (I + (Wᵀ − Rᵀ) R) + b R. LRP therefore propagates through it
exactly with the epsilon rule. The bias b R takes no relevance, as with every Linear layer here.

Like a split LoRA layer, the edit is treated as two paths: the identity, which keeps h, and the
edit, which rewrites the subspace. The rule runs on both together. `last_edit_relevance()` returns
the edit's part, `(N, L, d)`, which is zero at positions the intervention didn't touch. Large
values show where the fine-tune's rewrite drives the explanation.

### Files

`SaveLoReFT()` writes one safetensors file with, for each layer i, pyreft's state dict in pyreft's
layouts:
- `layer.<i>.rotate_layer` (`(d, r)`);
- `layer.<i>.learned_source.weight` (`(r, d)`);
- `layer.<i>.learned_source.bias`.

The positions are stored as `first` and `last` in the file's metadata. `LoadLoReFT()` reads such a
file. pyreft saves each intervention as a PyTorch `.bin` state dict: convert those with
`tools/convert/pickle_to_safetensors.py`, rename the keys as above, and add the metadata.

**Checked against pyreft.** `tools/golden/make_loreft_golden.py` uses pyreft 0.1.0's own
`LoreftIntervention` on a tiny Qwen2 model, with rank 2 at layers 0 and 1 and positions
"f1+l2". Loaded into pulsatrix:

- the logits match to 1e-4, on the CPU and on the GPU;
- the gradients of R, W and b match;
- greedy generation gives pyreft's tokens, which differ from the base model's.

### LoReFT's limits

- Interventions run on `CausalLM` only, after a block. Other places pyreft can intervene, such as
  attention outputs, aren't supported.
- Every sequence in a batch is edited at the same positions, so batches must not be padded.
- Only the linear activation, pyreft's default, is supported; it is what keeps LRP exact.
- pyreft's dropout on the intervention isn't supported.

## IA³

IA³ (Liu et al., 2022, "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than
In-Context Learning") learns one rescaling vector per adapted layer and nothing else:

```
y = (x W + b) ⊙ l        keys and values: the layer's output, bias included
y = (x ⊙ l) W + b        the MLP's down projection: its input, the inner activations
```

Each vector starts at 1, so the adapted model starts out as the base model. There is one value
per channel: about 58 thousand values for SmolLM2-135M on the default layers, against 4.9
million for rank-16 LoRA on every projection. That makes the fine-tune trivially explainable. Each
entry of l says how much the fine-tune amplified (above 1), damped (between 0 and 1) or flipped
(below 0) one key, value or MLP channel.

```cpp
#include "pulsatrix/ia3.hpp"

std::vector<std::string> adapted = ApplyIA3(*model);  // k_proj, v_proj, down_proj; freezes the rest
// ... train as with LoRA ...
SaveIA3Adapter(*model, "my-ia3", HuggingFaceMapping(model->config()));
```

`IA3Config::target_modules` and `feedforward_modules` choose the layers, with PEFT's matching:
- **Targets.** The default is the paper's: every `k_proj` and `v_proj`, and the MLP's output
  projection, `down_proj` in a `CausalLM` or `fc2` in an `EncoderLM`. PEFT's own default for
  Qwen2 rescales `q_proj` instead of `k_proj`.
- **Feedforward modules.** The layers whose input is rescaled rather than their output. Each one
  must also be a target.

The vector is in `named_parameters()` as `<layer>.ia3_l`: `(out)` for output scaling and `(in)`
for a feedforward layer.

| Call | Effect |
|---|---|
| `MergeIA3(model)` | Folds each vector into its weights: W diag(l), and b ⊙ l for output scaling; diag(l) W for a feedforward layer. |
| `UnmergeIA3(model)` | Divides it back out by l + 1e-8, as PEFT does. The weights return up to float rounding. |
| `SetIA3Enabled(model, false)` | Runs the base model. Not allowed while merged. |
| `RemoveIA3(model)` | Removes the vectors; merged ones stay in the weights. |
| `IA3Layers(model)` | The adapted layers, by name. |

A layer takes an IA³ vector or a LoRA adapter, not both.

**Explaining an IA³ model.** Both forms are the base layer with rescaled weights, so
`propagate_relevance()` runs every rule on the rescaled layer, exactly as if it were merged. There
is nothing to split. Under the epsilon rule, rescaling an output channel doesn't change how that
channel's relevance spreads over the inputs. What changes is how much relevance reaches the channel
from above, which is the fine-tune's effect.

**PEFT files.** `SaveIA3Adapter()` writes PEFT's adapter directory: `base_model.model.<module>.ia3_l`
with shape `(out, 1)` or `(1, in)`, and an `adapter_config.json` with `target_modules` and
`feedforward_modules`. `LoadIA3Adapter()` reads one. It tells a feedforward layer by its vector's
shape and checks that against `feedforward_modules`. It refuses `fan_in_fan_out` and
`modules_to_save`.

**Checked against PEFT.** `tools/golden/make_ia3_golden.py` makes an IA³ adapter with PEFT 0.21.0
on a tiny Qwen2 model, on `k_proj`, `v_proj` and `down_proj`. The vectors are redrawn around 1 so
they matter, and `k_proj` and `v_proj` have biases, which output scaling multiplies too. Loaded
into pulsatrix:

- the logits match PEFT's to 1e-4, on the CPU and on the GPU;
- the gradients of all six vectors match;
- merging and unmerging leave the logits unchanged;
- `SaveIA3Adapter()` writes back PEFT's keys, shapes and values.
