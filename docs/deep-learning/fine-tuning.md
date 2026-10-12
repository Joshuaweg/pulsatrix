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

### Explaining an adapted model

An adapted layer's `propagate_relevance()` treats it as the single linear map
W + (alpha / r) A B, so every LRP rule, AttnLRP and the other explainers run on a LoRA model
unchanged. The relevance is the same whether the adapter is merged or not, up to float32
rounding. Splitting relevance between the base path and the adapter path, to see what the
fine-tuning changed, is the next roadmap item (TRN-8).

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

### Limits

- Only `LinearModule` layers take adapters: not convolutions, embeddings or the tied
  language-model head.
- The models that report their layers to `ApplyLora()` (`named_children()`) are `CausalLM`,
  `EncoderLM`, the transformer and encoder blocks, attention, the MLP blocks,
  `SequentialModule` and `ResidualModule`. A custom container can override `named_children()`
  the same way.
- rsLoRA, LoRA+ and DoRA are roadmap item TRN-13.
