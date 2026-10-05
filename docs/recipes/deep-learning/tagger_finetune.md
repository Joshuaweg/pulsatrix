# Recipe: Full Fine-Tuning

**What you'll build:** a tiny transformer that tags each token of a sequence. You'll pretrain it on
one rule, save it, reload it and fine-tune every parameter on a related rule, using the whole
training stack: AdamW, parameter groups, a warmup-plus-cosine schedule, gradient accumulation and
gradient clipping. Full fine-tuning is the baseline every other method (LoRA and its variants) is
compared against.

CMake target: `tagger_finetune_recipe` (`examples/recipes/tagger_finetune.cpp`).

Run it: `./build/tagger_finetune_recipe [steps] [learning_rate]` (Windows:
`build\Release\tagger_finetune_recipe.exe`). Defaults are 300 steps at 0.01.

!!! note "Why a self-pretrained model"
    Importing real pretrained models (SmolLM2, ResNet18) is planned for v1.2. Until then the
    "pretrained" model is one the recipe trains itself. Only step 1 changes once real checkpoints
    load. The task is per-token tagging rather than next-token prediction because attention has
    no causal mask yet; without one, a language model could see the token it's asked to predict.

## The model and the task

`TinyTagger` (`tagger_finetune_example.hpp`) is `EmbeddingModule` → `TransformerBlock` →
a per-token `LinearModule` head, mapping `(N, L)` token ids to `(N, L, 4)` logits. Sequences hold
5–8 real tokens followed by padding (target `-100`), so every batch is ragged.

- **Rule A (pretraining):** tag = (token + previous token) mod 4.
- **Rule B (fine-tuning):** tag = (token − previous token) mod 4.

Both rules need the same skill: attend to the previous token and combine it with the current
one. Pretraining on A should therefore help with B.

## Code

```cpp
CPUBackend backend;
FineTuneConfig config;  // 300 steps, lr 1e-2, 2 micro-batches of 8, warmup 10, wd 0.01, clip 1.0

// 1. "Pretrain" on rule A, then save.
TinyTagger pretrained(&backend);
InitTagger(pretrained, 3);
TrainTagger(pretrained, TaggingRule::SumWithPrevious, config, &backend);
SaveCheckpoint("tagger_pretrained.safetensors", pretrained);

// 2. Reload and fine-tune every parameter on rule B.
TinyTagger tuned(&backend);
LoadCheckpoint("tagger_pretrained.safetensors", tuned);
TrainTagger(tuned, TaggingRule::DifferenceWithPrevious, config, &backend);
```

Each training step of `TrainTagger` does the following:

```cpp
AdamWOptimizer optimizer(config.learning_rate, backend, config.weight_decay);
optimizer.set_param_groups({{"no decay", param_select::one_dimensional(), config.learning_rate, 0.0f}});
LRScheduler scheduler(optimizer, LRSchedule::Cosine(config.warmup, config.steps));

// per step: count the window's real tokens, then accumulate the micro-batches
int64_t tokens = 0;
for (const TaggingBatch& b : window) tokens += CountTargetTokens(b.targets);
optimizer.zero_grad(tagger);
for (const TaggingBatch& b : window) {
    TokenCrossEntropyLoss loss(backend);
    (void)loss.forward(tagger.forward(b.inputs), b.targets, static_cast<float>(tokens));
    (void)tagger.backward(loss.backward());
}
(void)ClipGradNorm(tagger, config.max_grad_norm);
optimizer.step(tagger);
scheduler.step();
```

To pause and resume a run, set `stop_after` and `checkpoint_path`, then `resume_from`. A
resumed run reproduces the uninterrupted one exactly: the checkpoint stores the model, the
AdamW moments and the schedule step, and each step's data depends only on the step number.

## Expected output

```
pretrain (rule A) loss:  step 0: 1.397  step 60: 0.856  step 120: 0.272  step 180: 0.184  step 240: 0.169  final: 0.158
rule A accuracy after pretraining: 0.954
rule B accuracy before fine-tuning: 0.494
fine-tune (rule B) loss:  step 0: 3.189  step 60: 0.245  step 120: 0.148  step 180: 0.056  step 240: 0.020  final: 0.016
rule B accuracy after fine-tuning:  0.985
rule B accuracy from scratch, same budget: 0.862
```

The printed losses can differ slightly with other compilers; the data and initialization use a
portable generator, so the trends don't.

## What's happening

- **The first fine-tuning loss (3.19) is worse than chance (ln 4 ≈ 1.39).** The pretrained model
  is confident about rule A, which contradicts rule B on most tokens. It still learns rule B
  faster than a model trained from scratch, because the skill it needs is already there.
- **Pretraining helps.** After the same 300 steps, the fine-tuned model tags 98.5% of tokens
  correctly; the same model trained from scratch tags 86.2%.
- **Gradient accumulation divides by the window's token count.** Micro-batches hold different
  numbers of real tokens, so averaging their means would weight tokens unequally.

See also: [Deep Learning Modules and Layers](../../deep-learning/index.md) (parameter groups,
schedules, clipping, token losses, checkpoints).
