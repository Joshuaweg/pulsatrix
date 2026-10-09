# Deep Learning Modules and Layers

This is the core of pulsatrix: tensors, layers, optimizers and losses for building and
training neural networks in C++. Start here to build a model. Every other section, from data
loading to interpretability, builds on these types.

Everything is a concrete C++ type you can read and step into in a debugger:

- a `Tensor` owns a buffer on a device through a `DeviceBackend*` (CPU, CUDA or HIP/ROCm);
- `Module` subclasses implement layers as plain tensor-in, tensor-out operations;
- an opt-in `ComputationGraph` and `Autograd` (automatic differentiation) record `Node`s when
  you run modules through `Module::forward_traced()`. The explainers use this path.

Every `Module` must implement an LRP (Layer-wise Relevance Propagation) rule, because
`propagate_relevance()` is pure virtual. All modules support the epsilon rule. `LinearModule`
and `Conv2DModule` also support the gamma, alpha-beta (ZPlus) and z-box rules. To run LRP over a
whole network, see [Layer-wise Relevance Propagation](../interpretability/lrp.md).

## What's inside

- **Core**: `Tensor`, `Shape`, `Module`, `ComputationGraph`/`Autograd`, `Node`, `OpType`,
  `DeviceBackend` (`CPUBackend`/`CUDABackend`/`HIPBackend`)
- **Layers**: `LinearModule`, `Conv2DModule`, `ReluModule`, `ActivationModule` (ReLU, tanh,
  sigmoid, SiLU, exact GELU or GELU-tanh, chosen at construction), `FlattenModule`,
  `SequentialModule`, `DropoutModule`, `EmbeddingModule`, `ResidualModule`; normalization
  (`LayerNormModule`/`RMSNormModule`/`GroupNormModule`/`BatchNormModule`, with `BatchNormFold`
  to fold an eval-mode BatchNorm into the layer before it); pooling
  (`MaxPool2DModule`/`AvgPool2DModule`/`AdaptiveAvgPool2DModule`). `Conv2DModule` takes an
  optional `stride` and zero `padding`, as in `torch.nn.Conv2d`, and every LRP rule handles both.
  Its im2col patches use a bounded workspace (`set_max_workspace_bytes()`, 16 MiB by default):
  large batches run in chunks, with identical results (see [GPU profiling](../gpu-profiling.md#hip-7-conv2d-in-batch-chunks)).
  So does `MaxPool2DModule`, whose windows may overlap (ResNet's `MaxPool2d(3, 2, 1)`).
  `ResidualModule` takes an optional shortcut module, such as ResNet's downsampling convolution.
- **Vision models**: `TorchvisionResNet` and `TorchvisionVGG` (ResNet18/34 and VGG11 to VGG19)
  load torchvision's published ImageNet weights. See [Vision models](#vision-models-resnet-and-vgg).
- **Sequence & attention**: `RNNModule`/`LSTMModule`/`GRUModule`, `SoftmaxModule`,
  `RoPEModule`, `MultiHeadAttentionModule`, `SwiGLUModule`, `FeedForwardModule`,
  `TransformerBlock`, `EncoderBlock`, `MambaModule`, `RWKVModule`, `RetNetModule`
- **Attention for pretrained LLMs**: `MultiHeadAttentionModule` and `TransformerBlock` take an
  `AttentionConfig` with the settings Hugging Face checkpoints use: grouped-query attention
  (`num_kv_heads`), a `head_dim` independent of `d_model`, optional projection biases, a causal
  mask, the RoPE pair layout (`RoPELayout::RotateHalf` for Llama-family models) and base
  (`rope_theta`), and QK-Norm. `set_key_padding_mask()` masks padding tokens and
  `set_position_offset()` shifts RoPE positions. `LinearModule` can be built without a bias.
  The settings Gemma 3 adds (LLM-9) are:
  - `AttentionConfig::sliding_window`, so each query sees only its last positions
  - `score_scale` (Gemma's `query_pre_attn_scalar^-0.5`)
  - `qk_norm_weight_offset`
  - `TransformerBlockOptions`, which gives a GeGLU MLP (`GatedActivation::GeluTanh`), sandwich
    norms around attention and the MLP, and `(1 + w)` RMSNorms (`RMSNormModule::set_weight_offset`)
  `TiedLMHeadModule` is the output layer of models with `tie_word_embeddings`: it computes logits
  with an `EmbeddingModule`'s table, which stays a single parameter whose gradient sums both
  uses.
- **Text generation** (`generation.hpp`): `Generate(model, prompt, config)` decodes greedily or
  samples, with temperature, top-k, top-p, min-p, a repetition penalty and EOS tokens (Hugging
  Face's `GenerationConfig` names and order). Sampling is seeded (`config.seed`, or the global
  seed). `MakeNextTokenLogits` turns a module stack such as embedding → transformer blocks →
  `TiedLMHeadModule` into the model function. Each result also gives every new token's
  log-probability under the model.
- **KV cache** (`kv_cache.hpp`): `MultiHeadAttentionModule::forward_cached` and
  `TransformerBlock::forward_cached` process only new positions against a preallocated
  `KVCache`. `MakeCachedNextTokenLogits(embedding, blocks, head, backend, max_length)` uses them
  for generation, reusing the cache for any prefix it has already seen; on CPU, 128 tokens from a
  4-block, 128-wide model take 47 ms instead of 3.3 s.
- **Optimizers & losses**: `SGDOptimizer` (momentum, Nesterov), `AdamOptimizer`,
  `AdamWOptimizer`, all with parameter groups; `MSELoss`, `CrossEntropyLoss`,
  `TokenCrossEntropyLoss`, `BCEWithLogitsLoss`, `KLDivergenceLoss`, `CalibrationLoss`
- **Generative building blocks**: `Reparameterize` (VAE), `NoiseSchedule` and
  `SinusoidalTimestepEmbedding()` (diffusion)
- **Training utilities**: `ClipGradNorm`, `LRSchedule`/`LRScheduler`, parameter groups
  (`param_groups.hpp`), checkpoints (`SaveCheckpoint`/`LoadCheckpoint`, `safetensors.hpp`),
  `set_seed`/`set_deterministic`, and `MetricsSink`/`NoOpMetricsSink` (where `train_step` logs
  its loss)
- **Selection**: `top_k()`, the k largest or smallest entries of every row along the last
  dimension, with their indices, on any device. NaN ranks above every number and ties keep the
  lower index first, so every backend selects the same entries in the same order.
- **Matrix decompositions** (CPU, for matrices up to a few hundred wide): `SymmetricEigen`,
  `PowerIteration`, `QR` and `SVD`, the building blocks for PCA, stable rank and orthonormal
  projections. Vector signs are fixed (each vector's largest entry is positive) and values come
  largest first, so the same matrix always gives the same answer.
- **Ready-made examples**: `XorNetwork` (a tiny MLP that learns XOR), `MnistConvNet` (a
  Conv2D MNIST classifier) and `MnistIdxLoader` (reads the MNIST IDX files)

The CPU backend is always built, and its large matrix multiplies use every core, splitting the
work so results are identical to a single-threaded run. To build the GPU backends, configure
CMake with `-DPULSATRIX_ENABLE_CUDA=ON` or `-DPULSATRIX_ENABLE_HIP=ON`.

Every layer and loss checks that the tensors it's given live on its own device, and throws
`std::invalid_argument` if not; move a tensor first with `Tensor::to()`. `EmbeddingModule` is the
exception: it accepts its indices from any device.

Full API reference: [Doxygen: Deep Learning Modules and Layers](../api/group__dl__modules.html)

## How to implement

### Building and training a small network

A network is a plain C++ object with `Module` fields. Training is explicit, with no hidden
tape: you call `forward()` on each module, compute the loss, call each module's `backward()` in
reverse order, then call `optimizer.step(module)` for each layer with parameters.

`XorNetwork` packages that loop as `train_step()`:

```cpp
#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/xor_training_example.hpp"

using namespace pulsatrix;

CPUBackend backend;
XorNetwork net(&backend);  // Linear(2,4) -> ReLU -> Linear(4,1)
AdamOptimizer optimizer(0.01f, &backend);
NoOpMetricsSink sink;      // discards the logged loss

Tensor input(Shape({1, 2}), &backend, {1.0f, 0.0f});
Tensor target(Shape({1, 1}), &backend, {1.0f});

float loss = net.train_step(input, target, optimizer, sink, /*step=*/0);
Tensor pred = net.forward(input);  // forward pass only
```

**What's happening:** `XorNetwork` owns two `LinearModule`s and a `ReluModule` as plain
members (see `include/pulsatrix/xor_training_example.hpp`). `forward()` chains
`linear1 → relu → linear2`. `train_step()` does five things:

1. zeroes the gradients;
2. runs `forward()`;
3. computes the `MSELoss`;
4. calls `backward()` on each layer in reverse order;
5. applies one Adam `step()` per `LinearModule` and logs the loss to `sink`.

Runnable version: [`examples/xor_demo.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/xor_demo.cpp)
(CMake target `xor_demo`). See also the [recipe](../recipes/deep-learning/xor_training.md).

### Saving and loading weights

Checkpoints are [safetensors](https://github.com/huggingface/safetensors) files, the format most
Hugging Face models ship in. It holds only numbers, so loading a file can't run code, unlike a
PyTorch pickle.

```cpp
#include "pulsatrix/checkpoint.hpp"

SaveCheckpoint("model.safetensors", model, optimizer);  // also writes model.optim.safetensors
// ... later, or in another process:
LoadCheckpoint("model.safetensors", model, optimizer);  // training resumes exactly
```

The optimizer state that can be saved is `AdamOptimizer`'s or `AdamWOptimizer`'s; SGD's momentum
buffers aren't saved yet.

- Every parameter and buffer (such as BatchNorm's running statistics) is stored under its
  `named_parameters()` / `named_buffers()` name. Leave out the optimizer to save or load just
  the model.
- A resumed run reproduces the original loss curve bit for bit.
- Loading is strict: a missing, unexpected or wrongly shaped entry throws, and nothing is changed.
  Pass `CheckpointLoadOptions{false}` to skip missing and unexpected names. Shapes are always
  checked.
- Each file records a `format_version`. Older versions load through a migration table, and newer
  ones are rejected. A plain safetensors file with matching names loads as version 0.
- The optimizer file records which save of the model it belongs to, so stale optimizer state is
  refused.

The reader underneath (`safetensors.hpp`) treats every file as untrusted, and throws
`std::invalid_argument` for anything outside the format: offsets past the end of the file,
overlapping or missing byte ranges, sizes that overflow, and malformed headers. It's fuzzed under
AddressSanitizer (`tools/fuzz/safetensors_fuzz.cpp`). `SafetensorsFile::Map` memory-maps a file
instead of reading it, so only the parts you use are loaded: a 2.2 GB checkpoint opens in about a
millisecond.

### Hugging Face checkpoints

`hf_model.hpp` reads a model directory as downloaded from the Hub.

```cpp
#include "pulsatrix/hf_model.hpp"

HfModelConfig config = ReadHfConfig("SmolLM2-135M/config.json");
if (!config.unsupported.empty()) { /* features pulsatrix can't run yet, e.g. "sliding-window attention (LLM-9)" */ }
AttentionConfig attention = ToAttentionConfig(config);  // GQA, rotate-half RoPE, biases, QK-Norm

HfCheckpoint weights = HfCheckpoint::Open("SmolLM2-135M");  // model.safetensors or its sharded index
for (const std::string& name : weights.names()) { /* weights.info(name), weights.tensor(name, &backend) */ }
```

- `HfModelConfig` fills in what the file leaves to the model class: `head_dim` from the sizes,
  Qwen2's Q/K/V biases, Qwen3's and Gemma 3's QK-Norm, and `eos_token_id` as a list. It reads
  older (`rope_scaling`, `torch_dtype`) and newer (`rope_parameters`, `dtype`) field names, and a
  multimodal config's `text_config`.
- `unsupported` lists everything the config asks for that pulsatrix can't run yet, each with the
  roadmap item that adds it, including architectures that aren't causal language models. A
  loader should refuse a model whose list isn't empty rather than run it subtly wrong.
- `HfCheckpoint` memory-maps every shard. The index must match its shards exactly (every listed
  tensor present, nothing unlisted), and a shard must be a plain file name in the checkpoint
  directory, so an index can't point outside it.
- Hub weights are usually bf16. `tensor()` widens bf16, fp16 and both fp8 formats to fp32
  exactly, and rounds fp64; integer tensors stay raw bytes (`bytes()`).

**Loading a model.** `LoadCausalLM` builds a `CausalLM` (embedding, transformer blocks, final
RMSNorm, tied or untied head: Llama, SmolLM2, Qwen2/2.5, Qwen3, Gemma 3) from a downloaded model directory
and loads its weights by name:

```cpp
#include "pulsatrix/causal_lm.hpp"

std::unique_ptr<CausalLM> model = LoadCausalLM("SmolLM2-135M", &backend);
Tensor logits = model->forward(ids);                      // (1, L) token ids -> (1, L, vocab)
GenerationResult out = Generate(model->next_token_logits(512), prompt_ids, GenerationConfig{});
```

The real SmolLM2-135M loads in under a second, and its greedy continuations are identical to
transformers'. SmolLM2-135M, Qwen2.5-0.5B, Qwen3-0.6B and Llama-3.2-1B (with Llama 3's RoPE
scaling) all give logits within 1e-3 of transformers'. For tokenizing, generating and explaining
with these models end to end, see the [Language Models guide](../language-models/index.md). `CausalLM` is an ordinary module, so `backward()`,
`propagate_relevance()` and checkpoints work on it. Loading is strict: a config with features
pulsatrix can't run yet is refused, and every parameter must be loaded and every checkpoint
tensor used. `LoadWeights` with a `WeightMapping` manifest (source name, target name, a transpose,
an optional row slice for fused tensors) loads other layouts the same way.

### PyTorch checkpoints (.pt, .pth)

pulsatrix never reads a PyTorch pickle file in C++. A `.pt`, `.pth` or `.pkl` file is a small
program, and loading one can run arbitrary code; malware has been found in published models, and
scanners that look for it get bypassed. Instead, convert the file to safetensors once, in Python,
with PyTorch's restricted loader:

```bash
pip install "torch>=2.6" safetensors
python3 tools/convert/pickle_to_safetensors.py resnet18-f37072fd.pth resnet18.safetensors
# wrote resnet18.safetensors: 102 tensors, 11,699,112 values, dtypes float32, verified
```

The output loads like any safetensors file (`SafetensorsFile::Map`, `LoadWeights`,
`LoadCheckpoint`). What the converter does:
- **Loads safely.** It uses `torch.load(weights_only=True)`, which only rebuilds tensors and
  plain containers. It refuses torch older than 2.6, where `weights_only` could still run code
  (CVE-2025-32434).
- **Refuses what it can't load safely**, with the reason:
  - TorchScript archives;
  - files that need any other Python object;
  - sparse, quantized and complex tensors.

  There is deliberately no unsafe mode. If a file you trust needs more, load it in a separate
  environment and re-save `model.state_dict()` with `torch.save`.
- **Finds the weights.** A training checkpoint such as `{"epoch": ..., "state_dict": {...},
  "optimizer": ...}` is unwrapped automatically, if exactly one of `state_dict`, `model`,
  `model_state_dict`, `module`, `net` or `ema` holds tensors. Otherwise pass `--key` with a
  dotted path. Nested dicts and lists become dotted names; numbers and strings are skipped and
  listed.
- **Options:**
  - `--strip-prefix module.` removes DataParallel's prefix.
  - `--float32` casts floating-point tensors to float32.
  - `--force` allows overwriting the output; without it, an existing file is left alone.
- **Writes clean tensors.** Each tensor is made contiguous and gets its own bytes: tied weights
  are written twice. Its dtype is kept: pulsatrix reads bf16 and fp16 by upcasting, and integer
  buffers such as `num_batches_tracked` as raw bytes.
- **Verifies.** The source file's SHA-256 goes into the metadata, and the output is read back and
  compared bit for bit.

The converted torchvision ResNet18 reads back in pulsatrix with every one of its 11.7 million
values matching PyTorch's.

### Vision models: ResNet and VGG

`TorchvisionResNet` and `TorchvisionVGG` (`vision_models.hpp`) rebuild torchvision's ResNet
(basic blocks: ResNet18 and ResNet34) and VGG (without BatchNorm: VGG11 to VGG19) from pulsatrix
layers. They use torchvision's parameter names, so the published ImageNet weights load once
converted:

```bash
curl -LO https://download.pytorch.org/models/resnet18-f37072fd.pth
python3 tools/convert/pickle_to_safetensors.py resnet18-f37072fd.pth resnet18.safetensors
```

```cpp
#include "pulsatrix/vision_models.hpp"

TorchvisionResNet model(TorchvisionResNet::ResNet18(), &backend);  // or TorchvisionVGG::VGG16()
LoadTorchvisionWeights(model, "resnet18.safetensors");  // Linear weights are transposed for you
model.set_training(false);                              // BatchNorm's running statistics
Tensor logits = model.forward(x);  // x: (N, 3, 224, 224), normalized with ImageNet's mean and std
```

- **The input** is torchvision's: resize the short side to 256, crop the center 224x224, scale
  to [0, 1], subtract the mean (0.485, 0.456, 0.406) and divide by the standard deviation
  (0.229, 0.224, 0.225), per channel.
- **Smaller variants** come from the config: `ResNetConfig{blocks, width, num_classes}` and
  `VGGConfig{features, pool_size, hidden, num_classes}`. The tests use a ResNet of width 4 and
  a VGG with 8 channels.
- **Matches PyTorch.** On the published weights, the logits agree with torchvision's to 1e-4
  (relative to the largest), on CPU and GPU.

To explain one, see [LRP on ImageNet models](../recipes/interpretability/imagenet_lrp.md): fold
the ResNet's BatchNorms, then explain through `model.layers()`.

### Reproducibility

Every random choice in pulsatrix comes from a seed, and nothing is seeded from the clock.
`set_seed` sets one global seed (default 0). Anything built without its own seed, such as a
`DropoutModule`, `LinearProbe`, `SparseAutoencoder` or a shuffling `DataLoader`, draws a distinct
seed from it in construction order:

```cpp
#include "pulsatrix/determinism.hpp"

set_seed(1234);  // same seed + same construction order = same weights, masks and shuffles
DropoutModule a(0.5f, &backend), b(0.5f, &backend);  // different masks, both reproducible
DropoutModule c(0.5f, &backend, /*seed=*/7);           // an explicit seed ignores the global one
```

Deterministic mode is on by default (`set_deterministic(false)` turns it off). pulsatrix's own
kernels use no atomics, so they always give the same result. In deterministic mode the GPU
backends also forbid atomics in hipBLAS and cuBLAS, so matrix products are reproducible too.
Random draws that use standard-library distributions (`std::normal_distribution` and friends)
are reproducible on one platform, but not between libstdc++ and MSVC.

### Parameter groups and weight decay

`SGDOptimizer` and `AdamOptimizer` take a learning rate and weight decay per group of
parameters, chosen by name. Each parameter joins the first group that selects it; the rest use
the optimizer's own settings. The usual transformer recipe decays the weight matrices but not the
biases or normalization scales:

```cpp
#include "pulsatrix/param_groups.hpp"

AdamOptimizer adam(3e-4f, &backend);
adam.set_weight_decay(0.01f);  // the default group: every parameter no group selects
adam.set_param_groups({
    {"no decay", param_select::one_dimensional(), 3e-4f, 0.0f},             // biases, norm scales
    {"head", param_select::name_prefix("blocks.11"), 1e-3f, 0.01f},         // a faster last block
});
```

Weight decay works as in PyTorch's SGD and Adam: the step uses `grad + weight_decay * value`,
and the stored gradient is left unchanged. A group's `learning_rate` and `weight_decay` can be
changed between steps through `param_groups()`.

For transformer fine-tuning, use `AdamWOptimizer` instead (decoupled weight decay, Loshchilov
and Hutter). It shrinks the weights directly, `w <- (1 - lr * wd) * w`, rather than adding the
decay to the gradient, where Adam's normalization would rescale it. Its default weight decay is
0.01, as in PyTorch. `SGDOptimizer(lr, momentum, nesterov)` adds momentum and Nesterov momentum,
with the same buffer rule as `torch.optim.SGD`.

### Learning-rate schedules

`LRSchedule` gives a multiplier per step, with the formulas of Hugging Face's
`get_*_schedule_with_warmup`: `Constant(warmup)`, `Linear(warmup, total)` and
`Cosine(warmup, total, min_ratio)`. Warmup ramps from 0 to 1, then linear decays to 0 at `total`
and cosine to `min_ratio`. `LRScheduler` applies one to an optimizer: every rate, including each
parameter group's, becomes its starting rate times the multiplier.

```cpp
#include "pulsatrix/lr_scheduler.hpp"

AdamWOptimizer opt(3e-4f, &backend);
LRScheduler scheduler(opt, LRSchedule::Cosine(/*warmup=*/100, /*total=*/10000));
for (int64_t step = 0; step < 10000; ++step) {
    // ... forward, backward, ClipGradNorm ...
    opt.step(model);
    scheduler.step();
}
```

Set up parameter groups before building the scheduler. To resume, call
`scheduler.set_last_step(n)`.

### Token losses and gradient accumulation

`TokenCrossEntropyLoss` takes logits `(..., classes)` and integer targets with the same leading
shape, and skips targets equal to `-100`, PyTorch's padding convention. By default it averages
over the batch's real tokens.

To accumulate gradients over several micro-batches, don't average each micro-batch's own mean.
When micro-batches hold different numbers of real tokens, that weights tokens unequally, which is
the bug Hugging Face Trainer fixed in v4.46. Instead, divide every micro-batch by the token count
of the whole window:

```cpp
#include "pulsatrix/token_cross_entropy_loss.hpp"

int64_t total = 0;
for (const Tensor& t : window_targets) total += CountTargetTokens(t);
for (size_t k = 0; k < window_inputs.size(); ++k) {
    Tensor logits = model.forward(window_inputs[k]);
    (void)loss.forward(logits, window_targets[k], static_cast<float>(total));
    (void)model.backward(loss.backward());  // gradients accumulate across micro-batches
}
(void)ClipGradNorm(model, 1.0f);  // returns the norm before clipping
optimizer.step(model);
```

The accumulated gradients then equal those of one batch holding every micro-batch.

### Gradient clipping

`ClipGradNorm(model, max_norm)` scales all trainable gradients by one factor so their global L2
norm is at most `max_norm`, as `torch.nn.utils.clip_grad_norm_` does. It returns the norm before
clipping, which is worth logging. Call it after accumulating gradients and before the optimizer
step:

```cpp
#include "pulsatrix/grad_clipping.hpp"

const float norm = ClipGradNorm(model, 1.0f);
if (std::isfinite(norm)) {
    optimizer.step(model);  // a NaN or infinite norm leaves the gradients alone: skip this step
}
```

### Freezing parameters

Every parameter has a name (`named_parameters()`), and `set_requires_grad` freezes or unfreezes
parameters by name. A name selects that parameter and everything under it:

```cpp
TransformerBlock block(64, 4, 256, &backend);
block.set_requires_grad(false);              // freeze everything
block.set_requires_grad(true, "mha.q_proj");  // then train only the query projection
```

A frozen parameter is never changed by any optimizer (`SGDOptimizer`, `AdamOptimizer`,
`AdamWOptimizer`), `ClipGradNorm` leaves it out, and `backward()` doesn't add to its gradient. The gradient passed back to the previous layer is exactly the same
as without freezing. `LinearModule`, `Conv2DModule` and `EmbeddingModule` skip computing a
frozen weight's gradient altogether, which is where fine-tuning saves time. A name that matches
nothing throws `std::invalid_argument`, so a typo can't leave the model silently trainable.

### Encoder layers

`EncoderBlock` is one transformer encoder layer whose layout is chosen with
`EncoderBlockOptions`:

- `norm`: `NormType::LayerNorm` or `RMSNorm`.
- `norm_position`:
  - `NormPosition::Pre` gives `x + f(norm(x))` (ESM, GPT-2, Llama).
  - `Post` gives `norm(x + f(x))` (BERT, the original Transformer).
- `mlp`:
  - `MlpType::Plain` is `FeedForwardModule`: Linear → activation → Linear, as in BERT and ESM. Its default activation is exact GELU.
  - `Gated` is `SwiGLUModule`: SwiGLU, or GeGLU with `gate_activation`.

Attention comes from the same `AttentionConfig` as `TransformerBlock`; an encoder leaves
`causal` off. The defaults are ESM-2's layer:

```cpp
AttentionConfig attention;
attention.d_model = 320;
attention.num_heads = 20;
attention.rope_layout = RoPELayout::RotateHalf;
EncoderBlock esm_layer(attention, 1280, &backend);  // pre-LayerNorm (eps 1e-5), GELU MLP

EncoderBlockOptions bert;
bert.norm_position = NormPosition::Post;
bert.norm_eps = 1e-12f;
attention.use_rope = false;  // BERT adds learned positions before the first layer
EncoderBlock bert_layer(attention, 1280, &backend, bert);
```

- **Checked against `transformers`.** The output and the input gradient match the library's own
  `EsmLayer` and `BertLayer` to 2e-5 (`tools/generate_encoder_reference.py`).
- **The decoder layout reproduces `TransformerBlock`.** Pre-RMSNorm with a gated SiLU MLP gives
  `TransformerBlock`'s output, gradients and relevance exactly.
- **Parameter names.** Parameters are named `norm1.*`, `mha.*`, `norm2.*` and `mlp.*`, so
  `set_requires_grad` and weight loading work the same way.
- **LRP.**
  - Each residual add uses the two-term epsilon rule.
  - Norms and activations pass relevance through unchanged, as AttnLRP prescribes.
  - Attention uses its AttnLRP rule.
  - The result is linear in the incoming relevance in both layouts.

### Choosing a normalization layer

`LayerNormModule`, `RMSNormModule`, `GroupNormModule` and `BatchNormModule` all implement the
same `Module` contract (`forward()`, `backward()`, `propagate_relevance()`,
`named_parameters()`). That makes them interchangeable in a `SequentialModule`. Pick one by
what it normalizes over:

- `LayerNormModule` and `RMSNormModule`: each row's features (the last dimension).
  RMSNorm skips the mean-centering.
- `GroupNormModule`: groups of channels, per example.
- `BatchNormModule`: each channel across the whole batch and spatial dimensions. In training
  mode it uses the current batch and updates running statistics (PyTorch's rule, momentum 0.1).
  After `set_training(false)` it uses the running statistics, so each sample's output no longer
  depends on the rest of its batch. Switch to eval mode before explaining a model.

## Recipes

- [XOR training walkthrough](../recipes/deep-learning/xor_training.md)
- [RNN vs. LSTM vs. GRU on a parity task](../recipes/deep-learning/sequence_models_rnn_lstm_gru.md)
- [Residual connections and normalization layers](../recipes/deep-learning/residual_and_norm_layers.md)
- [Full fine-tuning: pretrain, save, reload, fine-tune](../recipes/deep-learning/tagger_finetune.md)
