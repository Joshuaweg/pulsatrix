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
- **Layers**: `LinearModule`, `Conv2DModule`, `ReluModule`, `FlattenModule`,
  `SequentialModule`, `DropoutModule`, `EmbeddingModule`, `ResidualModule`; normalization
  (`LayerNormModule`/`RMSNormModule`/`GroupNormModule`/`BatchNormModule`); pooling
  (`MaxPool2DModule`/`AvgPool2DModule`). `Conv2DModule` takes an optional `stride` and zero
  `padding`, as in `torch.nn.Conv2d`, and every LRP rule handles both.
- **Sequence & attention**: `RNNModule`/`LSTMModule`/`GRUModule`, `SoftmaxModule`,
  `RoPEModule`, `MultiHeadAttentionModule`, `SwiGLUModule`, `TransformerBlock`,
  `MambaModule`, `RWKVModule`, `RetNetModule`
- **Optimizers & losses**: `SGDOptimizer`, `AdamOptimizer`; `MSELoss`, `CrossEntropyLoss`,
  `BCEWithLogitsLoss`, `KLDivergenceLoss`, `CalibrationLoss`
- **Generative building blocks**: `Reparameterize` (VAE), `NoiseSchedule` and
  `SinusoidalTimestepEmbedding()` (diffusion)
- **Training utilities**: `MetricsSink`/`NoOpMetricsSink` (where `train_step` logs its loss)
- **Selection**: `top_k()`, the k largest or smallest entries of every row along the last
  dimension, with their indices, on any device. NaN ranks above every number and ties keep the
  lower index first, so every backend selects the same entries in the same order.
- **Matrix decompositions** (CPU, for matrices up to a few hundred wide): `SymmetricEigen`,
  `PowerIteration`, `QR` and `SVD`, the building blocks for PCA, stable rank and orthonormal
  projections. Vector signs are fixed (each vector's largest entry is positive) and values come
  largest first, so the same matrix always gives the same answer.
- **Ready-made examples**: `XorNetwork` (a tiny MLP that learns XOR), `MnistConvNet` (a
  Conv2D MNIST classifier) and `MnistIdxLoader` (reads the MNIST IDX files)

The CPU backend is always built. To build the GPU backends, configure CMake with
`-DPULSATRIX_ENABLE_CUDA=ON` or `-DPULSATRIX_ENABLE_HIP=ON`.

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
AddressSanitizer (`tools/fuzz/safetensors_fuzz.cpp`).

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

### Freezing parameters

Every parameter has a name (`named_parameters()`), and `set_requires_grad` freezes or unfreezes
parameters by name. A name selects that parameter and everything under it:

```cpp
TransformerBlock block(64, 4, 256, &backend);
block.set_requires_grad(false);              // freeze everything
block.set_requires_grad(true, "mha.q_proj");  // then train only the query projection
```

A frozen parameter is never changed by `SGDOptimizer` or `AdamOptimizer`, and `backward()`
doesn't add to its gradient. The gradient passed back to the previous layer is exactly the same
as without freezing. `LinearModule`, `Conv2DModule` and `EmbeddingModule` skip computing a
frozen weight's gradient altogether, which is where fine-tuning saves time. A name that matches
nothing throws `std::invalid_argument`, so a typo can't leave the model silently trainable.

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
