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
  (`MaxPool2DModule`/`AvgPool2DModule`)
- **Sequence & attention**: `RNNModule`/`LSTMModule`/`GRUModule`, `SoftmaxModule`,
  `RoPEModule`, `MultiHeadAttentionModule`, `SwiGLUModule`, `TransformerBlock`,
  `MambaModule`, `RWKVModule`, `RetNetModule`
- **Optimizers & losses**: `SGDOptimizer`, `AdamOptimizer`; `MSELoss`, `CrossEntropyLoss`,
  `BCEWithLogitsLoss`, `KLDivergenceLoss`, `CalibrationLoss`
- **Generative building blocks**: `Reparameterize` (VAE), `NoiseSchedule` and
  `SinusoidalTimestepEmbedding()` (diffusion)
- **Training utilities**: `MetricsSink`/`NoOpMetricsSink` (where `train_step` logs its loss)
- **Ready-made examples**: `XorNetwork` (a tiny MLP that learns XOR), `MnistConvNet` (a
  Conv2D MNIST classifier) and `MnistIdxLoader` (reads the MNIST IDX files)

The CPU backend is always built. To build the GPU backends, configure CMake with
`-DPULSATRIX_ENABLE_CUDA=ON` or `-DPULSATRIX_ENABLE_HIP=ON`.

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

### Choosing a normalization layer

`LayerNormModule`, `RMSNormModule`, `GroupNormModule` and `BatchNormModule` all implement the
same `Module` contract (`forward()`, `backward()`, `propagate_relevance()`,
`named_parameters()`). That makes them interchangeable in a `SequentialModule`. Pick one by what it normalizes over:

- `LayerNormModule` and `RMSNormModule`: each row's features (the last dimension).
  RMSNorm skips the mean-centering.
- `GroupNormModule`: groups of channels, per example.
- `BatchNormModule`: each channel across the whole batch and spatial dimensions. It has no
  running statistics or eval mode; it always uses the current batch.

## Recipes

- [XOR training walkthrough](../recipes/deep-learning/xor_training.md)
- [RNN vs. LSTM vs. GRU on a parity task](../recipes/deep-learning/sequence_models_rnn_lstm_gru.md)
- [Residual connections and normalization layers](../recipes/deep-learning/residual_and_norm_layers.md)
