# Deep Learning Modules and Layers

Pulsatrix's core is a small, explicit autograd system — a `Tensor` owning a buffer via a
`DeviceBackend*`, a `ComputationGraph` recording `Node`s as ops run, and `Module` subclasses
that wire those ops into layers. There's no hidden global state and no Python-style dynamic
typing to work around: every op, every layer, and every backend (CPU, CUDA, HIP/ROCm) is a
concrete, inspectable C++ type. This section covers that core plus every layer, optimizer,
loss, and normalization module built on top of it.

Every layer that carries an LRP (Layer-wise Relevance Propagation) rule ships it alongside
its forward/backward math, not bolted on later — see
[Ad-hoc Interpretability](../interpretability/index.md) for how those rules get invoked.

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
- **Generative building blocks**: `Reparameterize` (VAE), `NoiseSchedule`/
  `SinusoidalTimestepEmbedding` (diffusion)

Full API reference: [Doxygen: Deep Learning Modules and Layers](../api/group__dl__modules.html)

## How to implement

### Building and training a small network

Every network is a plain C++ object composed of `Module` fields, trained with an explicit
forward/backward/optimizer loop — there's no autograd tape hidden behind a `.backward()`
call on a loss scalar; you call `backward()` on the graph yourself.

```cpp
#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/xor_training_example.hpp"

using namespace pulsatrix;

CPUBackend backend;
XorNetwork net(&backend);              // Linear(2,4) -> ReLU -> Linear(4,1)
AdamOptimizer optimizer(0.01f, &backend);

Tensor input(Shape({1, 2}), &backend, {1.0f, 0.0f});
Tensor pred = net.forward(input);      // forward pass only
```

**What's happening:** `XorNetwork` owns its `LinearModule`/`ReluModule` layers as plain
members (see `include/pulsatrix/xor_training_example.hpp`). `forward()` runs the ops through
`backend`, recording each one onto the network's `ComputationGraph`. A full training step
(`net.train_step(input, target, optimizer, sink, step)`) additionally computes the loss,
walks the graph backward to populate gradients, and applies one `optimizer` update per
layer's parameters.

Runnable version: [`examples/xor_demo.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/xor_demo.cpp).
See also the [recipe](../recipes/deep-learning/xor_training.md).

### Choosing a normalization/pooling layer

`LayerNormModule`, `RMSNormModule`, `GroupNormModule`, and `BatchNormModule` all implement
the same `Module` contract (`forward()`/`backward()`/`propagate_relevance()`), so they're
interchangeable in a `SequentialModule` — pick based on what you're normalizing over (last
dimension vs. channel groups vs. the batch), not on any API difference.

## Recipes

- [XOR training walkthrough](../recipes/deep-learning/xor_training.md)
- [RNN vs. LSTM vs. GRU on a parity task](../recipes/deep-learning/sequence_models_rnn_lstm_gru.md)
- [Residual connections and normalization layers](../recipes/deep-learning/residual_and_norm_layers.md)
