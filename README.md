# exai_dl_library

ExAI-first C++ deep learning library — explainability as a first-class property of the computation graph, not a post-hoc wrapper. Every relevance-bearing layer ships a real, cited, conservation-tested Layer-wise Relevance Propagation (LRP) rule alongside its forward/backward math — never a placeholder or a post-hoc explainer bolted on afterward.

Governed by `cpp_engineering.aDNA` (agent persona **Bjarne**):
- Charter: `../cpp_engineering.aDNA/what/docs/charter.md`
- Project state: `../cpp_engineering.aDNA/what/projects/exai_dl_library/STATE.md`
- Campaign roadmap: `../cpp_engineering.aDNA/how/campaigns/campaign_exai_dl_library_phase*/`
- Context library: `../cpp_engineering.aDNA/what/context/`

## What's here

**Core**: `Tensor`/`Shape` (RAII), `DeviceBackend` (CPU + CUDA), `ComputationGraph`/`Autograd`, `Module` (NVI forward, pure-virtual LRP contract), `SGDOptimizer`/`AdamOptimizer`.

**Layers**: `LinearModule`, `Conv2DModule`, `ReluModule`, `FlattenModule`, `SequentialModule`; normalization (`LayerNorm`/`RMSNorm`/`GroupNorm`/`BatchNorm`); pooling (`MaxPool2D`/`AvgPool2D`); `DropoutModule`, `EmbeddingModule`, `ResidualModule`.

**Sequence & attention**: `RNNModule`/`LSTMModule`/`GRUModule` (Arras et al. LRP), `SoftmaxModule`, `RoPEModule`, `MultiHeadAttentionModule`, `SwiGLUModule`, `TransformerBlock` (AttnLRP, validated against an independent reference implementation), `MambaModule` (S6 selective scan, MambaLRP).

**Explainers**: `Saliency`, `IntegratedGradients`, `GradCAM`, `LIME`, `KernelSHAP`, `PDP`.

**Modern architectures with LRP explicitly deferred** (real forward/backward, `propagate_relevance` throws rather than approximates — see the campaign doc for the charter deviation this represents): `RWKVModule`, `RetNetModule`; VAE (`Reparameterize`, `KLDivergenceLoss`); GAN (`BCEWithLogitsLoss`); Diffusion/DDPM (`NoiseSchedule`, `SinusoidalTimestepEmbedding`).

**Reinforcement learning** (`Environment`/`Agent` interfaces, gymnasium-API-shaped): `CartPoleEnv`/`ContinuousCartPoleEnv`, `ReplayBuffer`/`RolloutBuffer`, DQN (+ Double DQN), REINFORCE, A2C, PPO (GAE + clipped surrogate objective), SAC (twin critics, reparameterized tanh-squashed policy, entropy regularization) — every algorithm trained end-to-end and verified against a fixed, pre-declared performance bar on a real environment, not just unit-tested in isolation.

**Bindings**: pybind11 (`bindings/exai_py.cpp`) exposing `Tensor`, core modules, and the explainer suite to Python.

**Examples** (`examples/`): `xor_demo`, `mnist_training_demo`, `explainer_demo`, `grad_cam_mnist_demo`, `sequence_model_demo`, `transformer_block_demo`, `dqn_cartpole_demo`, `reinforce_cartpole_demo`, `a2c_cartpole_demo`, `ppo_cartpole_demo`, `sac_continuous_cartpole_demo`.

1000+ tests, green in both Debug and Release.

## Build

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Requires: CMake 3.20+, a C++17 compiler (MSVC 19.4x verified), network access for GoogleTest via `FetchContent`.
