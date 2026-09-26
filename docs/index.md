# Pulsatrix

<p align="center">
  <img src="assets/pulsatrix_lockup.png" alt="Pulsatrix" width="500">
</p>

**ExAI-first C++ deep learning library** — explainability as a first-class property of the
computation graph, not a post-hoc wrapper. Every relevance-bearing layer ships a real, cited,
conservation-tested Layer-wise Relevance Propagation (LRP) rule alongside its forward/backward
math — never a placeholder or a post-hoc explainer bolted on afterward.

[![CI](https://github.com/Joshuaweg/pulsatrix/actions/workflows/ci.yml/badge.svg)](https://github.com/Joshuaweg/pulsatrix/actions/workflows/ci.yml)

## What's here

**Core**: `Tensor`/`Shape` (RAII), `DeviceBackend` (CPU + CUDA + HIP/ROCm),
`ComputationGraph`/`Autograd`, `Module` (NVI forward, pure-virtual LRP contract),
`SGDOptimizer`/`AdamOptimizer`.

**Layers**: `LinearModule`, `Conv2DModule`, `ReluModule`, `FlattenModule`, `SequentialModule`;
normalization (`LayerNorm`/`RMSNorm`/`GroupNorm`/`BatchNorm`); pooling (`MaxPool2D`/`AvgPool2D`);
`DropoutModule`, `EmbeddingModule`, `ResidualModule`.

**Sequence & attention**: `RNNModule`/`LSTMModule`/`GRUModule` (Arras et al. LRP),
`SoftmaxModule`, `RoPEModule`, `MultiHeadAttentionModule`, `SwiGLUModule`, `TransformerBlock`
(AttnLRP, validated against an independent reference implementation), `MambaModule` (S6
selective scan, MambaLRP).

**Explainers**: `Saliency`, `IntegratedGradients`, `GradCAM`, `LIME`, `KernelSHAP`, `PDP`.

**Reinforcement learning** (`Environment`/`Agent` interfaces, gymnasium-API-shaped):
`CartPoleEnv`/`ContinuousCartPoleEnv`, `ReplayBuffer`/`RolloutBuffer`, DQN (+ Double DQN),
REINFORCE, A2C, PPO, SAC — every algorithm trained end-to-end and verified against a fixed,
pre-declared performance bar on a real environment.

**GFlowNet & calibration training objectives**: `HyperGridEnv`, `TrajectoryBalanceLoss`,
`DetailedBalanceLoss`, `SubTBLoss(λ)`, `CalibrationLoss` (Brier score) — non-goal-directed
training objectives motivated by Bengio's Scientist AI / LawZero research direction.

**Mechanistic interpretability**: activation caching/snapshots, `LinearProbe`,
`SparseAutoencoder`, activation patching, `logit_lens`, `attention_weights`, `CircuitGraph`.

**Modern architectures with LRP explicitly deferred** (real forward/backward,
`propagate_relevance` throws rather than approximates): `RWKVModule`, `RetNetModule`; VAE
(`Reparameterize`, `KLDivergenceLoss`); GAN (`BCEWithLogitsLoss`); Diffusion/DDPM
(`NoiseSchedule`, `SinusoidalTimestepEmbedding`).

**Bindings**: pybind11 (`bindings/pulsatrix_py.cpp`) exposing `Tensor`, core modules, and the
explainer suite to Python.

1250+ tests, green in both Debug and Release, on Windows and Linux.

## Where to go next

- **[Getting Started](getting-started.md)** — build the library and run your first example.
- **[API Reference](api/index.html)** — the full Doxygen-generated class/function reference.
- [Examples](https://github.com/Joshuaweg/pulsatrix/tree/master/examples) — 11 runnable demos on GitHub.
