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

**[Deep Learning Modules and Layers](deep-learning/index.md)**: the `Tensor`/`Shape`/
`DeviceBackend` (CPU + CUDA + HIP/ROCm) autograd core; layers from `LinearModule`/
`Conv2DModule` through normalization, pooling, `RNNModule`/`LSTMModule`/`GRUModule`,
attention (`MultiHeadAttentionModule`, `TransformerBlock`, `MambaModule`, `RWKVModule`,
`RetNetModule`); `SGDOptimizer`/`AdamOptimizer`; losses including `CalibrationLoss`; and VAE/
GAN/Diffusion building blocks (`Reparameterize`, `NoiseSchedule`,
`SinusoidalTimestepEmbedding`). Modern architectures with LRP explicitly deferred
(`propagate_relevance` throws rather than approximates) are noted per-layer.

**[Ad-hoc Interpretability](interpretability/index.md)**: post-hoc explainers, split into
**Model-Agnostic** (`KernelSHAP`, `LIME`, `PDP`) and **Deep Learning Approaches**
(`Saliency`, `IntegratedGradients`, `GradCAM`) — distinct from the per-layer LRP
(Layer-wise Relevance Propagation) rule every relevance-bearing layer carries alongside its
forward/backward math (Arras et al. for RNN/LSTM/GRU, AttnLRP for `TransformerBlock`,
MambaLRP for `MambaModule` — every rule real, cited, and conservation-tested).

**[Reinforcement Learning](reinforcement-learning/index.md)**: gymnasium-API-shaped
`Environment`/`Agent` interfaces, `CartPoleEnv`/`ContinuousCartPoleEnv`, `ReplayBuffer`/
`RolloutBuffer`, DQN (+ Double DQN), REINFORCE, A2C, PPO, SAC — every algorithm trained
end-to-end and verified against a fixed, pre-declared performance bar on a real environment.

**[Mechanistic Interpretability](mechanistic-interpretability/index.md)**: activation
caching/snapshots, `LinearProbe`, `SparseAutoencoder`, `CircuitGraph`, plus the GFlowNet
implementation (`HyperGridEnv`, `TrajectoryBalanceLoss`, `DetailedBalanceLoss`,
`SubTBLoss(λ)`) — non-goal-directed training objectives motivated by Bengio's Scientist AI /
LawZero research direction.

**[Evolutionary Computation](evolutionary-computation/index.md)**: a from-scratch,
DEAP-free genetic-algorithm core (population/fitness/selection/crossover/mutation, NSGA-II),
neuroevolution (`NEATGenome`, Evolution Strategies), evolutionary hyperparameter optimization
(`CMAES`), Population Based Training (`RunPBT`), and evolutionary generative-model training
(`GeneratorPopulation`, E-GAN's `MutationObjective`s).

**[Hyperparameter Optimization](hyperparameter-optimization/index.md)**: a typed
`SearchSpace`/`Configuration`/`Trial` core, grid/random search, Bayesian optimization
(`GaussianProcessRegressor`, TPE), and bandit-based early stopping (Successive Halving,
Hyperband, ASHA) — every algorithm a from-scratch reimplementation, never a runtime
dependency on a Python HPO library.

**Bindings**: pybind11 (`bindings/pulsatrix_py.cpp`) exposing `Tensor`, core modules, and the
explainer suite to Python.

1700+ tests, green in both Debug and Release, on Windows and Linux.

## Where to go next

- **[Getting Started](getting-started.md)** — build the library and run your first example.
- **[Recipes](recipes/index.md)** — small, runnable programs demonstrating one tool at a time.
- **[API Reference](api/index.html)** — the full Doxygen-generated class/function reference.
- [Examples](https://github.com/Joshuaweg/pulsatrix/tree/master/examples) — 11 runnable demos on GitHub.
