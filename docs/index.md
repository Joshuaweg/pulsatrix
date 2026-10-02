# Pulsatrix

<p align="center">
  <img src="assets/pulsatrix_lockup.png" alt="Pulsatrix" width="500">
</p>

**ExAI-first C++ deep learning library** — explainability as a first-class property of the
computation graph, not a post-hoc wrapper. Every relevance-bearing layer ships a real, cited
Layer-wise Relevance Propagation (LRP) rule alongside its forward/backward math — never a
placeholder or a post-hoc explainer bolted on afterward. Rules that conserve relevance by
construction are conservation-tested; the AttnLRP rules for softmax and attention do not conserve
exactly, and their tests report the measured gap rather than assert it away. LRP currently
implements the ε-rule family only.

[![CI](https://github.com/Joshuaweg/pulsatrix/actions/workflows/ci.yml/badge.svg)](https://github.com/Joshuaweg/pulsatrix/actions/workflows/ci.yml)

## What's here

**[Deep Learning Modules and Layers](deep-learning/index.md)**: the `Tensor`/`Shape`/
`DeviceBackend` (CPU + CUDA + HIP/ROCm) autograd core; layers from `LinearModule`/
`Conv2DModule` through normalization, pooling, `RNNModule`/`LSTMModule`/`GRUModule`,
attention (`MultiHeadAttentionModule`, `TransformerBlock`, `MambaModule`, `RWKVModule`,
`RetNetModule`); `SGDOptimizer`/`AdamOptimizer`; losses including `CalibrationLoss`; and VAE/
GAN/Diffusion building blocks (`Reparameterize`, `NoiseSchedule`,
`SinusoidalTimestepEmbedding`). The VAE/GAN/Diffusion pieces are losses and sampling/noise
steps rather than `Module`s, so they carry no LRP rule.

**[Ad-hoc Interpretability](interpretability/index.md)**: post-hoc explainers, split into
**Model-Agnostic** (`KernelSHAP`, `LIME`, `PDP`) and **Deep Learning Approaches**
(`Saliency`, `IntegratedGradients`, `GradCAM`) — distinct from the per-layer LRP
(Layer-wise Relevance Propagation) rule every relevance-bearing layer carries alongside its
forward/backward math (Arras et al. for RNN/LSTM/GRU, AttnLRP for `TransformerBlock`,
MambaLRP for `MambaModule` — every rule real and cited; conservation-tested where the rule
conserves by construction). Whole-model
`LRP::explain()` is checked against Zennit 1.0.0 (Epsilon, ZPlus, AlphaBeta, Gamma and the
`EpsilonPlus` / `EpsilonAlpha2Beta1` / `EpsilonGammaBox` presets on an MLP and a CNN) and against
LXT 2.1's AttnLRP rules (`MultiHeadAttentionModule`, `TransformerBlock`, RoPE / QK-Norm off) to
float32 precision, with the epsilon rule's bias in the denominator as both libraries use it — see
`tests/lrp_reference_test.cpp` and the README's "LRP validated against Zennit / LXT".

**[Reinforcement Learning](reinforcement-learning/index.md)**: gymnasium-API-shaped
`Environment`/`Agent` interfaces, `CartPoleEnv`/`ContinuousCartPoleEnv`, `ReplayBuffer`/
`RolloutBuffer`, DQN (+ Double DQN), REINFORCE, A2C, PPO, SAC — every algorithm trained
end-to-end and verified against a fixed, pre-declared performance bar on a real environment.

**[Mechanistic Interpretability](mechanistic-interpretability/index.md)**: activation
caching/snapshots, `LinearProbe`, `SparseAutoencoder`, `CircuitGraph`, plus the GFlowNet
implementation (`HyperGridEnv`, `TrajectoryBalanceLoss`, `DetailedBalanceLoss`,
`SubTBLoss(λ)`) — non-goal-directed training objectives motivated by Bengio's Scientist AI /
LawZero research direction.

**[Neuro-Symbolic Reasoning](neuro-symbolic/index.md)**: a differentiable fuzzy-logic core
(`ConjunctionModule`/`DisjunctionModule`/`NegationModule`/`AggregatorModule`, Logic Tensor
Networks-shaped) trainable via ordinary gradient descent, plus a from-scratch Datalog engine
(`naive_evaluate`, a real-valued/weighted generalization, and a hand-derived LRP rule) bridged
to a real neural predicate via `NeuralPredicateDatalogBridge` — relevance traces from a
symbolic derivation back into the network.

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

**[Visualization](visualization/index.md)**: a native C++ visualization layer (Dear ImGui +
ImPlot, opt-in via `PULSATRIX_ENABLE_VIZ`) turning `Attribution`/`CircuitGraph`/training-metric
data into charts — feature-importance bars, waterfalls, saliency heatmaps, a circuit-graph
view, and a live `TrainingDashboard` — built on a pure, always-available data-transform layer
that stays unit-tested independent of the GUI stack.

**Bindings**: pybind11 (`bindings/pulsatrix_py.cpp`) exposing `Tensor`, core modules, and the
explainer suite to Python.

1700+ tests, green in both Debug and Release, on Windows and Linux.

## Where to go next

- **[Getting Started](getting-started.md)** — build the library and run your first example.
- **[Recipes](recipes/index.md)** — small, runnable programs demonstrating one tool at a time.
- **[API Reference](api/index.html)** — the full Doxygen-generated class/function reference.
  Populated by CI when this site deploys; running `mkdocs serve` locally, generate it yourself
  first with `cmake --build build --target docs` (see [Getting Started](getting-started.md)) —
  otherwise this link is empty.
- [Examples](https://github.com/Joshuaweg/pulsatrix/tree/master/examples) — 16 runnable demos on GitHub.
