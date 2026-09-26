# Mechanistic Interpretability

Where [Ad-hoc Interpretability](../interpretability/index.md) explains a model's
input-output behavior, mechanistic interpretability opens up *how* the model computes that
behavior internally: caching activations, testing whether a concept is linearly decodable
from them, and (for sparse autoencoders / circuit graphs) decomposing them into more
interpretable structure. This section also includes pulsatrix's GFlowNet implementation — a
non-goal-directed training objective motivated by Bengio's Scientist AI / LawZero research
direction, included here because sampling trajectories proportional to reward is itself a
tool for *exploring* a model/environment's structure rather than optimizing it.

## What's inside

- **Activation access**: `ActivationSnapshot` (self-contained, enumerable copy of one forward
  pass's cached activations)
- **Probing**: `LinearProbe` (trains a linear classifier to test whether a binary concept is
  linearly decodable from a layer's activations)
- **Decomposition**: `SparseAutoencoder`, `CircuitGraph`
- **GFlowNet**: `HyperGridEnv`, `GFlowNetForwardPolicy`, `sample_gflownet_trajectory`
  (`GFlowNetTrajectory`), `TrajectoryBalanceLoss`, `DetailedBalanceLoss`, `SubTBLoss(λ)`,
  `LearnableScalar` (the bare trainable `log Z` scalar Trajectory Balance needs, outside the
  `Module` hierarchy entirely)

Full API reference: [Doxygen: Mechanistic Interpretability](../api/group__mech__interp.html)

## How to implement

### Probing for a linearly-decodable concept

```cpp
#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_probe.hpp"

using namespace pulsatrix;

CPUBackend backend;
LinearProbe probe(/*activation_dim=*/64, &backend);
AdamOptimizer optimizer(0.01f, &backend);

// activation_batch: (N, 64) activations captured via ActivationSnapshot
// label_batch: (N, 1) binary concept labels, values in {0, 1}
float loss = probe.train_step(activation_batch, label_batch, optimizer);
float acc = probe.accuracy(activation_batch, label_batch);  // chance-level == concept not linearly represented
```

**What's happening:** a `LinearProbe` is just `LinearModule(activation_dim, 1)` +
`BCEWithLogitsLoss` trained on `(activation, concept label)` pairs — high post-training
accuracy means the concept is linearly decodable from those activations; chance-level
accuracy means it isn't, at least not linearly. It consumes any plain `(N, activation_dim)`
batch, so it works equally well against hand-built synthetic data (useful for a
positive/negative-control sanity check) or real `ActivationSnapshot` output.

Recipe: [Sparse autoencoder + linear probe](../recipes/mechanistic-interpretability/sparse_autoencoder_probe.md).

### Sampling a GFlowNet trajectory

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/gflownet_forward_policy.hpp"
#include "pulsatrix/gflownet_trajectory.hpp"
#include "pulsatrix/hypergrid_env.hpp"

using namespace pulsatrix;

CPUBackend backend;
HyperGridEnv env(/* grid size, dims, ... */);
GFlowNetForwardPolicy policy(/* ... */, &backend);

GFlowNetTrajectory traj = sample_gflownet_trajectory(env, policy);
// traj.states / traj.actions: every decision point and action taken
// traj.sum_log_pf, traj.sum_log_pb: trajectory-level quantities Trajectory Balance needs
// traj.terminal_reward: R(x) at the terminal state
```

**What's happening:** `sample_gflownet_trajectory` resets `env`, then repeatedly samples a
masked action from `forward_policy` and steps `env` until termination (an explicit `stop` or
the environment's own step cap), accumulating `Σ log P_F` and `Σ log P_B` along the way. The
resulting trajectory feeds directly into `TrajectoryBalanceLoss`/`DetailedBalanceLoss`/
`SubTBLoss`, which train the policy to sample trajectories proportional to `R(x)` rather than
to maximize it.

Recipe: [GFlowNet on HyperGrid](../recipes/mechanistic-interpretability/gflownet_hypergrid.md).

## Recipes

- [Sparse autoencoder + linear probe](../recipes/mechanistic-interpretability/sparse_autoencoder_probe.md)
- [GFlowNet on HyperGrid](../recipes/mechanistic-interpretability/gflownet_hypergrid.md)
