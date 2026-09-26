# Reinforcement Learning

Pulsatrix's RL layer follows a gymnasium-shaped `Environment`/`Agent` interface built on the
[Deep Learning Modules and Layers](../deep-learning/index.md) tensor/autograd core — every
policy and critic is a plain `Module`, trained through the same `forward()`/`backward()`/
optimizer loop as any other network. Five algorithms ship end-to-end, each trained and
verified against a fixed, pre-declared performance bar on a real environment (not just a
loss-goes-down check): DQN (+ Double DQN), REINFORCE, A2C, PPO, and SAC.

## What's inside

- **Environments**: `Environment` (interface), `CartPoleEnv`, `ContinuousCartPoleEnv`
- **Agents**: `Agent` (interface), `DQNAgent`, `CategoricalPolicyAgent` (REINFORCE/A2C/PPO),
  `TanhGaussianPolicy` (SAC)
- **Buffers**: `ReplayBuffer` (off-policy, DQN), `RolloutBuffer` (on-policy, REINFORCE/A2C/PPO)
- **Value estimation**: `ComputeGAE` (Generalized Advantage Estimation), `ComputeDQNTarget`/
  `ComputeDoubleDQNTarget`
- **Target-network updates**: `SyncTargetNetwork` (hard periodic copy, DQN),
  `PolyakUpdate` (soft exponential-moving-average blend, SAC)
- **Losses**: `PolicyGradientLoss` (REINFORCE/A2C), `PPOClippedLoss`

Full API reference: [Doxygen: Reinforcement Learning](../api/group__rl.html)

## How to implement

### Generalized Advantage Estimation

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/gae.hpp"

using namespace pulsatrix;

CPUBackend backend;
// rewards/dones/values: each shape (N, 1), one row per stored rollout step
GAEResult gae = ComputeGAE(rewards, dones, values, /*bootstrap_value=*/0.0f,
                            /*gamma=*/0.99f, /*lambda=*/0.95f, &backend);
// gae.advantages: the actor's per-step weight
// gae.returns: advantages + values -- the critic's regression target
```

**What's happening:** `ComputeGAE` walks the rollout backward once, computing the TD residual
`delta[t]` at each step and accumulating it into an exponentially-decayed running advantage.
`lambda == 0` reduces to the single-step TD residual (low variance, biased); `lambda == 1`
telescopes to the full Monte-Carlo advantage (unbiased, high variance) — most training loops
use something in between (0.95 is PPO's usual default).

Recipe: [GAE and PPO's clipped objective](../recipes/rl/gae_and_ppo_clipped.md).

### Soft target-network updates

```cpp
#include "pulsatrix/polyak_update.hpp"

using namespace pulsatrix;

// online_critic, target_critic: same architecture, independently constructed
PolyakUpdate(online_critic, target_critic, /*tau=*/0.005f);
// target_critic's parameters move a small step toward online_critic's, every call
```

**What's happening:** unlike `SyncTargetNetwork`'s hard periodic copy (which freezes the
target for `k` steps, then jumps), `PolyakUpdate` blends
`destination[i] = tau * source[i] + (1 - tau) * destination[i]` element-wise, in place, on
every step — the slowly-drifting bootstrapping target DDPG/TD3/SAC depend on.

Recipe: [DQN on CartPole](../recipes/rl/dqn_cartpole.md) (uses `SyncTargetNetwork`'s hard-copy
sibling — see `PolyakUpdate`'s own doc comment for how the two differ).

## Recipes

- [DQN on CartPole](../recipes/rl/dqn_cartpole.md)
- [GAE and PPO's clipped objective](../recipes/rl/gae_and_ppo_clipped.md)
