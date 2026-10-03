# Reinforcement Learning

Use this section when you want an agent to learn by trial and error: it acts in an
environment, receives rewards, and improves its policy (the rule that picks actions).
Pulsatrix ships five algorithms: DQN (with Double DQN), REINFORCE, A2C, PPO, and SAC.

Every network here is a plain `Module` from
[Deep Learning Modules and Layers](../deep-learning/index.md). You train it with the same
`forward()`/`backward()`/optimizer loop as any other network. The environment and agent
interfaces follow the shape of Gymnasium (the standard Python RL toolkit): `reset()`, `step()`,
and `act()`.

## Which algorithm should I use?

| Algorithm | Action space | Learns from | Use it when |
|---|---|---|---|
| DQN / Double DQN | Discrete | Replayed past experience (off-policy) | You have a small, discrete action set and want sample efficiency. |
| REINFORCE | Discrete | Its own latest episodes (on-policy) | You want the simplest policy-gradient baseline. |
| A2C | Discrete | Its own latest rollouts (on-policy) | You want lower variance than REINFORCE by adding a critic. |
| PPO | Discrete | Its own latest rollouts (on-policy) | You want a stable, general-purpose default. |
| SAC | Continuous | Replayed past experience (off-policy) | Your actions are real numbers (e.g. a force or a torque). |

Each algorithm has an integration test that must reach a fixed score on CartPole
(`tests/*_cartpole_integration_test.cpp`).

## What's inside

- **Environments**: `Environment` (interface), `CartPoleEnv`, `ContinuousCartPoleEnv`
- **Agents**: `Agent` (interface), `DQNAgent`, `CategoricalPolicyAgent` (REINFORCE/A2C/PPO)
- **Policies**: `TanhGaussianPolicy`, SAC's continuous-action sampler. It is not a `Module` or
  an `Agent`: it takes three tensors and returns two.
- **Buffers**: `ReplayBuffer` (off-policy: DQN, SAC), `RolloutBuffer` (on-policy:
  REINFORCE/A2C/PPO)
- **Value estimation**: `ComputeGAE` (Generalized Advantage Estimation), `ComputeDQNTarget`/
  `ComputeDoubleDQNTarget`
- **Target-network updates**: `SyncTargetNetwork` (hard periodic copy, used by DQN),
  `PolyakUpdate` (small blend every step, used by SAC)
- **Losses**: `DQNLoss`, `PolicyGradientLoss` (REINFORCE/A2C), `PPOClippedLoss`

Full API reference: [Doxygen: Reinforcement Learning](../api/group__rl.html)

## Runnable demos

Each algorithm has a demo target, built by default (`PULSATRIX_BUILD_EXAMPLES=ON`):

| Target | Algorithm |
|---|---|
| `dqn_cartpole_demo` | Double DQN on `CartPoleEnv` |
| `reinforce_cartpole_demo` | REINFORCE on `CartPoleEnv` |
| `a2c_cartpole_demo` | A2C on `CartPoleEnv` |
| `ppo_cartpole_demo` | PPO on `CartPoleEnv` |
| `sac_continuous_cartpole_demo` | SAC on `ContinuousCartPoleEnv` |

```bash
cmake --build build --target ppo_cartpole_demo --config Release
```

## How to implement

### The environment/agent loop

```cpp
#include "pulsatrix/cartpole_env.hpp"
#include "pulsatrix/categorical_policy_agent.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"

using namespace pulsatrix;

CPUBackend backend;
CartPoleEnv env(&backend);                       // 4-dim observation, 2 discrete actions
LinearModule policy_net(4, 2, &backend);         // observation -> action logits
CategoricalPolicyAgent agent(&policy_net, /*action_dim=*/2, &backend);

Tensor obs = env.reset();                        // shape (1, 4)
float episode_return = 0.0f;
for (bool done = false; !done;) {
    Tensor action = agent.act(obs);              // samples an action from the policy
    StepResult step = env.step(action);
    episode_return += step.reward;
    obs = step.observation;
    done = step.done;
}
```

**What's happening:** `reset()` starts an episode and returns the first observation. Each
`step()` applies one action and returns a `StepResult`: the next observation, the reward, and
whether the episode ended. Training adds a buffer and a loss around this loop. The demos above
show the full version for each algorithm.

### Generalized Advantage Estimation

An *advantage* measures how much better an action turned out than the critic (a network that
predicts future reward) expected. Policy-gradient methods like A2C and PPO weight each update
by it.

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/gae.hpp"

using namespace pulsatrix;

CPUBackend backend;
// One row per rollout step, each shape (N, 1). Here N = 3 and the last step ends the episode.
Tensor rewards(Shape({3, 1}), &backend, {1.0f, 1.0f, 1.0f});
Tensor dones(Shape({3, 1}), &backend, {0.0f, 0.0f, 1.0f});
Tensor values(Shape({3, 1}), &backend, {2.5f, 1.8f, 0.9f});  // critic's V(s_t)

GAEResult gae = ComputeGAE(rewards, dones, values, /*bootstrap_value=*/0.0f,
                           /*gamma=*/0.99f, /*lambda=*/0.95f, &backend);
// gae.advantages: how much better each action was than the critic expected (actor weight)
// gae.returns:    advantages + values, the critic's regression target
```

**What's happening:** `ComputeGAE` walks the rollout backward once. At each step it computes
the one-step prediction error `delta[t] = r[t] + gamma * V[t+1] - V[t]` and adds it to a
decaying running sum. `lambda` sets the trade-off:

- `lambda == 0` uses only the one-step error: low variance, but biased by critic mistakes.
- `lambda == 1` uses the full episode return: unbiased, but high variance.

Most training loops pick something in between; 0.95 is PPO's usual default. A `done` flag
stops the sum from crossing into the next episode.

Recipe: [GAE and PPO's clipped objective](../recipes/rl/gae_and_ppo_clipped.md).

### Soft target-network updates

Off-policy methods train against a *target network*: a slowly changing copy of the critic that
keeps the learning target stable.

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/polyak_update.hpp"

using namespace pulsatrix;

CPUBackend backend;
LinearModule online_critic(4, 1, &backend);
LinearModule target_critic(4, 1, &backend);     // same architecture

PolyakUpdate(online_critic, target_critic, /*tau=*/0.005f);
// target_critic moves 0.5% of the way toward online_critic; call this after every update
```

**What's happening:** `PolyakUpdate` blends each parameter in place:
`target[i] = tau * online[i] + (1 - tau) * target[i]`. The target drifts a little on every
step. `SyncTargetNetwork` instead copies the online network outright every `k` steps, so the
target stays frozen and then jumps. SAC (like DDPG and TD3) uses the soft blend; DQN uses the
hard copy. `tau` must be in `(0, 1]`, and both networks must have matching parameter shapes.

The SAC demo (`sac_continuous_cartpole_demo`, training loop in
`examples/sac_continuous_cartpole_training.hpp`) calls `SyncTargetNetwork` once at start-up and
`PolyakUpdate` after every update. The [DQN on CartPole](../recipes/rl/dqn_cartpole.md) recipe
uses the periodic `SyncTargetNetwork` copy.

## Recipes

- [DQN on CartPole](../recipes/rl/dqn_cartpole.md)
- [GAE and PPO's clipped objective](../recipes/rl/gae_and_ppo_clipped.md)
