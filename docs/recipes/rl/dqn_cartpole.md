# Recipe: DQN on CartPole

**What you'll build:** a Double DQN agent (`Linear(4,32) -> ReLU -> Linear(32,2)`) trained
end-to-end on `CartPoleEnv`, reusing the exact validated training loop and hyperparameters
from `examples/dqn_cartpole_demo.cpp` and `tests/dqn_cartpole_integration_test.cpp`.

CMake target: `dqn_cartpole_recipe` (`examples/recipes/dqn_cartpole.cpp`).

## Code

```cpp
#include "dqn_cartpole_training.hpp"  // shared training loop -- see examples/README.md

using namespace pulsatrix::dqn_cartpole;

const TrainingConfig config;  // 260 episodes, batch 32, Adam(lr=0.002), gamma=0.99
const TrainingResult result = RunTraining(config, [](int64_t, int64_t, float) {});
```

Full source: [`examples/recipes/dqn_cartpole.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/dqn_cartpole.cpp).

## Expected output

```
DQN CartPole recipe -- Double DQN, Linear(4,32)->ReLU->Linear(32,2)
episodes=260  max_steps=200  Adam(lr=0.0020)  gamma=0.99

first 20 episodes, mean length: 21.10
last  20 episodes, mean length: 200.00
improvement ratio: 9.48x
```

## What's happening

`RunTraining` drives the standard DQN loop: epsilon-greedy action selection against
`DQNAgent`, transitions stored in a `ReplayBuffer`, minibatches sampled to compute
`ComputeDoubleDQNTarget()` (Double DQN's max-selection/evaluation split, which reduces DQN's
well-known overestimation bias), and `SyncTargetNetwork()` copying the online network's
weights into the target network every `target_sync_interval` steps — keeping the regression
target stable between syncs rather than chasing a target that moves every single step. The
agent starts by surviving only ~21 steps per episode (near-random) and ends reliably hitting
the 200-step cap.

See also: [Reinforcement Learning](../../reinforcement-learning/index.md#soft-target-network-updates)
for `PolyakUpdate`, the soft alternative to `SyncTargetNetwork`'s hard periodic copy.
