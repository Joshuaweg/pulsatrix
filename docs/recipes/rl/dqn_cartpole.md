# Recipe: DQN on CartPole

**What you'll build:** a Double DQN agent (`Linear(4,32) -> ReLU -> Linear(32,2)`) trained
end-to-end on `CartPoleEnv`. It uses the same training loop and hyperparameters as
`examples/dqn_cartpole_demo.cpp` and `tests/dqn_cartpole_integration_test.cpp`.

CMake target: `dqn_cartpole_recipe` (`examples/recipes/dqn_cartpole.cpp`).

Run it: `./build/dqn_cartpole_recipe` (Windows: `build\Release\dqn_cartpole_recipe.exe`).

## Code

```cpp
#include "dqn_cartpole_training.hpp"  // shared training loop -- examples/dqn_cartpole_training.hpp

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
last  20 episodes, mean length: 198.80
improvement ratio: 9.42x
...
```

Exact numbers vary by compiler and standard library. This output is from GCC on Linux; MSVC
prints, for example, `200.00` and `9.48x`.

## What's happening

`RunTraining` drives the standard DQN loop:

- It picks actions epsilon-greedily from `DQNAgent` (mostly the best-known action, sometimes a
  random one to explore).
- It stores each transition in a `ReplayBuffer` and samples minibatches from it.
- It computes targets with `ComputeDoubleDQNTarget()`. Double DQN picks the next action with
  the online network but scores it with the target network, which reduces DQN's tendency to
  overestimate values.
- Every `target_sync_interval` steps, `SyncTargetNetwork()` copies the online network's weights
  into the target network. This keeps the regression target stable between syncs, instead of
  moving every step.

The agent starts out surviving only about 21 steps per episode (near-random). By the end it
gets close to the 200-step cap.

See also: [Reinforcement Learning](../../reinforcement-learning/index.md#soft-target-network-updates)
for `PolyakUpdate`, the soft alternative to `SyncTargetNetwork`'s hard periodic copy.
