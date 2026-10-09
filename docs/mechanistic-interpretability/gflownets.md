# GFlowNets

A **GFlowNet** (generative flow network) builds an outcome step by step, and learns a policy
whose finished outcomes come out with probability proportional to a reward `R(x)`. Reinforcement
learning would converge on the single best outcome; a GFlowNet keeps sampling all the good ones,
each as often as its reward deserves. That makes it a tool for exploring the structure of a
reward or an environment: many good answers, not one.

GFlowNets aren't an interpretability method. They live in this section because they are used
here to explore structure, alongside the tools that explain it.

pulsatrix includes:
- **The environment:** `HyperGridEnv`, the standard test environment.
- **The policy:** `GFlowNetForwardPolicy`, a forward policy over any network.
- **Sampling:** `sample_gflownet_trajectory`.
- **The three standard losses:**
  - **Trajectory Balance**, `TrajectoryBalanceLoss`. It needs one learned `log Z`, held in a
    `LearnableScalar`.
  - **Detailed Balance**, `DetailedBalanceLoss`.
  - **Sub-trajectory Balance, SubTB(λ)**, `SubTBLoss`. You pass each sub-trajectory pair's
    λ-weight to `forward()`.

## Sampling a trajectory

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/gflownet_forward_policy.hpp"
#include "pulsatrix/gflownet_trajectory.hpp"
#include "pulsatrix/hypergrid_env.hpp"
#include "pulsatrix/linear_module.hpp"

using namespace pulsatrix;

CPUBackend backend;
HyperGridEnv env(&backend, /*ndim=*/2, /*side_length=*/5);
LinearModule policy_net(2, 3, &backend);     // 2-dim state -> 3 actions (+x, +y, stop)
GFlowNetForwardPolicy policy(&policy_net, /*action_dim=*/3, &backend);

GFlowNetTrajectory traj = sample_gflownet_trajectory(env, policy);
// traj.states / traj.actions: every decision point and the action taken there
// traj.sum_log_pf, traj.sum_log_pb: the log-probability sums Trajectory Balance needs
// traj.terminal_reward: R(x) at the final state
```

`sample_gflownet_trajectory` works in four steps:
1. It resets `env`.
2. It samples actions from the policy, skipping invalid ones.
3. It steps `env` until the episode ends, on an explicit `stop` action or at the environment's
   step cap.
4. Along the way it sums the forward and backward log-probabilities (`Σ log P_F`, `Σ log P_B`).

## Training with Trajectory Balance

Trajectory Balance needs only the trajectory's sums, and one learned scalar, `log Z`:

```cpp
#include <cmath>

#include "pulsatrix/learnable_scalar.hpp"
#include "pulsatrix/trajectory_balance_loss.hpp"

LearnableScalar log_z(0.0f);
TrajectoryBalanceLoss tb;
const float loss = tb.forward(traj.sum_log_pf, traj.sum_log_pb, std::log(traj.terminal_reward), log_z.value());
log_z.accumulate_grad(tb.grad_log_z());
// tb.grad_weight_for_log_pf() is the weight of each step's policy-gradient term.
```

The loss gives its gradients in closed form, so no autograd is involved. The recipe shows the
full loop: it re-runs the policy at each recorded state to apply the per-step gradient.

**Detailed Balance and SubTB** work on single transitions and sub-trajectories, with a learned
flow `log F(s)` for each state. The trajectory only records sums, so you compute those
per-transition terms yourself and pass them to `DetailedBalanceLoss::forward` or
`SubTBLoss::forward`.

Recipe: [GFlowNet on HyperGrid](../recipes/mechanistic-interpretability/gflownet_hypergrid.md).

## References

- Bengio et al., "Flow Network based Generative Models for Non-Iterative Diverse Candidate
  Generation", NeurIPS 2021.
- Malkin et al., "Trajectory Balance: Improved Credit Assignment in GFlowNets", NeurIPS 2022.
- Madan et al., "Learning GFlowNets from Partial Episodes for Improved Convergence and
  Stability" (SubTB), ICML 2023.
