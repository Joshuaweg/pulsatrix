# Recipe: GFlowNet on HyperGrid

**What you'll build:** a `GFlowNetForwardPolicy` trained via `TrajectoryBalanceLoss` on
`HyperGridEnv`, showing training shifts sampling toward the reward's far corner — reusing the
exact training loop and hyperparameters from
`tests/trajectory_balance_integration_test.cpp`.

CMake target: `gflownet_hypergrid_recipe`
(`examples/recipes/gflownet_hypergrid.cpp`).

## Code

```cpp
HyperGridEnv env(&backend, /*ndim=*/2, /*side_length=*/5);
LinearModule policy_net(2, 3, &backend);
GFlowNetForwardPolicy policy(&policy_net, 3, &backend, /*seed=*/2);
AdamOptimizer optimizer(0.05f, &backend);
LearnableScalar log_z(0.0f);

// Per trajectory: sample, compute TrajectoryBalanceLoss, accumulate its gradient into
// policy_net and log_z (see the full source for the two-pass rollout this needs).
GFlowNetTrajectory traj = sample_gflownet_trajectory(env, policy);
TrajectoryBalanceLoss tb;
tb.forward(traj.sum_log_pf, traj.sum_log_pb, std::log(traj.terminal_reward), log_z.value());
```

Full source: [`examples/recipes/gflownet_hypergrid.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/gflownet_hypergrid.cpp).

## Expected output

```
GFlowNet HyperGrid recipe -- Trajectory Balance, 5x5 grid

untrained policy:  far-corner visitation rate over 500 trajectories: 5.2%
trained policy:    far-corner visitation rate over 500 trajectories: 10.4%
```

## What's happening

`sample_gflownet_trajectory` resets the environment, then repeatedly samples a masked action
from `forward_policy` and steps the environment until termination, accumulating
`sum_log_pf`/`sum_log_pb` along the way. All 4 corners of this 5x5 grid share the same
maximum reward, so a GFlowNet trained to convergence should sample all 4 with roughly equal
frequency — but reaching the far corner needs 8 consecutive non-stop actions in exactly the
right split, astronomically unlikely under an untrained (uniform-at-every-state) policy.
`TrajectoryBalanceLoss` trains `log Zθ` and the policy network jointly to minimize
`Δ(τ) = log Zθ + Σ log P_F − log R(x) − Σ log P_B`, which — unlike a reward-maximizing
objective — pushes the policy to sample trajectories *proportional to* `R(x)`, not to hunt for
the single best one. The measured far-corner rate roughly doubles after training, the same
qualitative shift the underlying integration test asserts more strictly.

See also: [Mechanistic Interpretability](../../mechanistic-interpretability/index.md#sampling-a-gflownet-trajectory).
