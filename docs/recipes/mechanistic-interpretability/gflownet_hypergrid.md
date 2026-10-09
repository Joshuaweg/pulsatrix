# Recipe: GFlowNet on HyperGrid

**What you'll build:** a `GFlowNetForwardPolicy` trained with `TrajectoryBalanceLoss` on
`HyperGridEnv`. It shows that training shifts sampling toward the grid's far corner. The training
loop and hyperparameters are the same as in `tests/trajectory_balance_integration_test.cpp`.

CMake target: `gflownet_hypergrid_recipe`
(`examples/recipes/gflownet_hypergrid.cpp`).

Run it: `./build/gflownet_hypergrid_recipe` (Windows:
`build\Release\gflownet_hypergrid_recipe.exe`).

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
...
```

## What's happening

A GFlowNet learns a policy that builds objects step by step, so that each finished object is
sampled with probability *proportional to* its reward `R(x)`. Here the object is a cell on a 5x5
grid. Each trajectory starts at the origin, and each action either increments one coordinate
or stops.

`sample_gflownet_trajectory` resets the environment and samples allowed actions from the policy
until it stops. Along the way it adds up `sum_log_pf` (the forward policy's log-probabilities)
and `sum_log_pb` (the backward policy's).

All 4 corners of the grid share the same maximum reward. A fully trained GFlowNet should sample
them with roughly equal frequency. The rate printed here counts the far-corner region (both
coordinates ≥ 3). Reaching it takes 6 to 8 non-stop actions in a row. An untrained policy tends
to stop early, so it rarely gets there.

`TrajectoryBalanceLoss` trains the policy network together with `log Z`, a learned estimate of
the log of the total reward. It minimizes `Δ(τ) = log Z + Σ log P_F − log R(x) − Σ log P_B` for
each trajectory `τ`. Unlike a reward-maximizing objective, this pushes the policy to sample in
proportion to `R(x)` rather than hunting for the single best cell. The far-corner rate roughly
doubles after training. The integration test asserts the same shift with stricter thresholds.

See also: [Mechanistic Interpretability](../../mechanistic-interpretability/gflownets.md).
