# Recipe: GAE and PPO's Clipped Objective

**What you'll build:** `ComputeGAE` (Generalized Advantage Estimation) over a 6-step synthetic
rollout, then `PPOClippedLoss` using those advantages. It compares an unchanged policy (every
probability ratio exactly 1) with one that has drifted since the rollout was collected.

CMake target: `gae_and_ppo_clipped_recipe`
(`examples/recipes/gae_and_ppo_clipped.cpp`).

Run it: `./build/gae_and_ppo_clipped_recipe` (Windows:
`build\Release\gae_and_ppo_clipped_recipe.exe`).

## Code

```cpp
Tensor rewards(Shape({6, 1}), &backend, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f});
Tensor dones(Shape({6, 1}), &backend, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f});
Tensor values(Shape({6, 1}), &backend, {0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f});

GAEResult gae = ComputeGAE(rewards, dones, values, /*bootstrap_value=*/0.0f,
                            /*gamma=*/0.99f, /*lambda=*/0.95f, &backend);

PPOClippedLoss ppo(&backend);
float loss = ppo.forward(new_logits, actions, old_log_probs, gae.advantages, /*clip_epsilon=*/0.2f);
```

Full source: [`examples/recipes/gae_and_ppo_clipped.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/gae_and_ppo_clipped.cpp).

## Expected output

```
GAE recipe -- 6-step synthetic rollout, gamma=0.99, lambda=0.95

step    reward     done    value  advantage     return
0         0.00        0     0.20     1.4255     1.6255
1         0.00        0     0.30     1.4125     1.7125
2         0.00        0     0.40     1.3998     1.7998
3         0.00        0     0.50     1.3873     1.8873
4         1.00        0     0.60     1.3752     1.9752
5         1.00        1     0.70     0.3000     1.0000

PPO clipped surrogate, comparing the current policy to the data-collecting one:
  policy unchanged (ratio == 1 everywhere): loss = -1.216702
  policy drifted (taken action's logit +3): loss = -1.460042
...
```

## What's happening

**GAE.** A step's one-step TD residual is `reward + gamma * next_value - value`: how much
better the step went than the critic predicted. `ComputeGAE` walks the rollout backward once and
blends these residuals into an exponentially decayed running advantage (`lambda=0.95`).

Step 5 is terminal (`dones[5]=1`), so nothing is bootstrapped past it. Its advantage is just
`reward - value = 1.0 - 0.7 = 0.3`. The `return` column is `advantages + values`, the target
the critic is trained toward.

**PPO.** The surrogate loss weights each step's advantage by the ratio between the current
policy's probability of the taken action and the old policy's. When the policy is unchanged,
every ratio is 1 and the clip never activates. `PPOClippedLoss` then equals
`PolicyGradientLoss`'s advantage-weighted surrogate.

In the drifted policy, the taken action's logit is pushed up by 3. Every per-step ratio now
exceeds `1 + clip_epsilon`, so each row is clipped to `1.2 * advantage`. The loss is exactly 1.2×
the unchanged one (−1.216702 × 1.2 = −1.460042). Clipped rows contribute no gradient, which is
what limits how far one PPO update can move the policy away from the data that collected it.

See also: [Reinforcement Learning](../../reinforcement-learning/index.md#generalized-advantage-estimation).
