# Recipe: GAE and PPO's Clipped Objective

**What you'll build:** `ComputeGAE` over a 6-step synthetic rollout, then `PPOClippedLoss`
using those advantages — comparing an unchanged policy (every ratio exactly 1) against one
that has drifted since the rollout was collected.

CMake target: `gae_and_ppo_clipped_recipe`
(`examples/recipes/gae_and_ppo_clipped.cpp`).

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
```

## What's happening

`ComputeGAE` walks the rollout backward once, blending each step's one-step TD residual into
an exponentially-decayed running advantage (`lambda=0.95`). Step 5 is terminal
(`dones[5]=1`), so its advantage collapses to exactly the raw TD residual (`return - value =
1.0 - 0.7 = 0.3`) with no bootstrap or backward trace crossing the terminal boundary — the
`return` column is `advantages + values`, the critic's regression target.

At ratio `== 1` (the "unchanged" policy) `PPOClippedLoss` degenerates to
`PolicyGradientLoss`'s own advantage-weighted surrogate — the clip never activates. Once the
current policy's taken-action logits are pushed up (the "drifted" policy), some per-step
ratios exceed `1 + clip_epsilon`; those rows contribute a flat (zero-gradient) term instead of
their raw ratio-weighted term, which is exactly the mechanism that bounds how far a single
PPO update can move the policy from the data that collected it.

See also: [Reinforcement Learning](../../reinforcement-learning/index.md#generalized-advantage-estimation).
