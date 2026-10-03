# Recipe: Residual Connections and Normalization Layers

**What you'll build:** a `ResidualModule` wrapping a small `LinearModule`, and a
`BatchNormModule` standardizing a synthetic per-channel batch. Both run forward only, so no
training is needed to see what they compute.

CMake target: `residual_and_norm_layers_recipe`
(`examples/recipes/residual_and_norm_layers.cpp`).

Run it: `./build/residual_and_norm_layers_recipe` (Windows:
`build\Release\residual_and_norm_layers_recipe.exe`).

## Code

```cpp
LinearModule inner(2, 2, &backend);
inner.set_weight({1.0f, 0.0f, 0.0f, 1.0f});  // identity
inner.set_bias({0.5f, -0.5f});

ResidualModule residual(&inner, &backend);
Tensor x(Shape({1, 2}), &backend, {1.0f, 2.0f});
Tensor y = residual.forward(x);  // y = x + inner->forward(x)

BatchNormModule bn(/*num_channels=*/2, &backend);
bn.set_gamma({1.0f, 1.0f});
bn.set_beta({0.0f, 0.0f});

// 4 batch rows, 2 channels, 1x1 spatial -- channel 0 centered at 10, channel 1 at -5.
Tensor batch(Shape({4, 2, 1, 1}), &backend, {10.0f, -5.0f, 12.0f, -3.0f, 8.0f, -7.0f, 14.0f, -1.0f});
Tensor normalized = bn.forward(batch);
```

Full source: [`examples/recipes/residual_and_norm_layers.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/residual_and_norm_layers.cpp).

## Expected output

```
Residual connections and normalization layers recipe

=== ResidualModule ===
inner = Linear(identity weight, bias=[0.5, -0.5])
x = [1.0, 2.0]
inner->forward(x) = [1.5, 1.5]  (identity + bias)
y = x + inner->forward(x) = [2.5, 3.5]

=== BatchNormModule ===
channel 0: input mean 11.00 -> normalized mean 0.0000 (expect ~0.0)
channel 1: input mean -4.00 -> normalized mean 0.0000 (expect ~0.0)
...
```

## What's happening

`ResidualModule` computes `y = x + inner->forward(x)` for any `Module`. Here `inner` is an
identity weight plus a bias, so `y = 2x + bias = [2.5, 3.5]`.

`BatchNormModule` computes a mean and standard deviation per channel, over every batch row and
spatial position of that channel. With `gamma=1` and `beta=0`, each channel's output is exactly
mean-zero, whatever its original scale. Channel 0 is centered near 10 and channel 1 near -5, and
both land at 0.

See also: [Deep Learning Modules and Layers](../../deep-learning/index.md#choosing-a-normalization-layer).
