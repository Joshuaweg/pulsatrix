# Recipe: Sparse Autoencoder + Linear Probe

**What you'll build:** a `SparseAutoencoder` trained to reconstruct synthetic activations. Then a
`LinearProbe` is trained on the same activations to test whether a concept built into them (the
sign of dimension 0) can be read out with a linear classifier.

CMake target: `sparse_autoencoder_probe_recipe`
(`examples/recipes/sparse_autoencoder_probe.cpp`).

Run it: `./build/sparse_autoencoder_probe_recipe` (Windows:
`build\Release\sparse_autoencoder_probe_recipe.exe`).

## Code

```cpp
SparseAutoencoder sae(/*dim=*/4, /*hidden_dim=*/12, /*l1_lambda=*/0.01f, &backend);
AdamOptimizer sae_optimizer(0.01f, &backend);
for (int epoch = 0; epoch <= 300; ++epoch) {
    float loss = sae.train_step(activations, sae_optimizer);
}

LinearProbe probe(/*activation_dim=*/4, &backend);
AdamOptimizer probe_optimizer(0.1f, &backend);
for (int epoch = 0; epoch <= 200; ++epoch) {
    probe.train_step(activations, labels, probe_optimizer);
}
float accuracy = probe.accuracy(activations, labels);
```

Full source: [`examples/recipes/sparse_autoencoder_probe.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/sparse_autoencoder_probe.cpp).

## Expected output

```
Sparse autoencoder + linear probe recipe

=== SparseAutoencoder (dim=4, hidden_dim=12, l1_lambda=0.01) ===
epoch   0 | reconstruction loss 0.468599 | mean hidden activation 0.0480
epoch 100 | reconstruction loss 0.004234 | mean hidden activation 0.2122
epoch 200 | reconstruction loss 0.002660 | mean hidden activation 0.1929
epoch 300 | reconstruction loss 0.002269 | mean hidden activation 0.1824
final reconstruction error: 0.002264

=== LinearProbe (activation_dim=4) ===
probe accuracy: 93.8% (chance level is 50%)
...
```

Exact numbers vary by compiler and standard library, because the synthetic data comes from
`std::normal_distribution`. This output is from GCC on Linux; MSVC gives, for example, a probe
accuracy of 100.0%.

## What's happening

The `SparseAutoencoder` is `LinearModule(4,12) -> ReLU -> LinearModule(12,4)`. Its hidden layer
is overcomplete: 3x wider than the input. It is trained with MSE to reconstruct its own input.
An L1 penalty on the hidden ReLU activations keeps them sparse. Each feature's decoder direction
is kept at unit length, so the penalty can't be met by shrinking the activations and growing the
decoder instead (see [Featurizers](../../mechanistic-interpretability/index.md#featurizers)).

The `LinearProbe` is a separate, simpler model: `LinearModule(4,1)` trained with
`BCEWithLogitsLoss` on the *same* activations against a binary label. The data is built so the
label follows the sign of dimension 0. Positive examples are shifted to +1 on that dimension and
negative examples to -1, with Gaussian noise (standard deviation 0.5) on every dimension. The
noise can push a few points across the boundary, so the classes aren't guaranteed to be
perfectly separable.

The probe scores well above chance (50%), so the concept is linearly decodable from these
activations. A negative control, with labels independent of every feature, would stay near
chance. `tests/linear_probe_test.cpp` runs that paired comparison. The control is what shows the
probe isn't just fitting noise.

See also: [Mechanistic Interpretability](../../mechanistic-interpretability/index.md#probing-for-a-linearly-decodable-concept).
