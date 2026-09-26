# Recipe: Sparse Autoencoder + Linear Probe

**What you'll build:** a `SparseAutoencoder` trained to reconstruct synthetic activations,
then a `LinearProbe` trained on the same activations to test whether a concept baked into
their construction (dimension 0's sign) is linearly decodable.

CMake target: `sparse_autoencoder_probe_recipe`
(`examples/recipes/sparse_autoencoder_probe.cpp`).

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
=== SparseAutoencoder (dim=4, hidden_dim=12, l1_lambda=0.01) ===
epoch   0 | reconstruction loss 0.542100 | mean hidden activation 0.1737
epoch 100 | reconstruction loss 0.004179 | mean hidden activation 0.2805
epoch 200 | reconstruction loss 0.001678 | mean hidden activation 0.1961
epoch 300 | reconstruction loss 0.001107 | mean hidden activation 0.1599
final reconstruction error: 0.001104

=== LinearProbe (activation_dim=4) ===
probe accuracy: 100.0% (chance level is 50%)
```

## What's happening

The `SparseAutoencoder` is `LinearModule(4,12) -> ReLU -> LinearModule(12,4)` (an
overcomplete, 3x-wider hidden basis), trained with MSE to reconstruct its own input while an
L1 penalty on the hidden ReLU activation pushes the mean hidden activation down over
training. The `LinearProbe` is a separate, much simpler model —
`LinearModule(4,1) + BCEWithLogitsLoss` — trained on the *same* activations against a binary
label. Because this recipe constructs the concept to be linearly separable (positive
examples have dimension 0 shifted positive, negative examples shifted negative, both with
noise), the probe reaches 100% accuracy: the concept **is** linearly decodable from these
activations. A negative control — labels independent of every feature — would instead
plateau near chance (50%); see `tests/linear_probe_test.cpp` for that paired comparison,
which is what actually validates the probe methodology rather than its ability to fit
anything.

See also: [Mechanistic Interpretability](../../mechanistic-interpretability/index.md#probing-for-a-linearly-decodable-concept).
