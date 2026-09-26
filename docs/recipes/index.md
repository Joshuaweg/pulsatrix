# Recipes

Small, runnable, self-contained programs — each a real CMake target under
`examples/recipes/`, not a code snippet — demonstrating one tool at a time. Every recipe pairs
a `.cpp` file with the page below it: what you'll build, the code, the expected output, and
what's happening. Build any recipe directly instead of the whole project:

```bash
cmake --build build --target <recipe_target> --config Release
```

These are deliberately smaller and more didactic than the full demos in
[`examples/`](https://github.com/Joshuaweg/pulsatrix/tree/master/examples) (see
[`examples/README.md`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/README.md)
for those) — the existing demos stay as-is and are cross-linked here where relevant.

## Deep Learning Modules and Layers

- [XOR training walkthrough](../recipes/deep-learning/xor_training.md)
- [RNN vs. LSTM vs. GRU on a parity task](../recipes/deep-learning/sequence_models_rnn_lstm_gru.md)
- [Residual connections and normalization layers](../recipes/deep-learning/residual_and_norm_layers.md)

## Ad-hoc Interpretability

- [KernelSHAP basics](../recipes/interpretability/kernel_shap_basics.md)
- [LIME basics](../recipes/interpretability/lime_basics.md)
- [Saliency and Integrated Gradients](../recipes/interpretability/saliency_and_integrated_gradients.md)
- [Grad-CAM walkthrough](../recipes/interpretability/grad_cam_walkthrough.md)

## Reinforcement Learning

- [DQN on CartPole](../recipes/rl/dqn_cartpole.md)
- [GAE and PPO's clipped objective](../recipes/rl/gae_and_ppo_clipped.md)

## Mechanistic Interpretability

- [Sparse autoencoder + linear probe](../recipes/mechanistic-interpretability/sparse_autoencoder_probe.md)
- [GFlowNet on HyperGrid](../recipes/mechanistic-interpretability/gflownet_hypergrid.md)
