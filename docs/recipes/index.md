# Recipes

Small, runnable, self-contained programs that each demonstrate one tool. Each recipe is a CMake
target built from a `.cpp` file in `examples/recipes/`, paired with one of the pages below. Every
page covers what you'll build, the code, how to run it, the expected output, and what's
happening. (The visualization page uses an existing GUI demo instead of a recipe file.)

## Build and run a recipe

Build a single recipe instead of the whole project, then run it:

```bash
cmake --build build --target <recipe_target> --config Release
./build/<recipe_target>                # Linux/macOS (single-config generators)
build\Release\<recipe_target>.exe      # Windows (Visual Studio, multi-config)
```

Printed numbers can differ slightly across compilers and standard libraries. The expected output
on each page notes where this is noticeable.

Recipes are smaller and more didactic than the full demos in
[`examples/`](https://github.com/Joshuaweg/pulsatrix/tree/master/examples) (see
[`examples/README.md`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/README.md)).
Recipes link to the related full demo where one exists.

## Deep Learning Modules and Layers

- [XOR training walkthrough](deep-learning/xor_training.md)
- [RNN vs. LSTM vs. GRU on a parity task](deep-learning/sequence_models_rnn_lstm_gru.md)
- [Residual connections and normalization layers](deep-learning/residual_and_norm_layers.md)
- [Full fine-tuning: pretrain, save, reload, fine-tune](deep-learning/tagger_finetune.md)

## Data Loading, Transformation & Validation

- [CSV + DataLoader training](data-pipeline/csv_dataloader_training.md)

## Interpretability

- [KernelSHAP basics](interpretability/kernel_shap_basics.md)
- [LIME basics](interpretability/lime_basics.md)
- [Saliency and Integrated Gradients](interpretability/saliency_and_integrated_gradients.md)
- [Grad-CAM walkthrough](interpretability/grad_cam_walkthrough.md)
- [LRP on a trained MNIST classifier](interpretability/mnist_lrp.md)
- [LRP on ImageNet models (ResNet18, VGG16)](interpretability/imagenet_lrp.md)

## Reinforcement Learning

- [DQN on CartPole](rl/dqn_cartpole.md)
- [GAE and PPO's clipped objective](rl/gae_and_ppo_clipped.md)

## Mechanistic Interpretability

- [Sparse autoencoder + linear probe](mechanistic-interpretability/sparse_autoencoder_probe.md)
- [GFlowNet on HyperGrid](mechanistic-interpretability/gflownet_hypergrid.md)

## Neuro-Symbolic Reasoning

- [Datalog LRP bridge](neuro-symbolic/datalog_lrp_bridge.md)

## Protein Language Models

- [One protein end to end: TEM-1 β-lactamase](protein-models/protein_tem1.md)

## Visualization

- [A notebook report from C++](visualization/notebook_report.md)
- [Live training dashboard](visualization/training_dashboard.md)
