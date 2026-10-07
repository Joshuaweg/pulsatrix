# Pulsatrix

<p align="center">
  <img src="assets/pulsatrix_lockup.png" alt="Pulsatrix" width="500">
</p>

[![CI](https://github.com/Joshuaweg/pulsatrix/actions/workflows/ci.yml/badge.svg)](https://github.com/Joshuaweg/pulsatrix/actions/workflows/ci.yml)

**Pulsatrix is a C++17 deep learning library built around explainability.** Every layer
implements Layer-wise Relevance Propagation (LRP) next to its forward and backward pass, so any
network you build can explain its own predictions. On the tested MLP, CNN and attention
models, the LRP results match the Zennit and LXT reference libraries to float32 precision.

New here? Start with **[Getting Started](getting-started.md)** to build the library and run
your first example, then try a **[recipe](recipes/index.md)**.

## What's in the library

**[Deep Learning Modules and Layers](deep-learning/index.md)**: tensors, autograd and
optimizers; linear, convolution, normalization and pooling layers; RNN/LSTM/GRU, attention,
`TransformerBlock`, Mamba, RetNet and RWKV; and VAE, GAN and diffusion building blocks. The
training stack has AdamW with parameter groups, learning-rate schedules, gradient clipping,
token-accurate gradient accumulation and parameter freezing. Models save and load as
safetensors or as checkpoints with optimizer state, and `set_seed` makes runs reproducible.
Runs on CPU, CUDA or HIP/ROCm.

**[Data Loading, Transformation & Validation](data-pipeline/index.md)**: `Dataset` and
`DataLoader` with readers for CSV, images, text, audio and video frames, plus dataset
statistics and outlier checks. Tokenizers load from a Hugging Face `tokenizer.json` and return
offsets back into the text, and token scores can be merged into word scores.

**[Language Models](language-models/index.md)**: load Llama, SmolLM2, Qwen2/2.5, Qwen3 and Gemma 3 models
from the Hugging Face Hub, generate text, and explain each prediction per token or per word with
AttnLRP. Logits, token ids and explanations are checked against `transformers`, `tokenizers`
and LXT, and `pulsatrix_explain_text` turns a prompt into an explanation figure in one command.

**[Interpretability](interpretability/index.md)**: explain a trained model's predictions.

- [Layer-wise Relevance Propagation](interpretability/lrp.md): Epsilon, Gamma, AlphaBeta and
  ZBox rules, Zennit's composite presets, and AttnLRP/MambaLRP for attention and state-space
  layers. AttnLRP matches LXT on real language models.
- [Model-agnostic explainers](interpretability/model-agnostic.md): KernelSHAP, LIME and partial
  dependence plots.
- [Gradient-based explainers](interpretability/deep-learning-approaches.md): Saliency,
  Integrated Gradients and Grad-CAM.
- [Checking an explanation](interpretability/index.md): deletion and insertion curves (with
  ROAD), the model-parameter randomization test, sparseness and complexity, and a random-model
  baseline that shows whether an explanation depends on what the model learned.

**[Reinforcement Learning](reinforcement-learning/index.md)**: CartPole environments, replay and
rollout buffers, and DQN, REINFORCE, A2C, PPO and SAC. Each algorithm's tests train it to a
fixed score on CartPole.

**[Mechanistic Interpretability](mechanistic-interpretability/index.md)**: look inside a
network with activation caching, linear probes, sparse autoencoders and circuit graphs. This
section also covers GFlowNets.

**[Neuro-Symbolic Reasoning](neuro-symbolic/index.md)**: differentiable fuzzy logic you can
train with gradient descent, and a Datalog engine that traces relevance from a logical
conclusion back into a neural network.

**[Evolutionary Computation](evolutionary-computation/index.md)**: genetic algorithms,
NSGA-II, NEAT, Evolution Strategies, CMA-ES, Population Based Training and E-GAN.

**[Hyperparameter Optimization](hyperparameter-optimization/index.md)**: grid and random
search, Bayesian optimization (Gaussian process and TPE), and early stopping with Successive
Halving, Hyperband and ASHA.

**[Visualization](visualization/index.md)**: versioned JSON documents for explanations,
heatmaps, token relevance, circuit graphs and training logs; dependency-free SVG charts
(including token and word relevance for text) and the `pulsatrix_svg` tool; and, optionally,
native Dear ImGui + ImPlot windows with a live training dashboard and a token relevance view.

**[System Monitoring](system-monitoring.md)**: log CPU/GPU utilization, memory and
temperatures while you train.

**[GPU Profiling](gpu-profiling.md)** and **[Benchmarks](benchmarks.md)**: per-op GPU kernel
times with `scripts/profile_hip.sh`, and `pulsatrix_bench` for step time, explanation time and
LRP conservation, with an A/B comparison between two builds.

**Python bindings**: `Tensor`, the core layers, LRP and the other explainers, and seeding from
Python. See [Getting Started](getting-started.md#python-bindings).

Everything is implemented in C++ with no Python dependency at runtime, and covered by about
2,460 CPU tests (2,600 with the HIP backend) that run in CI on Windows and Linux.

## More resources

- **[Using pulsatrix in your project](getting-started.md#using-pulsatrix-in-your-own-project)**:
  `cmake --install` and `find_package(pulsatrix)`.
- **[Customization](customization/index.md)**: add your own layers, metrics sinks or device
  backends.
- **[API Reference](api/index.html)**: the Doxygen reference for every class and function.
- **[Examples](https://github.com/Joshuaweg/pulsatrix/tree/master/examples)**: demo programs
  on GitHub.
