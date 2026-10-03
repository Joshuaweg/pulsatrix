# Examples

Small programs you can build and run to see each feature work. They're for watching, not
testing: the GoogleTest suites under `tests/` are what CI checks.

For shorter programs that each focus on one feature, with a walkthrough page for each, see the
[recipes](recipes/) and their [docs](https://joshuaweg.github.io/pulsatrix/recipes/).

## Building and running

Examples are built by default (`PULSATRIX_BUILD_EXAMPLES=ON`). To build just one:

```bash
cmake --build build --target <target_name> --config Release
```

Then run it:

```bash
./build/<target_name>                 # Linux / macOS
build\Release\<target_name>.exe       # Windows (Visual Studio)
```

If you haven't configured a build yet, see
[Getting Started](https://joshuaweg.github.io/pulsatrix/getting-started/).

**MNIST examples** need the MNIST files, which aren't in the repository. Download them once
with `python3 tools/fetch_mnist.py` (requires `torchvision`), then run the example from the
repository root.

## Core and layers

| Target | What it shows |
|---|---|
| `xor_demo` | Trains `XorNetwork` (Linear(2,4) → ReLU → Linear(4,1), Adam) on XOR and prints the loss curve and final predictions. **Start here** to see `Tensor`, `Module` and the optimizer in action. |
| `sequence_model_demo` | Trains `RNNModule`, `LSTMModule` and `GRUModule` side by side on a running-parity task. The LSTM and GRU solve it; the plain RNN gets stuck. |
| `transformer_block_demo` | Runs a small `TransformerBlock` (multi-head attention, SwiGLU, residual connections) over a toy sequence and prints the attention weights and LRP relevance. |

## Explainability

| Target | What it shows |
|---|---|
| `explainer_demo` | Runs `Saliency` and `IntegratedGradients` on a small Linear → ReLU → Linear network and prints the attribution scores. |
| `grad_cam_mnist_demo` | Grad-CAM on a Conv2D → ReLU → Flatten → Linear network shaped for MNIST. The weights are untrained and the input is a synthetic blob, so it shows how Grad-CAM works, not what a trained model learned. For a trained model, see the [MNIST LRP recipe](recipes/mnist_lrp.cpp). |
| `mnist_training_demo` | Trains `MnistConvNet` on MNIST and prints test-set accuracy. Needs MNIST data. |

## Data pipeline

| Target | What it shows |
|---|---|
| `mnist_dataloader_demo` | The same training as `mnist_training_demo`, but loading data through `Dataset`/`DataLoader` (`MnistDatasetAdapter`). Needs MNIST data. |

See also the [CSV + DataLoader recipe](recipes/csv_dataloader_training.cpp), which trains a
`LinearModule` regressor from a CSV file.

## Reinforcement learning

All five train on CartPole and print how episode length improves. Each shares its training
loop with an integration test (`examples/<name>_training.hpp`), so the demo and the test can't
drift apart. They're deterministic, so repeated runs on the same machine print the same numbers.

| Target | Algorithm |
|---|---|
| `dqn_cartpole_demo` | Double DQN |
| `reinforce_cartpole_demo` | REINFORCE (vanilla policy gradient) |
| `a2c_cartpole_demo` | A2C (actor-critic); also prints the critic's learning curve |
| `ppo_cartpole_demo` | PPO (GAE + clipped objective); also prints how far the policy moves during each update |
| `sac_continuous_cartpole_demo` | SAC on continuous-action CartPole; also prints SAC diagnostics (exploration scale, entropy, twin-critic balance) |

## Neuro-symbolic reasoning

| Target | What it shows |
|---|---|
| `neuro_symbolic_toy_kb_demo` | Trains the fuzzy-logic rule `A(x) → not B(x)` with gradient descent and prints how well the rule is satisfied as training goes on. |

## System monitoring

| Target | What it shows |
|---|---|
| `system_monitor_demo` | Logs `SystemMonitor` samples to `system_monitor.jsonl` while it runs an idle phase, a CPU matrix-multiply phase and a GPU phase (if a GPU backend is built). Prints a live status line each second. Usage: `system_monitor_demo [seconds=10]`. |

## Visualization

These open GUI windows (Dear ImGui + ImPlot) and need a display. They're only built when you
configure with `-DPULSATRIX_ENABLE_VIZ=ON`. See the
[Visualization docs](https://joshuaweg.github.io/pulsatrix/visualization/) for setup.

| Target | What it shows |
|---|---|
| `explanation_dashboard_demo` | Attribution bar and waterfall charts, a saliency heatmap, a circuit graph and an explanation score card for the `explainer_demo` network. |
| `training_dashboard_demo` | Trains `XorNetwork` live and animates the loss curve in `TrainingDashboard`. |
| `dataset_preview_demo` | Dataset statistics (histograms, validation issues) and an image thumbnail grid on small synthetic datasets. |
| `live_inference_demo` | Cycles through the XOR inputs and recomputes Saliency and LRP for each one live. |
| `mnist_viz_gallery` | Every visualization widget on real MNIST digits: live training, a dataset preview, then Saliency, Integrated Gradients, Grad-CAM, four LRP rule sets, LIME and KernelSHAP on digits you pick. `--screenshot DIR` saves one PNG per page and exits. Needs MNIST data. See the [MNIST gallery](https://joshuaweg.github.io/pulsatrix/visualization/#mnist-gallery). |
