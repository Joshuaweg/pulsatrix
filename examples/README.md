# Examples

Every example below is a standalone, manually-run demo — none of them are part of the
test suite (`ctest`). Each one exists so you can *watch* the corresponding feature run,
in addition to reading its GoogleTest acceptance criteria under `tests/`. All are built
by default (`PULSATRIX_BUILD_EXAMPLES=ON`, the default — see the [root README](../README.md#build)).

Build any single target directly instead of the whole project:

```
cmake --build build --target <target_name> --config Release
```

Then run the resulting binary (Windows: `build/Release/<target_name>.exe`; Linux/macOS: `build/<target_name>`).

See also [`examples/recipes/`](recipes/) — smaller, more didactic programs than the demos
below, each paired with a walkthrough page under
[`docs/recipes/`](https://joshuaweg.github.io/pulsatrix/recipes/).

## Core / layers

| Target | What it demonstrates |
|---|---|
| `xor_demo` | Trains `XorNetwork` (Linear(2,4) -> ReLU -> Linear(4,1), Adam) on XOR and prints the loss curve and final predictions. The best starting point for seeing the core `Tensor`/`Module`/`Optimizer` API in action. |
| `sequence_model_demo` | Trains `RNNModule`, `LSTMModule`, and `GRUModule` side by side on a synthetic running-parity task, and prints their loss curves — shows the two gated modules solving it exactly while the vanilla RNN provably plateaus. |
| `transformer_block_demo` | Runs a small `TransformerBlock` (multi-head attention + SwiGLU + residual) forward pass over a toy sequence and prints its attention weights and LRP relevance attribution. |

## System monitoring

| Target | What it demonstrates |
|---|---|
| `system_monitor_demo` | `SystemMonitor` logging to `system_monitor.jsonl` while an idle phase, a CPU gemm phase and a GPU gemm phase (HIP/CUDA backend if built, else CPU) run, with `mark()` around each phase. Prints capabilities first, then a live one-line status per second. Usage: `system_monitor_demo [seconds=10]`. |

## Explainability

| Target | What it demonstrates |
|---|---|
| `explainer_demo` | Wires a small Linear->ReLU->Linear network through `ComputationGraph`/`Autograd` and runs `Saliency` and `IntegratedGradients` on it, printing real attribution scores. |
| `grad_cam_mnist_demo` | Runs Grad-CAM through a `Conv2DModule` -> `ReluModule` -> `FlattenModule` -> `LinearModule` network shaped like MNIST (1x28x28, 10 classes). **Note:** this is a synthetic/pipeline demo, not a real digit classifier — the network has randomly-initialized (untrained) weights and the input is a hand-drawn-shaped synthetic blob, not a real digit. It demonstrates Grad-CAM's mechanics (graph wiring, activation/gradient caching, per-channel weighting), not a meaningful "what the model actually learned" heatmap. See `mnist_training_demo` below for a real trained model. |
| `mnist_training_demo` | Trains `MnistConvNet` on **real** MNIST digit images and prints real measured test-set accuracy. **Prerequisite:** run `py -3.11 tools/fetch_mnist.py` once first (requires `torchvision` in that Python environment) to populate `data/MNIST/raw/` — this data is gitignored and not part of the repo. Nothing at C++ build/test time depends on Python; the fetch script is a one-time, offline step. |

## Data pipeline

| Target | What it demonstrates |
|---|---|
| `mnist_dataloader_demo` | Same network and hyperparameters as `mnist_training_demo`, but images/labels are pulled through `Dataset`/`DataLoader` (`MnistDatasetAdapter`) instead of iterating `MnistDataset`'s raw vectors directly — proves the data-loading pipeline end-to-end against real data. **Prerequisite:** same as `mnist_training_demo` (run `py -3.11 tools/fetch_mnist.py` once first). |

See also [`examples/recipes/csv_dataloader_training.cpp`](recipes/csv_dataloader_training.cpp) —
a smaller, self-contained recipe training a `LinearModule` regressor from a CSV file through
`CsvDataset`/`DataLoader`, paired with its own
[walkthrough page](https://joshuaweg.github.io/pulsatrix/recipes/data-pipeline/csv_dataloader_training/).

## Reinforcement learning

All five RL demos train on `CartPoleEnv` / `ContinuousCartPoleEnv`, are fully deterministic
(this codebase's LCG convention throughout — no `<random>` — so every run prints identical
numbers), and share their training loop and hyperparameters with a matching integration test
(`examples/<name>_training.hpp`, included by both the demo and its test, so they can't drift apart).

| Target | Algorithm |
|---|---|
| `dqn_cartpole_demo` | Double DQN. Prints episode-length progression and a first-20/last-20 average comparison. |
| `reinforce_cartpole_demo` | REINFORCE (vanilla policy gradient). Prints episode-length progression and a first-10%/last-10% average comparison. |
| `a2c_cartpole_demo` | A2C (actor-critic). Prints episode-length progression, the first-10%/last-10% comparison, and the critic's own learning curve. |
| `ppo_cartpole_demo` | PPO (GAE + clipped surrogate, multi-epoch updates). Prints episode-length progression, the first-10%/last-10% comparison, and the probability-ratio drift that proves the multi-epoch updates are real. |
| `sac_continuous_cartpole_demo` | SAC (twin critics, reparameterized tanh-squashed policy, entropy regularization) on the continuous-action variant of CartPole. Prints episode-length progression plus SAC-specific diagnostics (learned exploration scale, entropy log-probability, twin-critic min-selection balance, soft-target tracking distance). |

## Visualization

These are GUI demos (Dear ImGui + ImPlot windows, not stdout output) and are **not** built by
default — they require `-DPULSATRIX_ENABLE_VIZ=ON` at configure time (a display and GPU driver
are needed to run them). See the [Visualization docs](https://joshuaweg.github.io/pulsatrix/visualization/)
for the data-only/rendering split this module is built on.

| Target | What it demonstrates |
|---|---|
| `explanation_dashboard_demo` | `AttributionBarChart`, `AttributionWaterfallChart`, `SaliencyHeatmapView`, `CircuitGraphView`, and `ExplanationScoreCard`, all run against the same small Linear->ReLU->Linear network `explainer_demo` uses. |
| `training_dashboard_demo` | Trains `XorNetwork` live, a few steps per frame, logging through `ImPlotMetricsSink` and drawing `TrainingDashboard` so the loss curve visibly animates. |
| `dataset_preview_demo` | `DatasetStatisticsView` (per-field histograms + `DatasetValidator` issue count) and `ImageGridView` (texture-cached thumbnail grid) against small synthetic datasets — no external file/download dependency. |
| `live_inference_demo` | Reuses `ExplanationScoreCard` as a "live inference" view: cycles through the XOR inputs, recomputing Saliency + LRP conservation for whichever one is currently live. |
