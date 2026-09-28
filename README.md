<p align="center">
  <img src="docs/assets/pulsatrix_lockup.png" alt="Pulsatrix" width="500">
</p>

<p align="center">
  <a href="https://github.com/Joshuaweg/pulsatrix/actions/workflows/ci.yml"><img src="https://github.com/Joshuaweg/pulsatrix/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://joshuaweg.github.io/pulsatrix/"><img src="https://img.shields.io/badge/docs-mkdocs--material-306E22" alt="Docs"></a>
</p>

ExAI-first C++ deep learning library — explainability as a first-class property of the computation graph, not a post-hoc wrapper. Every relevance-bearing layer ships a real, cited, conservation-tested Layer-wise Relevance Propagation (LRP) rule alongside its forward/backward math — never a placeholder or a post-hoc explainer bolted on afterward.

## What's here

**Core**: `Tensor`/`Shape` (RAII), `DeviceBackend` (CPU + CUDA), `ComputationGraph`/`Autograd`, `Module` (NVI forward, pure-virtual LRP contract), `SGDOptimizer`/`AdamOptimizer`.

**Layers**: `LinearModule`, `Conv2DModule`, `ReluModule`, `FlattenModule`, `SequentialModule`; normalization (`LayerNorm`/`RMSNorm`/`GroupNorm`/`BatchNorm`); pooling (`MaxPool2D`/`AvgPool2D`); `DropoutModule`, `EmbeddingModule`, `ResidualModule`.

**Sequence & attention**: `RNNModule`/`LSTMModule`/`GRUModule` (Arras et al. LRP), `SoftmaxModule`, `RoPEModule`, `MultiHeadAttentionModule`, `SwiGLUModule`, `TransformerBlock` (AttnLRP, validated against an independent reference implementation), `MambaModule` (S6 selective scan, MambaLRP), `RetNetModule` (AttnLRP-style rule over its `Q@K^T`-shaped retention scores), `RWKVModule` (MambaLRP-style detach-gate rule adapted to its WKV quotient).

**Explainers**: `Saliency`, `IntegratedGradients`, `GradCAM`, `LIME`, `KernelSHAP`, `PDP`.

**Generative building blocks with LRP explicitly deferred** (real forward/backward, `propagate_relevance` throws rather than approximates): VAE (`Reparameterize`, `KLDivergenceLoss`); GAN (`BCEWithLogitsLoss`); Diffusion/DDPM (`NoiseSchedule`, `SinusoidalTimestepEmbedding`).

**Data pipeline**: `Dataset`/`IterableDataset`/`DataLoader`/`Transform`/`Compose`/`CollateFn` core, needing zero interface changes across every modality below; `CsvDataset` (tabular); `ImageDecoder`/`ImageFolderDataset`/image transforms (stb_image-backed); `Tokenizer`/`Vocabulary`/`TextDataset`/`PadCollate` (text); `WavReader`/`AudioFolderDataset`/`ResampleTransform`/`AudioPadCollate` (audio); `VideoFrameDirectoryDataset`/`UniformFrameSampleTransform` (video, reduced-scope pre-extracted-frames stub); `DatasetValidator` (descriptive statistics, missingness/outlier detection).

**Reinforcement learning** (`Environment`/`Agent` interfaces, gymnasium-API-shaped): `CartPoleEnv`/`ContinuousCartPoleEnv`, `ReplayBuffer`/`RolloutBuffer`, DQN (+ Double DQN), REINFORCE, A2C, PPO (GAE + clipped surrogate objective), SAC (twin critics, reparameterized tanh-squashed policy, entropy regularization) — every algorithm trained end-to-end and verified against a fixed, pre-declared performance bar on a real environment, not just unit-tested in isolation.

**Neuro-symbolic reasoning** (see [docs](https://joshuaweg.github.io/pulsatrix/neuro-symbolic/)): a differentiable fuzzy-logic core (`ConjunctionModule`/`DisjunctionModule`/`NegationModule`/`AggregatorModule`, Logic Tensor Networks-shaped) trainable via ordinary gradient descent; a from-scratch Datalog engine (`naive_evaluate`/`semi_naive_evaluate`, a real-valued/weighted generalization, and a hand-derived LRP rule for the derivation circuit); and `NeuralPredicateDatalogBridge`, wiring a real neural predicate's output into a Datalog derivation with relevance tracing end-to-end.

**Evolutionary computation** (see [docs](https://joshuaweg.github.io/pulsatrix/evolutionary-computation/)): a from-scratch, DEAP-free genetic-algorithm core (population/fitness/selection/crossover/mutation, NSGA-II); neuroevolution (`NEATGenome` + structural mutation + speciation, and Evolution Strategies, zero RL dependency); evolutionary hyperparameter optimization (`CMAES`); Population Based Training (`RunPBT`); and evolutionary generative-model training (`GeneratorPopulation`, E-GAN's Minimax/Heuristic/LeastSquares mutation objectives).

**Hyperparameter optimization** (see [docs](https://joshuaweg.github.io/pulsatrix/hyperparameter-optimization/)): a typed `SearchSpace`/`Configuration`/`Trial` core; grid/random search; Bayesian optimization (`GaussianProcessRegressor` + EI/PI/UCB acquisition functions, and TPE); bandit-based early stopping (Successive Halving, Hyperband, ASHA via a caller-owned `ResumableTrial` abstraction) — every algorithm a from-scratch reimplementation, never a runtime dependency on a Python HPO library.

**Bindings**: pybind11 (`bindings/pulsatrix_py.cpp`) exposing `Tensor`, core modules, and the explainer suite to Python.

**Examples** (`examples/`): see [`examples/README.md`](examples/README.md) for what each one demonstrates and how to run it.

1700+ tests, green in both Debug and Release, on Windows (MSVC) and Linux (GCC) — see the
[CI workflow](.github/workflows/ci.yml).

## Status / limitations

- CI covers Windows (MSVC, Visual Studio generator auto-detected) and Linux (GCC, Unix
  Makefiles) on every push to `master` and every pull request.
- There's no `install()`/export step yet — the supported way to consume this library today is building it as part of your own CMake tree (e.g. `add_subdirectory`), not `find_package(pulsatrix)` against a system-installed copy.

## Getting started

```
git clone https://github.com/Joshuaweg/cpp_exai_library.git pulsatrix
cd pulsatrix
```

A minimal end-to-end example — train a tiny network on XOR and print predictions
(the full, runnable version is [`examples/xor_demo.cpp`](examples/xor_demo.cpp)):

```cpp
#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/xor_training_example.hpp"

using namespace pulsatrix;

CPUBackend backend;
XorNetwork net(&backend);              // Linear(2,4) -> ReLU -> Linear(4,1)
AdamOptimizer optimizer(0.01f, &backend);

Tensor input(Shape({1, 2}), &backend, {1.0f, 0.0f});
Tensor pred = net.forward(input);      // forward pass
// net.train_step(input, target, optimizer, sink, step) runs forward + backward + update
```

Build it and run it yourself:

```
cmake --build build --target xor_demo --config Release
build/Release/xor_demo.exe
```

See [`examples/README.md`](examples/README.md) for the full list of 16 runnable
demos (layers, sequence models, transformers, explainers, five RL algorithms, and four opt-in
visualization demos).

## Build

- [CMake](https://cmake.org/download/) 3.20+ and a C++17 compiler (MSVC 2022, GCC 11+, or
  Clang 14+). Network access is needed at configure time (GoogleTest is fetched automatically).

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64   # Windows
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

See **[Getting Started](https://joshuaweg.github.io/pulsatrix/getting-started/)** for the full
walkthrough — Linux/macOS commands, the build options table (`PULSATRIX_ENABLE_CUDA`/
`_HIP`/`_PYTHON`, etc.), and a build-troubleshooting FAQ.

### HIP/ROCm backend (`PULSATRIX_ENABLE_HIP=ON`)

Needs an AMD GPU and a ROCm toolchain. Use the pinned container — see the header of
`docker/Dockerfile.rocm` for why the ROCm version is pinned and why installing ROCm from a
distro package manager is not an equivalent substitute:

```
scripts/rocm-build.sh 'cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Debug -DPULSATRIX_ENABLE_HIP=ON'
scripts/rocm-build.sh 'cmake --build build-hip -j"$(nproc)"'
scripts/rocm-build.sh './build-hip/tests/pulsatrix_tests'
```

`ROCM_PATH` may be set if ROCm is not installed at `/opt/rocm`. Verified directly on real
gfx1151 hardware — the HIP tests run against the actual device, there is no mocked path.

### Troubleshooting

- **Configure fails trying to fetch GoogleTest** — `PULSATRIX_BUILD_TESTS` is `ON` by default and clones GoogleTest from GitHub via `FetchContent` at configure time. Check network access / GitHub reachability, or configure with `-DPULSATRIX_BUILD_TESTS=OFF` if you just want to build the library and examples.
- **`PULSATRIX_ENABLE_PYTHON=ON` doesn't pick up your Python install** — make sure you're passing `-DPYTHON_EXECUTABLE=<path>`, not `-DPython3_EXECUTABLE=<path>`. See the build options table above.
- **`mnist_training_demo`, `mnist_loader_test`, or `mnist_classifier_example_test` fail or find no data** — these need real MNIST files that aren't checked into the repo. Run `py -3.11 tools/fetch_mnist.py` once (requires `torchvision` installed in that Python environment) to populate `data/MNIST/raw/`, then rebuild/rerun. This is a one-time, offline data-acquisition step — nothing at C++ build or test time depends on Python afterward.
- **CUDA build can't find the toolkit** — `PULSATRIX_ENABLE_CUDA=ON` requires a working CUDA Toolkit install discoverable by CMake's `find_package(CUDAToolkit)`; install the toolkit matching your driver version first.
- **HIP build can't find `hip`/`hipblas` packages** — `PULSATRIX_ENABLE_HIP=ON` requires a ROCm install discoverable via `ROCM_PATH` (or the conventional `/opt/rocm`); see the "HIP/ROCm backend" section above.

## Documentation

API documentation is generated on demand via Doxygen (not committed to the repo). If you have [Doxygen](https://www.doxygen.nl/) installed:

```
cmake --build build --target docs
```

Generated docs land in `build/docs/`.
