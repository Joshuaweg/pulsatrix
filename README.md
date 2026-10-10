<p align="center">
  <img src="docs/assets/pulsatrix_lockup.png" alt="Pulsatrix" width="500">
</p>

<p align="center">
  <a href="https://github.com/Joshuaweg/pulsatrix/actions/workflows/ci.yml"><img src="https://github.com/Joshuaweg/pulsatrix/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://joshuaweg.github.io/pulsatrix/"><img src="https://img.shields.io/badge/docs-mkdocs--material-306E22" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
</p>

**Pulsatrix is a C++17 deep learning library built around explainability.** Every layer
implements Layer-wise Relevance Propagation (LRP) next to its forward and backward pass, so any
network you build can explain its own predictions without a separate tool.

- **Explainable by construction.** `propagate_relevance()` is a required method on every
  `Module`. A layer can't be added without saying how relevance flows through it.
- **Checked against the reference libraries.** `LRP::explain()` matches
  [Zennit](https://github.com/chr5tphr/zennit) and [LXT](https://github.com/rachtibat/LRP-eXplains-Transformers)
  to float32 precision on MLPs, CNNs and attention blocks. Zennit's heatmaps of torchvision's
  ResNet18 and VGG16 are matched too, and AttnLRP matches LXT on real language models (SmolLM2,
  Qwen2.5, Qwen3, Llama 3.2, Gemma 3).
- **Real language models.** Load a model from the Hugging Face Hub, tokenize with its own
  `tokenizer.json`, generate text, and see which words drove each prediction, all in C++. The
  logits, token ids and explanations match `transformers`, `tokenizers` and LXT.
- **More than LRP.** Saliency, Integrated Gradients, Grad-CAM, LIME, KernelSHAP and PDP are
  included, along with reinforcement learning, mechanistic interpretability, neuro-symbolic
  reasoning, evolutionary computation and hyperparameter optimization.
- **Explanations you can check.** Explanation-quality metrics (deletion and insertion with ROAD,
  the model-parameter randomization test, sparseness, complexity) and a random-model baseline
  test whether an explanation reflects what the model learned.
- **CPU, CUDA and ROCm.** One `DeviceBackend` interface, with optional Python bindings.

📖 **Documentation:** <https://joshuaweg.github.io/pulsatrix/>

## Quick start

You need [CMake](https://cmake.org/download/) 3.20+ and a C++17 compiler (MSVC 2022, GCC 11+ or
Clang 14+). The first configure downloads GoogleTest, so it needs network access.

**Linux / macOS**

```bash
git clone https://github.com/Joshuaweg/pulsatrix.git
cd pulsatrix
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # optional: run the test suite
./build/xor_demo
```

**Windows** (Developer PowerShell or any shell with CMake on the path)

```powershell
git clone https://github.com/Joshuaweg/pulsatrix.git
cd pulsatrix
cmake -S . -B build -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure   # optional
build\Release\xor_demo.exe
```

`xor_demo` trains a tiny network on XOR and prints the loss curve and its predictions. See
[Getting Started](https://joshuaweg.github.io/pulsatrix/getting-started/) for build options
(CUDA, ROCm, Python, visualization) and troubleshooting.

## Explaining a prediction

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp.hpp"
#include "pulsatrix/relu_module.hpp"

using namespace pulsatrix;

CPUBackend backend;
LinearModule fc1(2, 4, &backend);
ReluModule relu(&backend);
LinearModule fc2(4, 2, &backend);
// ... train or load weights ...

ExplainerContext ctx({&fc1, &relu, &fc2});
Tensor input(Shape({1, 2}), &backend, {1.0f, 0.5f});

// Which input features pushed class 1 up?
Attribution eps = LRP().explain(ctx, input, /*target_index=*/1, &backend);   // epsilon rule
Attribution plus = LRP::epsilon_plus().explain(ctx, input, 1, &backend);    // Zennit preset
// eps.values has the input's shape: one relevance score per input feature.
```

The same thing from Python:

```python
import pulsatrix_py as px

ctx = px.ExplainerContext([fc1, relu, fc2])
attr = px.LRP().explain(ctx, px.Tensor.from_values([1, 2], [1.0, 0.5]), [1])
```

## Explaining a language model

Download a model from the Hugging Face Hub, then ask why it predicts what it does:

```bash
hf download HuggingFaceTB/SmolLM2-135M --include "*.json" --include "*.safetensors" --local-dir models/SmolLM2-135M
./build/pulsatrix_explain_text models/SmolLM2-135M "The Eiffel Tower is located in the city of" --words -o paris.svg
```

SmolLM2 predicts " Paris", and `paris.svg` colors each word by how much it supported that
prediction: "Eiffel" gets +4.09, and no other word gets more than 1.6 either way. The
[Language Models guide](https://joshuaweg.github.io/pulsatrix/language-models/) shows the same
steps in code, along with text generation and how to check a model against Hugging Face.

The [LRP guide](https://joshuaweg.github.io/pulsatrix/interpretability/lrp/) covers the
available rules (Epsilon, Gamma, AlphaBeta/ZPlus, ZBox), per-layer composites, contrastive
explanations and how the results were validated. For a full worked example, see the
[MNIST LRP recipe](https://joshuaweg.github.io/pulsatrix/recipes/interpretability/mnist_lrp/).

## Looking inside a language model

The same tool traces an **attribution graph** through a model's transcoder features: which
interpretable features, at which layer and token, carry the prediction. The graph opens in
Neuronpedia's or circuit-tracer's viewer.

```bash
./build/pulsatrix_explain_text models/gemma-3-270m "The Eiffel Tower is located in the city of" \
    --graph eiffel.json --transcoders transcoders/transcoder_all/width_16k_l0_small
```

This takes about 30 seconds on a CPU, or 14 with `--device hip` on an AMD GPU. The
[attribution graphs guide](https://joshuaweg.github.io/pulsatrix/mechanistic-interpretability/attribution-graphs/)
shows where to download the model and Gemma Scope 2's transcoders.

Other tools find a layer's features, steer a model, compare a model with its fine-tune, and
decompose a network's weights. The
[Mechanistic Interpretability guide](https://joshuaweg.github.io/pulsatrix/mechanistic-interpretability/)
starts with which tool answers which question, and reports what each method found on real models.

## What's included

| Area | Highlights | Docs |
|---|---|---|
| Core | `Tensor`, autograd (`ComputationGraph`), `Module` with `named_parameters()` and freezing, top-k, eigen/QR/SVD, `set_seed` and deterministic mode, CPU/CUDA/HIP backends | [Deep learning](https://joshuaweg.github.io/pulsatrix/deep-learning/) |
| Training | SGD (momentum, Nesterov), Adam, AdamW with parameter groups; learning-rate schedules; gradient clipping; token-accurate gradient accumulation | [Deep learning](https://joshuaweg.github.io/pulsatrix/deep-learning/) |
| Saving and loading | Native safetensors reader and writer; checkpoints with buffers, optimizer state and format versions; a safe converter for PyTorch `.pt`/`.pth` files (pickle is never read in C++) | [Deep learning](https://joshuaweg.github.io/pulsatrix/deep-learning/) |
| Layers | Linear, Conv2D (stride, padding), BatchNorm with running statistics, eval mode and folding, other normalization, pooling (overlapping and adaptive too), dropout, embeddings, residual blocks | [Deep learning](https://joshuaweg.github.io/pulsatrix/deep-learning/) |
| Vision models | torchvision's ResNet and VGG, loaded from the published ImageNet weights; ResNet18 and VGG16 are checked against PyTorch's logits and Zennit's heatmaps, and the other sizes build from a config | [Deep learning](https://joshuaweg.github.io/pulsatrix/deep-learning/#vision-models-resnet-and-vgg) |
| Sequence models | RNN/LSTM/GRU, multi-head attention, `TransformerBlock`, encoder layers in ESM-2, BERT or Llama layout (`EncoderBlock`), Mamba, RetNet, RWKV | [Deep learning](https://joshuaweg.github.io/pulsatrix/deep-learning/) |
| Generative blocks | VAE, GAN and diffusion losses and sampling steps | [Deep learning](https://joshuaweg.github.io/pulsatrix/deep-learning/) |
| Language models | Load Llama, SmolLM2, Qwen2/2.5, Qwen3 and Gemma 3 checkpoints from the Hugging Face Hub; generate with greedy or sampled decoding and a KV cache; explain predictions per token or per word with AttnLRP | [Language models](https://joshuaweg.github.io/pulsatrix/language-models/) |
| Protein language models | Load ESM-2 checkpoints (8M to 650M) and match transformers to float precision; per-residue representations and attention maps; zero-shot variant scoring that reproduces ProteinGym; contacts from attention, checked against PDB and mmCIF structures; residue views and checked explanations; masked-LM training and fine-tuning heads; ESM's tokenizer and FASTA files | [Protein language models](https://joshuaweg.github.io/pulsatrix/protein-models/) |
| LRP | Epsilon, Gamma, AlphaBeta, ZBox; Zennit composites; AttnLRP, MambaLRP | [LRP](https://joshuaweg.github.io/pulsatrix/interpretability/lrp/) |
| Other explainers | Saliency, Integrated Gradients, Grad-CAM, LIME, KernelSHAP, PDP | [Interpretability](https://joshuaweg.github.io/pulsatrix/interpretability/) |
| Checking explanations | Deletion/insertion curves with ROAD, the model-parameter randomization test, sparseness, complexity, `NullModelBaseline` | [Interpretability](https://joshuaweg.github.io/pulsatrix/interpretability/) |
| Data pipeline | `Dataset`/`DataLoader`; CSV, image, text, audio and video-frame datasets; dataset validation; tokenizers loaded from Hugging Face `tokenizer.json` (byte-level BPE, as in SmolLM2, Qwen, Llama 3 and gpt-oss, and SentencePiece-style BPE, as in Gemma 3, Llama 2 and Mistral) with offsets back into the text; merging token scores into word scores | [Data pipeline](https://joshuaweg.github.io/pulsatrix/data-pipeline/) |
| Reinforcement learning | CartPole environments, DQN, REINFORCE, A2C, PPO, SAC | [RL](https://joshuaweg.github.io/pulsatrix/reinforcement-learning/) |
| Mechanistic interpretability | Hooks and linear probes; sparse autoencoders (TopK, BatchTopK, Matryoshka, JumpReLU), block-sparse featurizers and transcoders, with SAEBench-style metrics and random-model baselines; steering with a reliability report; model diffing with crosscoders; parameter decomposition (SPD, VPD); attribution graphs through transcoder features for Neuronpedia's viewer; GFlowNets | [Mech interp](https://joshuaweg.github.io/pulsatrix/mechanistic-interpretability/) |
| Neuro-symbolic | Differentiable fuzzy logic, a Datalog engine, LRP through Datalog derivations | [Neuro-symbolic](https://joshuaweg.github.io/pulsatrix/neuro-symbolic/) |
| Evolutionary computation | Genetic algorithms, NSGA-II, NEAT, Evolution Strategies, CMA-ES, PBT, E-GAN | [Evolutionary](https://joshuaweg.github.io/pulsatrix/evolutionary-computation/) |
| Hyperparameter optimization | Grid/random search, Gaussian-process BO, TPE, Successive Halving, Hyperband, ASHA | [HPO](https://joshuaweg.github.io/pulsatrix/hyperparameter-optimization/) |
| Visualization | Versioned JSON documents and dependency-free SVG charts (bar, waterfall, heatmap, token and word relevance for text, beeswarm) and interactive Vega-Lite HTML pages (hover, zoom, export) in the core library; opt-in Dear ImGui + ImPlot windows, a live training dashboard and a token relevance view | [Visualization](https://joshuaweg.github.io/pulsatrix/visualization/) |
| Notebooks and reports | Write results as a Jupyter notebook or an HTML page from C++ (markdown, code, figures, tensors), with no Python or kernel; rich display for the xeus-cpp notebook kernel (experimental) | [Notebooks and Reports](https://joshuaweg.github.io/pulsatrix/visualization/notebooks/) |
| System monitoring | Live CPU/GPU utilization, memory and temperature logging | [System monitoring](https://joshuaweg.github.io/pulsatrix/system-monitoring/) |
| Python bindings | `Tensor`, core layers, LRP and every explainer, `SystemMonitor`, `set_seed` and deterministic mode | [Getting Started](https://joshuaweg.github.io/pulsatrix/getting-started/#python-bindings) |
| Performance tools | `pulsatrix_bench` (step time, explanation time, LRP conservation; A/B comparison of builds), `scripts/profile_hip.sh` (per-op GPU kernel time) | [Benchmarks](https://joshuaweg.github.io/pulsatrix/benchmarks/), [GPU profiling](https://joshuaweg.github.io/pulsatrix/gpu-profiling/) |

Everything is implemented in C++ with no Python dependency at runtime. Every RL algorithm is
trained end to end in its tests and has to reach a fixed score on CartPole.

## Examples and recipes

- [`examples/`](examples/README.md): demo programs you can build and run, from `xor_demo` to
  five CartPole agents and the visualization demos.
- [Recipes](https://joshuaweg.github.io/pulsatrix/recipes/): short programs that each show one
  feature, with a walkthrough page for each.
- Command-line tools built with the library:
  - `pulsatrix_explain_text` explains a language model's prediction and draws it, or writes an
    attribution graph (from AttnLRP or transcoder features) for Neuronpedia's and
    circuit-tracer's viewers.
  - `pulsatrix_probe_esm`, `pulsatrix_steer_esm`, `pulsatrix_diff_lm` and `pulsatrix_spd_toy`
    run the mechanistic-interpretability experiments in the
    [guide](https://joshuaweg.github.io/pulsatrix/mechanistic-interpretability/): features and
    probes on protein models, steering, model diffing and parameter decomposition.
  - `pulsatrix_svg` turns a saved explanation into an SVG figure or an interactive HTML page.
  - `pulsatrix_bench` runs the [benchmark suite](https://joshuaweg.github.io/pulsatrix/benchmarks/).
  - `pulsatrix_golden`, `pulsatrix_attnlrp` and `pulsatrix_tokenizer_parity` check a model's
    logits, explanations and tokenizer against Hugging Face and LXT (see
    [Checking against Hugging Face](https://joshuaweg.github.io/pulsatrix/language-models/#checking-against-hugging-face)).

## Using pulsatrix in your project

Build and install pulsatrix, then find it from your own CMake project:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPULSATRIX_BUILD_TESTS=OFF -DPULSATRIX_BUILD_EXAMPLES=OFF
cmake --build build --parallel
cmake --install build --prefix /path/to/prefix
```

```cmake
find_package(pulsatrix 1.1 REQUIRED)
target_link_libraries(my_app PRIVATE pulsatrix::core)
```

Configure your project with `-DCMAKE_PREFIX_PATH=/path/to/prefix`. The package installs the core
library, its headers and the command-line tools:
- **General:** `pulsatrix_svg`, `pulsatrix_bench`, `pulsatrix_explain_text`.
- **Checking against Hugging Face:** `pulsatrix_golden`, `pulsatrix_attnlrp`,
  `pulsatrix_tokenizer_parity`.
- **Protein models:** `pulsatrix_proteingym`, `pulsatrix_contacts`, `pulsatrix_protein_views`,
  `pulsatrix_explain_protein`, `pulsatrix_train_esm`, `pulsatrix_finetune_esm`.
- **Mechanistic interpretability:** `pulsatrix_probe_esm`, `pulsatrix_steer_esm`,
  `pulsatrix_diff_lm`, `pulsatrix_spd_toy`.

A build with the CUDA or HIP backend also installs that backend's
headers, and `find_package` then looks for the same CUDA or ROCm libraries (ROCm through
`ROCM_PATH`, as in the build). `pulsatrix_HAS_CUDA` and `pulsatrix_HAS_HIP` say which backends the
installed build has. The visualization module and the Python bindings aren't installed.

You can also add the repository to your CMake tree. Install rules are off in that case:

```cmake
set(PULSATRIX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PULSATRIX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/pulsatrix)
target_link_libraries(my_app PRIVATE pulsatrix::core)
```

## Status

- Version 1.1: the v1.1 "Foundations and trust" milestone is complete. The API may still change
  between minor versions.
- What's planned next, and why, is in the
  [Roadmap](https://joshuaweg.github.io/pulsatrix/roadmap/).
- About 2,700 tests on the CPU, and 2,850 with the HIP backend. CI builds and tests every push
  and pull request on Windows (MSVC) and Linux (GCC), runs the Python binding tests, and
  compiles the CUDA and HIP backends.
- macOS with Clang should work but isn't tested in CI.
- The HIP/ROCm backend is tested on real AMD hardware (gfx1151) by a self-hosted CI runner on
  every push to this repository ([GPU CI](https://joshuaweg.github.io/pulsatrix/gpu-ci/)). See
  [Getting Started](https://joshuaweg.github.io/pulsatrix/getting-started/#gpu-backends) for
  the pinned ROCm container (ROCm 10.0.0; 7.2.4 is still selectable) and the host kernel it
  needs.

## API reference

The [API reference](https://joshuaweg.github.io/pulsatrix/api/index.html) is generated by
Doxygen and published with the docs site. To build it locally (requires
[Doxygen](https://www.doxygen.nl/)):

```bash
cmake --build build --target docs   # output: build/html/index.html
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). To report a security
issue, see [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
