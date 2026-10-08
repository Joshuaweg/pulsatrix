# Roadmap

This page is the backlog for pulsatrix after v1.0. It says what we plan to build, in what order,
and why. Every item has an ID (for example `FND-1`) so issues, pull requests and chronicles can
point at it. The evidence behind each item, including its sources, how we would know it was a
mistake, and how it is known to fail, lives in [Research Notes](research-notes.md).

Nothing here is a promise. Priorities move as items land and as measurements come in.

## Where v1.0 stands

v1.0 is strong on explainability and thin on the foundations that larger models need.
**v1.1, "Foundations and trust", is complete** (2026-10-04): every item in its
[milestone](#v11-foundations-and-trust) is done.
**The PLM epic, protein language models, is complete** (2026-10-08): every item in
[PLM](#plm-protein-language-models) is done.

- **Strong.** LRP covers every layer and is checked against Zennit and LXT. Saliency, Integrated
  Gradients and Grad-CAM are checked against Captum. LIME, KernelSHAP, PDP, logit lens,
  activation patching, circuit graphs, probes and a basic sparse autoencoder are all in.
- **Missing.** There is no import from PyTorch or Hugging Face, and everything is `float32`.
  Tensors have no views or broadcasting. There is no LoRA.
- **Done since v1.0.** The whole [FND epic](#fnd-foundations): named parameters, freezing, top-k,
  eigensolver/QR/SVD, BatchNorm eval mode and folding, Conv2D stride and padding, seeding and
  deterministic mode, and device checks. Saving and loading: safetensors and native checkpoints
  with optimizer state ([IO-1, IO-2](#io-serialization-and-model-import)). The v1.1 training stack
  ([TRN-1 to TRN-6](#trn-training-and-fine-tuning)): parameter groups, AdamW, SGD momentum,
  gradient clipping, learning-rate schedules, token-accurate gradient accumulation and a full
  fine-tuning recipe. Explanation-quality metrics and the null-model baseline
  ([XAI-5, XAI-6](#xai-question-driven-explainability-framework)). Versioned JSON documents
  for every view and dependency-free SVG figures ([VIZ-1, VIZ-2](#viz-visualization-pack)).
  `cmake --install` with `find_package(pulsatrix)`, and a benchmark suite that compares builds
  and gates LRP conservation ([KS-1, KS-2](#ks-kitchen-sink)).
- **v1.2 so far.** Attention that loads current small LLMs
  ([LLM-1](#llm-running-real-language-models)): grouped-query attention, causal and padding
  masks, "rotate half" RoPE with a configurable base, QK-Norm and bias-free projections, checked
  against Hugging Face's Llama, Qwen2 and Qwen3 attention. A language-model head that shares the
  embedding table (LLM-2). Greedy and sampled generation (LLM-4) with a KV cache (LLM-5). Hugging
  Face configs, sharded checkpoints and bf16 weights load into a `CausalLM` (IO-4 to IO-6): the
  real SmolLM2-135M runs and matches transformers.
- **Protein language models** ([PLM-1 to PLM-9](#plm-protein-language-models)). ESM-2 8M to
  650M load from Hugging Face and match transformers to float precision. On top of that:
  - zero-shot variant scoring that reproduces ProteinGym's published numbers;
  - contacts checked against PDB and mmCIF structures;
  - mutation maps, sequence logos, contact maps, residue tracks and a 3D structure page;
  - AttnLRP residue explanations, matched to LXT and tested by deletion, randomization and
    agreement with DMS and conservation;
  - masked-LM training and fine-tuning heads;
  - probes and SAE features matched to Swiss-Prot annotations, through the new featurizer
    interface ([FEAT-1](#feat-featurizers-sparse-autoencoders-and-goodfire-bsf));
  - an end-to-end recipe on TEM-1 β-lactamase.
- **HIP backend.** It works on gfx1151 (Strix Halo). Since v1.0 it has a profiler, multi-block
  `dot` and `sum`, parallel BatchNorm, a caching allocator, and no per-op stream syncs
  ([HIP-1 to HIP-5](#hip-training-efficiency-on-amd-gpus)). Its container runs ROCm 10.0.0, the
  first release that lists gfx1151 officially (HIP-9). Small models are still only about 20%
  GPU-busy, because of host-side work inside each step.

## The goal that orders this list

**Explain real pretrained models.** That means you can load a Hugging Face model (a small LLM
such as SmolLM2-135M, or a vision model such as ResNet18), fine-tune it with LoRA on a Strix Halo
machine, and explain it with the tools pulsatrix already has. Items on that path come first.

## How to read this page

- **ID.** The epic prefix plus a number, for example `TRN-7`.
- **Priority.** P0 means the next release depends on it. P1 means it's important and planned.
  P2 means valuable but it can wait. P3 means maybe, or only if someone needs it.
- **Effort.** S is under a week, M is one to three weeks, L is one to two months, and XL is
  longer than that. These are rough and assume one person.
- **Depends on.** The IDs that have to land first.

## Rules that apply to every item

1. **Every new interpretability tool ships with baselines.** Run it on a randomly initialized
   copy of the model and compare it with a plain linear probe. Attribution maps, probes and
   sparse autoencoders all produce plausible-looking output on random networks, so a result that
   looks the same on a random network means nothing.
2. **Train in bf16, explain in fp32.** Once mixed precision exists, LRP, Integrated Gradients
   and the conservation checks still run in fp32. LRP's stabilized divisions and conservation
   tests don't survive bf16's 8-bit mantissa.
3. **Never unpickle in C++.** `.pt`, `.pth` and `.pkl` files are code, not data. They go through
   an optional converter that uses PyTorch's restricted loader. The C++ side reads safetensors
   only.
4. **Every importer passes a golden-logit test.** "It loaded" is not a test. Compare logits
   against Hugging Face Transformers on real weights and real text. A wrong RoPE layout loads
   fine and produces fluent-looking nonsense.
5. **Every new module implements `propagate_relevance` and has a conservation test.** This is
   what makes pulsatrix pulsatrix.
6. **Python is optional.** Agents, the MCP server, notebook output and model loading for
   safetensors checkpoints are native C++. Python appears only in the legacy-pickle converter
   and in optional notebook conveniences on top of the existing bindings.

## Where each requested area went

| Area | Epics |
|---|---|
| 1. New tools: embedding analysis, Goodfire BSF, surrogates, student-teacher | [INT](#int-embedding-and-representation-analysis), [TDA](#tda-topological-data-analysis), [FEAT](#feat-featurizers-sparse-autoencoders-and-goodfire-bsf), [KD](#kd-surrogates-and-student-teacher) |
| 2. Fine-tuning methods | [TRN](#trn-training-and-fine-tuning) |
| 3. Translating model files | [IO](#io-serialization-and-model-import) |
| 4. Visualization pack | [VIZ](#viz-visualization-pack), [NB](#nb-notebook-layer) |
| 5. LLMs and agent orchestration | [LLM](#llm-running-real-language-models), [TOK](#tok-tokenizers), [AGT](#agt-agents-native-c) |
| 6. Built-in XAI framework (4 reasons, 9 question categories) | [XAI](#xai-question-driven-explainability-framework), [CFS](#cfs-counterfactuals-and-sensitivity) |
| 7. New patterns and architectures | [ARCH](#arch-new-architectures) |
| 8. Kitchen sink | [KS](#ks-kitchen-sink), [FND](#fnd-foundations) |
| 9. HIP efficiency for training and tuning | [HIP](#hip-training-efficiency-on-amd-gpus) |
| 10. Protein language models | [PLM](#plm-protein-language-models) |

## Milestones

Each milestone lists the items it contains. The order follows the dependencies.

### v1.1 "Foundations and trust"

The plumbing everything else needs, plus the checks that keep explanations honest. **Complete**
(2026-10-04, PRs [#33](https://github.com/Joshuaweg/pulsatrix/pull/33) to
[#65](https://github.com/Joshuaweg/pulsatrix/pull/65)).

- FND-1 to FND-8 (**done**): named parameters, freezing, top-k, eigensolver, BatchNorm eval
  mode, Conv2D stride and padding, seeding, device checks
- IO-1, IO-2 (**done**): safetensors and the native checkpoint format
- TRN-1 to TRN-6 (**done**): parameter groups, AdamW, clipping, schedulers, gradient accumulation, full
  fine-tuning
- XAI-5, XAI-6 (**done**): explanation-quality metrics and the random-model baseline harness
- HIP-1, HIP-2, HIP-4, HIP-5 (**done**): profiling, parallel reductions, fewer syncs, multi-block
  `dot`/`sum`. HIP-3, the caching allocator, was pulled forward from v1.2 because HIP-4 depends on it
- VIZ-1, VIZ-2 (**done**): the JSON export format and the SVG renderer
- KS-1, KS-2 (**done**): `install()` and a benchmark suite
- HIP-9 (**done**): a ROCm 10.0 evaluation image, now the default, and a host-kernel check

### v1.2 "Real pretrained models"

Load SmolLM2-135M and ResNet18, run them, and match the reference implementations.

- IO-3 to IO-6: the pickle converter, name mapping, Hugging Face configs, bf16 upcast
- LLM-1 to LLM-7: attention upgrade, tied LM head, native tokenizers, generation, KV cache, the
  golden-logit harness, AttnLRP parity with LXT
- CFS-1 to CFS-7 (**done**, added and built 2026-10-05): ICE, ALE, local and global
  sensitivity, and counterfactuals, each with its view
- TOK-1 to TOK-4: the tokenizer interface with offsets, byte-level BPE (SmolLM2, Qwen, Llama),
  SentencePiece-style BPE (Gemma 3), and word-level aggregation of token scores
- HIP-6, HIP-7: fused kernels, bounded-memory Conv2D (HIP-3 landed early, in v1.1)
- VIZ-3, VIZ-6a (**done**): Vega-Lite HTML and the token relevance view
- NB-1, NB-2: native rich display and the `.ipynb`/HTML report writer
- AGT-1 to AGT-4: the native orchestrator core
- KS-8, KS-9: GPU CI on gfx1151 and a model zoo (ResNet18, SmolLM2)

### v1.3 "Tune and explain"

Fine-tune the models from v1.2 and explain what the fine-tuning changed.

- TRN-7 to TRN-11: LoRA, LRP through LoRA, LoReFT, IA³, model diffing
- KD-1, KD-2: distillation and teacher–student explanation agreement
- XAI-1 to XAI-4: the question-driven planner, explanation reports, counterfactuals, surrogates,
  Anchors
- INT-1 to INT-6: neighbors, CKA, probe controls, intrinsic dimension, PCA, TCAV and CRP
- TDA-1 to TDA-8, TDA-11: persistent homology and RTD
- KS-3, KS-4: uncertainty and adversarial robustness
- Llama-3.2-1B and Qwen2.5-0.5B pass the golden-logit test

### v1.4 "Features and agents"

Feature discovery on real models, and agents that can drive the library.

- FEAT-1 (**done**, for PLM-8), FEAT-2 to FEAT-5 (**done**) to FEAT-7: the featurizer interface, TopK
  and its family, metrics, transcoders, BSF, steering
- TDA-9, TDA-10, TDA-12: topology views, the manifold verifier for featurizers, layer-wise
  topology
- VIZ-4 (**done**), VIZ-6b, VIZ-6c: Neuronpedia export, feature dashboards, the embedding projector
- AGT-5, AGT-8: the native MCP server and automated feature descriptions
- NB-3: Python notebook display
- LLM-8, LLM-9: tuned lens, AtP*, Gemma 3
- KS-5: TracIn data attribution
- IO-7: `.npy`/`.npz`, for published SAE dictionaries
- ARCH-1 to ARCH-3: B-cos layers, concept bottleneck models, mixture of experts
- HIP-8: hipBLASLt and bf16 GEMM

### PLM "Proteins"

Its own epic, started 2026-10-07 after VIZ-4. **Complete** (2026-10-08, PRs
[#94](https://github.com/Joshuaweg/pulsatrix/pull/94) to
[#103](https://github.com/Joshuaweg/pulsatrix/pull/103), with FEAT-1 in
[#102](https://github.com/Joshuaweg/pulsatrix/pull/102)). The research behind it is in
[Protein language models: research and plan](protein-language-models.md).

- PLM-1, PLM-2: an encoder block, then ESM-2 matching `transformers` to float precision
- PLM-3, PLM-4: zero-shot variant scoring on ProteinGym, and contacts against real structures
- PLM-5, PLM-6: protein views (mutation map, logo, contact map, 3D structure) and checked
  residue-level explanations
- PLM-7 to PLM-9: masked-LM training and fine-tuning heads, probes and SAE features, and an
  end-to-end tutorial

### Backlog, not yet scheduled

These P2 and P3 items wait until a milestone needs them or someone asks: TRN-12, TRN-13, IO-10,
IO-11, INT-7, INT-8, TDA-13 to TDA-19, KD-3, KD-4, FEAT-8, ARCH-4 to ARCH-10, VIZ-7, HIP-10,
AGT-6, AGT-7, AGT-9, NB-4, NB-5, KS-6 and KS-7.

### v2.0

Larger changes that touch every module or backend.

- A dtype system and full bf16 training (HIP-11)
- Strided views and broadcasting (KS-10)
- Op-level autograd (KS-12)
- WMMA, HIP graphs and FlashAttention for training only (HIP-12)
- Attribution graphs and parameter decomposition (FEAT-9, FEAT-10)
- ARCH items at P1 and P2
- ONNX, GGUF and QLoRA (IO-8, IO-9, TRN-14)
- Python wheels and a vcpkg port (KS-11)

### Last

- VIZ-8: shaped text (HarfBuzz and bidi) in the ImGui widgets, so Arabic, Hebrew and Indic
  scripts read correctly in the native windows as well as in the SVG figures

## FND: Foundations

These unblock most of the other epics. **All eight are done** (2026-10-04, PRs
[#33](https://github.com/Joshuaweg/pulsatrix/pull/33) to
[#41](https://github.com/Joshuaweg/pulsatrix/pull/41)).

| ID | Item | Why | Depends on | P | Effort | Status |
|---|---|---|---|---|---|---|
| FND-1 | `named_parameters()` with hierarchical names (`blocks.3.attn.q_proj.weight`) | Saving, loading, freezing by name, LoRA targeting and Hugging Face name mapping all need names | — | P0 | M | Done, #33 |
| FND-2 | Parameter freezing: a per-parameter `requires_grad` flag, and a "skip the weight gradient, still compute the input gradient" branch in every backward | Fine-tuning and LoRA. Skipping `dW` is where the memory and compute savings come from | FND-1 | P0 | M | Done, #34 |
| FND-3 | Top-k selection, on the host first and then on the device | TopK and BSF featurizers, MoE routers, top-k sampling, nearest neighbors, landmarks | — | P0 | M | Done, #35 |
| FND-4 | Symmetric eigensolver, power iteration, and a small SVD/QR (Jacobi) | PCA, BSF stable rank, LoReFT's orthonormal projection, PiSSA, intruder-dimension checks, spectral distances | — | P0 | M | Done, #36 (see below) |
| FND-5 | BatchNorm running statistics and an eval mode, plus a canonizer that folds BatchNorm into the previous layer before LRP (`plans/lrp_issues.md` #8) | Today a sample's explanation depends on the rest of its batch | — | P0 | S | Done, #37 |
| FND-6 | Conv2D stride and padding | Needed to load VGG and ResNet, the standard LRP benchmark models | — | P0 | M | Done, #38 |
| FND-7 | A seeding and determinism API (`set_seed`, a flag that forbids nondeterministic paths) | Reproducible explanations. Most of the pieces already exist | — | P0 | S | Done, #39 (see below) |
| FND-8 | Device-consistency checks at module and loss boundaries, and fix `Tensor::Stack` tagging (`plans/gpu_review.md` #1, #2) | A CPU tensor fed to a GPU module aborts the process on gfx1151 | — | P0 | S | Done, #40 |
| FND-9 | Memory at load: allocate each parameter's gradient on the first backward instead of at construction, and have the loaders stop holding a second copy of the weights | ESM-2 650M (2.6 GB of weights) peaks at 10.2 GB while loading, which leaves little of a 32 GB host for anything else. Inference never needs the gradients | — | P1 | M | |

### How the FND work departed from the plan

- **FND-4.** The eigensolver is Householder tridiagonalization plus implicit-shift QL, not
  Jacobi. Cyclic Jacobi took 12.7 s at n = 512, and PCA on SmolLM2's 576-wide activations would
  hit that on every call; QL takes 0.77 s. The SVD is one-sided Jacobi, as planned.
- **FND-7.** Deterministic mode is on by default, unlike PyTorch. It forbids hipBLAS and cuBLAS
  atomics.

### Follow-ups the FND work surfaced

| Follow-up | Found in | Belongs with |
|---|---|---|
| Overlapping and padded `MaxPool2DModule`, and an adaptive average pool: the ResNet stem uses `MaxPool2d(3, stride 2, padding 1)` | FND-6 | KS-9 (model zoo) |
| A `named_buffers()` so checkpoints can save BatchNorm running statistics, which aren't parameters | FND-5 | IO-2 |
| `BatchNormModule` initializes gamma to 0; PyTorch uses 1 | FND-5 | IO-5 (configs) or a separate fix |
| `std::normal_distribution` and friends differ between libstdc++ and MSVC, so seeded runs match on one platform only | FND-7 | KS-1, or its own item |
| `named_parameters()` and `set_requires_grad()` in the Python bindings | FND-1, FND-2 | NB-3 |
| A device top-k for large k (radix or bitonic select); the current kernel is O(cols·k) per row | FND-3 | FEAT-2 if k grows |
| `GaussianMutationTest.ZeroSigma…` aborts in Debug: `std::normal_distribution` rejects σ = 0 under libstdc++ assertions | found while testing | a bug fix |

## IO: Serialization and model import

One format does everything: **safetensors**. It is a small JSON header followed by raw
little-endian data, it cannot run code, and nearly every Hugging Face model ships in it. It covers
pulsatrix's own checkpoints, optimizer state, LoRA adapters, imported models, published sparse
autoencoder dictionaries and a model zoo.

| ID | Item | Why | Depends on | P | Effort | Status |
|---|---|---|---|---|---|---|
| IO-1 | Native safetensors reader and writer. Check every offset against the file size, reject overlaps and holes, and use overflow-safe size arithmetic | The single file format | — | P0 | S–M | Done, [#43](https://github.com/Joshuaweg/pulsatrix/pull/43) (see below) |
| IO-2 | Native checkpoint format: safetensors with `format_version` metadata, optimizer state in a sibling file, and a migration table between versions | Save and resume training; the passing test is a bit-identical forward pass and loss curve after reload | IO-1, FND-1 | P0 | M | Done, [#44](https://github.com/Joshuaweg/pulsatrix/pull/44) |
| IO-3 | An optional converter for legacy pickle files (`.pt`, `.pth`, `.pkl`). It uses `torch.load(weights_only=True)` with torch 2.6 or newer and writes safetensors | Pickle is code, so it stays out of C++. Not needed for models that already ship safetensors | IO-1 | P1 | M | |
| IO-4 | A name-mapping manifest with transforms: transpose (pulsatrix `Linear` stores `(in, out)`, PyTorch stores `(out, in)`), RoPE layout permutation, splitting fused QKV, weight tying. Strict mode fails on unmapped or extra keys | Turns Hugging Face names and layouts into pulsatrix modules | FND-1, IO-1 | P1 | M || Done, [#80](https://github.com/Joshuaweg/pulsatrix/pull/80) (see below) |
| IO-5 | Read Hugging Face `config.json` and sharded `model.safetensors.index.json` | Every small LLM on the Hub uses these | IO-1 | P1 | S || Done, [#78](https://github.com/Joshuaweg/pulsatrix/pull/78) (see below) |
| IO-6 | Upcast bf16 and fp16 weights to fp32 on load (exact) | Most published weights are bf16 | IO-1 | P1 | S || Done, [#79](https://github.com/Joshuaweg/pulsatrix/pull/79) |
| IO-7 | `.npy` and `.npz` reading, with object and big-endian dtypes rejected | Many SAE dictionaries and Python users' arrays | — | P1 | S–M | |
| IO-8 | ONNX import, weights only | Graph import would mean pattern-matching ONNX ops back into modules, which is research | IO-1 | P3 | M | |
| IO-9 | GGUF import for F32, F16 and Q8_0, with every count and size checked for overflow | Some small models only exist as GGUF. GGUF parsers have a long CVE history | IO-1 | P3 | M | |
| IO-10 | Keras and TensorFlow weights, through the Python converter only. Never call `load_model` on an untrusted file | Small user base, repeated `safe_mode` bypasses | IO-3 | P3 | M | |
| IO-11 | joblib/sklearn pickles and TorchScript: not supported. Document how to export from them instead | TorchScript is deprecated upstream. Pickle is code | — | P3 | — | |

### How the IO work departed from the plan

- **IO-1** is stricter than the reference implementation (huggingface/safetensors 0.8) in two
  places. It rejects a tensor name that appears twice, where the reference keeps the last, so two
  tools could disagree about one file's weights. It also rejects unknown fields in a tensor entry.
  Every file the reference writes still loads. The header parser is hand-written for the format's
  JSON subset rather than a JSON library, and is fuzzed under AddressSanitizer
  (`tools/fuzz/safetensors_fuzz.cpp`).
- **IO-2** records a checksum of the model file in the optimizer file, so stale optimizer state
  from an earlier save is refused. A plain safetensors file with matching names loads as format
  version 0.
- **IO-4** comes with a model to load into: `CausalLM`, built from an `HfModelConfig`, with
  parameter names close to Hugging Face's, so the manifest is mostly renames and transposes. No
  RoPE permutation is needed, because LLM-1 runs Hugging Face's rotate-half layout natively.
  Fused tensors are split with row slices. It is checked against tiny Llama, Qwen2 and Qwen3
  models saved by transformers itself (logits within 1e-4, identical greedy generations), and the
  real SmolLM2-135M loads with logits within 4.3e-4 of transformers' and the same greedy text.
- **IO-5** fills in the values each architecture implies but its `config.json` leaves out
  (Qwen2's Q/K/V biases, Qwen3's and Gemma's QK-Norm, `head_dim`), and lists what pulsatrix can't
  run yet instead of ignoring it, so a loader can refuse a model rather than run it subtly wrong.
  It also did the memory-mapped reading the IO-1 follow-ups asked for: `SafetensorsFile::Map`, used
  for every shard. A 2.2 GB checkpoint (bge-m3) opened in 1.1 ms with a 4.8 MB peak resident size.
  A sharded index must match its shards exactly, and names only files inside the checkpoint
  directory.

### Follow-ups the IO work surfaced

| Follow-up | Belongs with |
|---|---|
| Memory-mapped reading; files are read whole for now | Done in IO-5 (`SafetensorsFile::Map`) |
| Checkpointing SGD's momentum buffers (Adam and AdamW are covered) | a small fix |
| Checkpointing Dropout's mask counter; a resumed run with active dropout draws different masks | a small fix |
| Running the safetensors writer's GPU path on hardware | HIP-9 or KS-8 |
| Gradient buffers are allocated with every parameter, so inference uses twice the memory (SmolLM2-135M peaks at 1.9 GB) | allocate on first backward |

## TRN: Training and fine-tuning

The infrastructure comes first. Most LoRA failures in the literature are configuration problems,
not method problems: a study of LoRA variants from January 2026 found that, once the learning
rate is tuned, plain LoRA matches or beats most of them.

| ID | Item | Why | Depends on | P | Effort | Status |
|---|---|---|---|---|---|---|
| TRN-1 | Optimizer parameter groups, each with its own learning rate and weight decay | No weight decay on norms and biases; LoRA+ | FND-1 | P0 | S | Done, [#46](https://github.com/Joshuaweg/pulsatrix/pull/46) |
| TRN-2 | AdamW (decoupled weight decay) and SGD with momentum and Nesterov | The default for transformer fine-tuning | TRN-1 | P0 | S | Done, [#47](https://github.com/Joshuaweg/pulsatrix/pull/47) |
| TRN-3 | Global gradient-norm clipping | Stability at LoRA's higher learning rates | — | P0 | S | Done, [#48](https://github.com/Joshuaweg/pulsatrix/pull/48) |
| TRN-4 | Learning-rate schedulers: warmup, cosine, linear, constant | Every published recipe assumes them | — | P0 | S | Done, [#49](https://github.com/Joshuaweg/pulsatrix/pull/49) |
| TRN-5 | Gradient accumulation that divides by the total token count, not the mean of micro-batch means | Averaging micro-batch means is wrong when sequence lengths vary | — | P0 | S | Done, [#50](https://github.com/Joshuaweg/pulsatrix/pull/50) |
| TRN-6 | A full fine-tuning recipe on a small pretrained model | The baseline every other method is compared against | FND-2, IO-2 | P0 | S | Done, [#51](https://github.com/Joshuaweg/pulsatrix/pull/51) (see below) |
| TRN-7 | `LoRALinear`: a frozen base weight plus a low-rank update, with merge and unmerge, adapter save and load using PEFT key names, and PEFT's defaults. Documentation follows "LoRA Without Regret": apply it to all layers, use about 10× the full fine-tuning learning rate | The main fine-tuning method | FND-2, IO-1 | P1 | M | |
| TRN-8 | LRP through LoRA that splits relevance between the base path and the adapter path. Document that the gamma and z+ rules give different results for merged and unmerged adapters | Shows what the fine-tuning changed, per input. No other library does this | TRN-7 | P1 | M | |
| TRN-9 | LoReFT (representation fine-tuning) and LRP through it. The intervention is affine, so the epsilon rule is exact | Fine-tuning that is itself an interpretable subspace | FND-4 | P1 | M | |
| TRN-10 | IA³ (learned rescaling vectors) | Tiny and trivially explainable | FND-2 | P2 | S | |
| TRN-11 | Model diffing: relevance of the fine-tuned model minus relevance of the base model, plus an intruder-dimension check on the weights | Shows what a fine-tune changed across a dataset | TRN-8, FND-4 | P2 | M | |
| TRN-12 | Activation checkpointing per block. Off in explain mode, because LRP needs the cached activations | Memory for 1B-parameter models | — | P2 | M | |
| TRN-13 | rsLoRA, LoRA+ and DoRA | Small gains that a tuned learning rate often matches | TRN-7 | P2 | S–M | |
| TRN-14 | PiSSA, VeRA, AdaLoRA, GaLore, QLoRA, prefix and prompt tuning, BitFit, adapter layers, and 2025–26 LoRA variants | Low value per the 2026 variant study, or blocked by missing dtypes | TRN-7 | P3 | — | |

### How the TRN work departed from the plan

- **TRN-3.** `ClipGradNorm` leaves the gradients unchanged when their norm is NaN or infinite, so
  the caller can skip the step. PyTorch scales them anyway, which spreads the NaN.
- **TRN-5** needed a batched token loss, which didn't exist. `TokenCrossEntropyLoss` reuses the
  policy-gradient row kernels, with a weight of 1 per real token and 0 per padding token.
- **TRN-6** fine-tunes a model it pretrains itself, because importing real pretrained models is
  v1.2. It is per-token tagging rather than next-token prediction, because attention has no causal
  mask. After the same budget, the pretrained model scores 0.985 on the new rule and the model
  trained from scratch 0.862.

### Follow-ups the TRN work surfaced

| Follow-up | Belongs with |
|---|---|
| A causal attention mask, which next-token language-model training needs | Done in LLM-1 |
| Rerun the TRN-6 recipe on SmolLM2-135M | after IO-5 and LLM-6 |
| Checkpointing an `LRScheduler` with the model (TRN-6 stores the step as metadata itself) | IO-2 follow-up |

## LLM: Running real language models

Target order: SmolLM2-135M, then Qwen2.5-0.5B, Llama-3.2-1B, Qwen3-0.6B and Gemma 3 270M. SmolLM2
needs the fewest changes. Gemma 3 has the most quirks, but Google's Gemma Scope 2 publishes sparse
autoencoders and transcoders for every layer, so it pays off for the FEAT epic.

| ID | Item | Why | Depends on | P | Effort | Status |
|---|---|---|---|---|---|---|
| LLM-1 | Attention upgrade: grouped-query attention (`num_kv_heads`), `head_dim` separate from `d_model / num_heads`, optional QKV bias, causal and padding masks, a RoPE layout flag (Hugging Face "rotate half" vs. adjacent pairs), and a position offset | Without these, no current small LLM loads | — | P0 | M | Done, [#67](https://github.com/Joshuaweg/pulsatrix/pull/67) (see below) |
| LLM-2 | A tied LM head that shares the embedding matrix | SmolLM2, Qwen3 and Gemma 3 all tie their embeddings | — | P0 | S || Done, [#68](https://github.com/Joshuaweg/pulsatrix/pull/68) (see below) |
| LLM-3 | Native tokenizers for the target models: [TOK-1 to TOK-3](#tok-tokenizers) | No Python needed to tokenize | — | P0 | M–L | Done through TOK-1 to TOK-3 |
| LLM-4 | Generation: greedy, temperature, top-k and top-p sampling, seeded, with EOS handling | Run the model, not just score it | LLM-1, FND-3 | P1 | S || Done, [#76](https://github.com/Joshuaweg/pulsatrix/pull/76) (see below) |
| LLM-5 | A preallocated KV cache | Generation without recomputing the whole prefix | LLM-1 | P1 | M || Done, [#77](https://github.com/Joshuaweg/pulsatrix/pull/77) (see below) |
| LLM-6 | Golden-logit harness: compare against Hugging Face on real text, fp32, maximum difference under 1e-3 (relative to the position's largest logit above 1) | Catches layout bugs that still "load" | IO-4 | P0 | S || Done, [#81](https://github.com/Joshuaweg/pulsatrix/pull/81) (see below) |
| LLM-7 | AttnLRP parity with LXT on SmolLM2 (per-token relevance correlation above 0.99), including the relevance split across shared key and value heads | The headline result: pulsatrix explains a real LLM and matches the reference | LLM-1, LLM-2, TOK-2, LLM-6 | P1 | M || Done, [#83](https://github.com/Joshuaweg/pulsatrix/pull/83) (see below) |
| LLM-8 | Tuned lens and AtP* (corrected attribution patching) | Better versions of the logit lens and patching that already exist | LLM-1 | P2 | S–M | |
| LLM-9 | Gemma 3 support: sliding-window attention, `(1 + w)` RMSNorm, embedding scaling | Unlocks Gemma Scope 2 dictionaries | LLM-1 | P3 | M || Done, [#91](https://github.com/Joshuaweg/pulsatrix/pull/91) (see below) |


### How the LLM work departed from the plan

- **LLM-1** is configured through one `AttentionConfig` rather than more constructor arguments;
  the old constructor still works and means what it did. It was checked against Hugging Face's
  own `LlamaAttention`, `Qwen2Attention` and `Qwen3Attention` (forward output and input gradient,
  `tools/generate_attention_reference_values.py`), with grouped and multi-query heads, biases,
  QK-Norm, and left and right padding.
  - Grouped-query attention copies each K/V head out to its query heads, as Hugging Face's
    `repeat_kv` does, and sums gradients and relevance back. A shared head's relevance is then the
    total over the query heads that read it, which LLM-7 checks against LXT.
  - Masked scores are set to the lowest float, not `-inf`, so a query with no visible key (left
    padding under a causal mask) gets uniform weights rather than NaN, as in Hugging Face. Their
    gradient and relevance are set to zero, which Hugging Face's additive mask does not do.
  - The position offset only moves RoPE. Under RoPE, scores depend on relative position only,
    so an offset is invisible until queries and keys come from different pieces of the
    sequence. The mask primitive already takes a query offset for the KV cache (LLM-5).
  - `LinearModule` gained a no-bias form, and `TransformerBlock` takes an `AttentionConfig` and
    an RMSNorm epsilon.
- **LLM-2** is a separate `TiedLMHeadModule` that reads an `EmbeddingModule`'s table, rather
  than tensor views (which don't exist yet). It owns no parameters, so the optimizer, freezing and
  checkpoints see the table once, under the embedding's name, and its gradient is the sum of the
  lookup's and the head's. Every LRP rule works on it, with the table transposed as the weight.
- **LLM-4** takes the model as a function from the token sequence to the next token's logits, so
  the KV cache (LLM-5) only has to provide a faster function. The logit processing follows
  Hugging Face's processors and their order exactly and is checked against them. Sampling draws
  from `std::mt19937_64`'s raw output, so a seed gives the same tokens on every platform for the
  same logits. Results carry each token's log-probability under the unprocessed model, which the
  explainers need. Beam search, stop strings and batched generation are not included.
- **LLM-5** caches each layer's keys and values after QK-Norm and RoPE, at the K/V head count, so
  grouped-query heads read their shared head in place instead of the repeated copies the training
  pass makes. Cached passes are inference only: backward and LRP refuse to run after one. The
  generation adapter compares each history with what it has cached and truncates to the shared
  prefix, so a new continuation of the same prompt costs only its new tokens. On CPU (Release), 128
  tokens from a 4-block, 128-wide model took 47 ms instead of 3.3 s. A padding query with no
  visible key can produce different (meaningless) output than in a full pass; real positions
  match.
- **LLM-6** stores each reference as a safetensors file holding token ids and transformers'
  float32 logits (eager attention), with the model's Hub revision and library versions as
  metadata (`tools/golden/make_golden.py`). Tokenization happens in Python, so the harness tests
  the model and not a tokenizer (TOK). `pulsatrix_golden` and `CompareToGolden` compare every logit
  at every position. CI checks the tiny models in `tests/fixtures/hf_tiny`; real models are
  checked locally through `PULSATRIX_GOLDEN_DIR`. On six texts (English, code, numbers, French and
  German, Chinese and Japanese, and emoji and symbols; up to 48 tokens), the largest differences on
  the GPU were 1.3e-4 (Llama-3.2-1B), 2.2e-4 (Qwen3-0.6B), 8.0e-4 (SmolLM2-135M) and 8.3e-4
  (Qwen2.5-0.5B), with no argmax disagreements.
  - Llama 3's `rope_scaling` (the `llama3` and `linear` types) is computed as Hugging Face
    computes it and passed to `RoPEModule` as explicit per-pair frequencies. Other types are still
    refused.
  - The 1e-3 threshold is relative to each position's largest logit when that exceeds 1, so
    |diff| <= 1e-3 * max(1, max |logit|). Qwen2.5's logits reach about 24, and fp32 accumulation
    order alone gave 1.1e-3 absolute (5e-5 relative) on CPU, with no argmax disagreements. A layout
    bug moves logits by about their own size, so it still fails by orders of magnitude.
  - The CPU backend's reference `gemm` was single-threaded and walked the weight matrix by column,
    so a 1B model took more than an hour on CPU. After #82 the CPU run takes 33 s for Llama-3.2-1B,
    and every model passes on both backends.
- **LLM-7** compares pulsatrix with LXT's own AttnLRP (`lxt.efficient`) on the Hugging Face
  models, rather than with LXT's explicit rules written out by hand, so the reference is the code
  people actually run (`tools/golden/make_attnlrp_reference.py`, `pulsatrix_attnlrp`).
  - The existing rules already matched. LLM-7 added the harness and
    `MultiHeadAttentionModule::key_relevance()` / `value_relevance()`.
  - Over the LLM-6 texts, the largest relative difference in token relevance was 4.1e-6
    (SmolLM2-135M), 8.0e-6 (Qwen2.5-0.5B), 1.2e-5 (Qwen3-0.6B) and 2.3e-5 (Llama-3.2-1B). Every
    token and K/V-head correlation was 1.000000, so the bar is well above 0.99.
  - A negative control, plain gradient × input with LXT's patches off, fails with correlations
    below 0.
  - Mutation-checked: dropping the grouped-query sum for V fails every model, and leaving biases
    out of the denominator fails Qwen2.
  - Token ids come from the Python tokenizer, as in LLM-6, so parity didn't wait for TOK-2. The
    per-word display that needs TOK-2 belongs with VIZ-6a.
  - Relevance at the output is the logit (unit gradient in LXT's gradient × input). LXT's README
    calls `logit.backward(logit)`, which multiplies every relevance by the logit again. A
    per-text scale like that leaves correlation unchanged, so the harness also reports the largest
    relative difference.
  - LXT 2.1 needs two shims under transformers 5; they are described in the generator and in
    `docs/interpretability/lrp.md`.

- **LLM-9** runs Gemma 3 (the text model, `gemma3_text`), checked on Gemma 3 270M and on a tiny
  fixture. Gemma 1 and 2, and logit softcapping, are still refused by name.
  - **The pieces:**
    - sliding-window attention, a window parameter on the attention mask on every backend,
      including the KV-cache path
    - each layer type's own RoPE base, with scaling only on full-attention layers, read from both
      the older flat fields and the newer nested `rope_parameters`
    - `query_pre_attn_scalar` as the score scale
    - a GeGLU MLP (`ElementwiseOp::GeluTanh`, written once for CPU and GPU)
    - `(1 + w)` RMSNorms, kept as an offset so the stored weight stays the checkpoint's
    - sandwich norms, and embeddings scaled by `sqrt(hidden_size)`
  - **Results on Gemma 3 270M:**
    - logits within 1.6e-4 of `transformers` on CPU (4.6e-6 relative) and 1.2e-4 on the GPU
    - greedy generation identical for 24 tokens
    - AttnLRP matching LXT's own Gemma 3 patch (largest relative difference 5.5e-5, correlation
      1.000000)
  - **The real model's window (512) is longer than the golden texts**, so the tiny fixture checks
    the window: 4 layers (sliding, sliding, full, sliding) with a 3-token window,
    `query_pre_attn_scalar` 12, RoPE base 100 on sliding layers and linear scaling on the full one,
    and norm weights drawn around 0.
  - **Mutation-checked:** removing the `(1 + w)`, the embedding scale, the window, the sandwich
    norms, the per-layer-type RoPE base or the score scale each fails the tiny fixture's logits,
    golden and AttnLRP tests.
  - **Config bugs fixed:** `attn_logit_softcapping: null` (off) was reported as unsupported, and
    `layer_types` must now name one type per layer.

### Follow-ups the LLM work surfaced

| Follow-up | Belongs with |
|---|---|
| Llama 3's `rope_scaling` changes the low RoPE frequencies at every position, so Llama-3.2 logits need it even on short text || Done in LLM-6 |
| `SwiGLUModule` without biases: Llama-family MLPs have none, and IO-4's strict mode fails on unmapped keys | Done in IO-4 |
| Gemma 3's `query_pre_attn_scalar` replaces the `1/sqrt(head_dim)` scale | Done in LLM-9 |
| Grouped-query attention materializes the repeated K and V; index the shared head inside the matmul instead | HIP-6 |
| The CPU `gemm` is a single-threaded i-j-p loop that strides down the weight matrix; reorder to i-p-j (same summation order per element) and thread over rows | Done in [#82](https://github.com/Joshuaweg/pulsatrix/pull/82): i-p-j, threaded over columns; Llama-3.2-1B's golden run takes 33 s instead of more than an hour |

## TOK: Tokenizers

pulsatrix has one tokenizer today: a lowercasing whitespace and punctuation splitter that builds
a word-level vocabulary. That is enough for the toy text examples and for nothing else. The v1.2
target models use four different pipelines, read from their `tokenizer.json` files (2026-10-04):

| Model | Normalizer | Pre-tokenizer | Model | Decoder |
|---|---|---|---|---|
| SmolLM2-135M | none | individual digits, then byte-level with the GPT-2 regex | BPE, 49k | byte-level |
| Qwen2.5, Qwen3 | NFC | Qwen regex (single digits), then byte-level | BPE, 151k | byte-level |
| Llama 3.2 | none | Llama 3 regex (`\p{N}{1,3}`), then byte-level | BPE, 128k, `ignore_merges` | byte-level |
| Gemma 3 | spaces to `▁` | none in effect | BPE, 262k, byte fallback | `▁` to space, bytes, fuse |
| BERT | lowercase, clean, CJK spacing | BERT punctuation split | WordPiece, `##` | WordPiece |
| gpt-oss, Mistral Nemo | none | o200k regex (splits on letter case and combining marks), then byte-level | BPE, 200k / 131k | byte-level |
| DeepSeek-V3 | — | digit groups, CJK runs, then its own regex, then byte-level | BPE, 128k | byte-level |
| T5 | precompiled SentencePiece charsmap | whitespace, `▁` | Unigram | `▁` |

OpenAI's tokenizers (`cl100k_base` for GPT-3.5 and GPT-4, `o200k_base` for GPT-4o and later) are
byte-level BPE too, distributed as `tiktoken` files. Anthropic doesn't publish Claude's tokenizer;
the only exact count is its token-counting API, which the Anthropic provider (AGT-3) can use for
budgets.

Byte-level BPE is nearly universal, and what differs between families is the regex that splits
text before BPE runs. So TOK-2 runs those regexes with a small engine rather than one hand-written
pre-tokenizer per family. "A byte-level BPE tokenizer" covers SmolLM2, Qwen and Llama but not Gemma 3, which LLM-9 and
FEAT need. The plan is the same component pipeline Hugging Face `tokenizers` uses (normalizer,
pre-tokenizer, model, post-processor, decoder), read from `tokenizer.json`, so each new model
family is a few components rather than a new tokenizer.

Every tokenizer returns character offsets with its ids. Explanations are made per token but read
per word: the token relevance view (VIZ-6a), AttnLRP on text (LLM-7) and agent tool-choice
attribution (AGT-7) all need to map a token back to its span of the input, and to merge subword
scores into word scores.

| ID | Item | Why | Depends on | P | Effort | Status |
|---|---|---|---|---|---|---|
| TOK-1 | A `Tokenizer` interface: `encode` returns ids, token strings, character offsets and a special-token mask; `decode` returns text. Added tokens (special tokens such as `<\|im_start\|>`) are split out before anything else runs. The current tokenizer becomes the `WordLevel` model with a whitespace pre-tokenizer, and a byte tokenizer (256 ids plus specials) and a character tokenizer join it | One interface for every model, and offsets for every explanation | — | P0 | S || Done, [#84](https://github.com/Joshuaweg/pulsatrix/pull/84) (see below) |
| TOK-2 | Byte-level BPE from `tokenizer.json`, with a small regex engine for the Split pre-tokenizer: alternation, character classes with Unicode categories (`\p{L}`, `\p{Lu}`, `\p{M}`, `\p{N}`…) over a generated category table, ranges, quantifiers, `(?i:…)` and the `(?!\S)` lookahead; `std::regex` can't match Unicode categories. Also digit splitting, an NFC normalizer from generated Unicode tables, both merge formats (`"a b"` strings and `["a", "b"]` pairs), `ignore_merges`, and template post-processing (BOS and EOS). It passes when ids match Hugging Face on a 10,000-line multilingual corpus for SmolLM2, Qwen2.5, Llama 3.2 and gpt-oss | Any byte-level BPE model loads without new code: SmolLM2, Qwen, Llama, gpt-oss, Mistral, DeepSeek, Phi-4 | TOK-1 | P0 | M || Done, [#85](https://github.com/Joshuaweg/pulsatrix/pull/85) (see below) |
| TOK-3 | SentencePiece-style BPE: byte fallback (`<0x41>` tokens), `▁` replacement, `fuse_unk`, and the matching decoder chain. It passes the same corpus test for Gemma 3 | Gemma 3 and Gemma Scope 2 (LLM-9, FEAT) | TOK-2 | P1 | S || Done, [#90](https://github.com/Joshuaweg/pulsatrix/pull/90) (see below) |
| TOK-4 | Word-level aggregation: merge per-token scores (relevance, attributions, probe outputs) into per-word scores using the offsets, by sum, mean or maximum | Explanations people can read | TOK-1 | P1 | S || Done, [#86](https://github.com/Joshuaweg/pulsatrix/pull/86) (see below) |
| TOK-5 | WordPiece (BERT normalizer and pre-tokenizer, `##` continuation) | BERT-family encoders, the most common models in XAI papers and tutorials | TOK-1 | P2 | S | |
| TOK-6 | A byte-level BPE trainer that writes `tokenizer.json`, so a model trained from scratch in pulsatrix gets a real subword vocabulary that Hugging Face can also load | Small models trained on your own corpus | TOK-2 | P2 | M | |
| TOK-7 | Unigram (SentencePiece) with the precompiled charsmap normalizer | T5, ALBERT, XLNet and mBART | TOK-1 | P3 | M | |
| TOK-8 | A `.tiktoken` loader (base64 token and rank per line; merge priority is the rank of the merged token) with OpenAI's `cl100k_base` and `o200k_base` patterns, checked against `tiktoken` | Count and inspect tokens exactly as GPT models see them, with no Python | TOK-2 | P2 | S | |

Not planned: SentencePiece `.model` protobuf files (every target model also ships
`tokenizer.json`) and a local Claude tokenizer (not public).

### How the TOK work departed from the plan

- **TOK-1** is one concrete `TextTokenizer` built from components (`text_tokenizer.hpp`), as Hugging
  Face's `Tokenizer` is, rather than an interface each tokenizer implements. The word-level, byte
  and character tokenizers are factory functions (`tokenizer_components.hpp`).
  - Offsets are UTF-8 byte offsets into the input, not characters. A token's source is then a plain
    substring. A byte-level token inside a multi-byte character covers just its own bytes.
  - `NormalizedString` records each byte's origin, so normalizers and pre-tokenizers that change
    text (NFC, byte-level mapping in TOK-2) keep offsets.
  - `Tokenizer::Tokenize` stays as it is for `TextDataset` and existing vocabularies. The new
    word-level tokenizer matches it on ASCII. It keeps a non-ASCII character as one token, where
    `Tokenize` splits it into bytes.
  - Added tokens are matched in the raw text, longest first. Hugging Face can also match them after
    normalization (`normalized: true`). No target model needs that, so the loader refuses it when
    there is a normalizer.
- **TOK-2** loads `tokenizer.json` (`LoadTokenizerJson`) with NFC, the Split, Digits, ByteLevel and
  Sequence pre-tokenizers, BPE, the ByteLevel and TemplateProcessing post-processors and the
  ByteLevel decoder. Anything else is refused by name.
  - **Parity:** on a 10,000-line corpus, ids and decoded text match `tokenizers` 0.23 exactly for
    SmolLM2-135M, Qwen2.5, Qwen3, Llama 3.2 and gpt-oss. That is 50,000 encodings, about 1.3 s per
    tokenizer for the whole corpus. The corpus is the 90 CI stress lines, the UDHR in about 530
    languages (downloaded, not committed) and seeded mixtures of the two.
  - **CI fixtures:** CI uses each model's real pipeline with a small trained vocabulary
    (`tools/tokenizers/make_tokenizer_reference.py tiny`), about 140 KB each. The Llama and
    gpt-oss fixtures add whole words no merge reaches, so `ignore_merges` is exercised.
    Mutation-checked: removing NFC, `ignore_merges` or the exact `\s` set fails a test.
  - **Regex engine:** `UnicodeRegex` is a backtracking matcher over code points with Oniguruma's
    leftmost-first semantics. `\s` is White_Space, as in Oniguruma, checked against `tokenizers`
    on the edge cases (U+001C to U+001F, U+0085, U+00A0, U+200B, U+FEFF).
  - **Unicode tables:** categories and NFC data are generated from Unicode 15.0, the version of the
    Oniguruma `tokenizers` bundles (`tools/unicode/generate_unicode_data.py`).
  - **Offsets:** they are compared exactly on lines that round-trip. On other lines, Hugging
    Face's are positional: a composed character gets only its first source character,
    characters after a canonical reordering get their neighbours' spans, and tokens after a
    character the vocabulary lacks (SmolLM2 has no byte 0x1D) shift left. pulsatrix keeps each
    byte's actual source, and the harness reports those lines separately.
  - **JSON parser:** duplicate keys are now found with a hash set past 32 keys. The old scan was
    quadratic, about 2×10¹⁰ comparisons for gpt-oss's 200k-entry vocabulary.
- **TOK-3** (`sentencepiece_bpe.hpp`) adds the SentencePiece-style pieces. It covers spaces as
  `▁` through the Replace and Prepend normalizers or the newer Metaspace pre-tokenizer, BPE byte
  fallback and fused unknown tokens, and the ByteFallback, Replace, Strip, Metaspace and Sequence
  decoders.
  - Decoders became chains, as in Hugging Face: `decode_chain` maps a token list to a token list.
    The TOK-2 decoders are unchanged.
  - **Parity:** Gemma 3, TinyLlama (the Llama 2 format) and Zephyr (Mistral 7B's) match
    `tokenizers` exactly on the 10,000-line corpus, in ids, offsets and decoded text.
  - **Prepend offsets:** a prepended `▁` that becomes a token of its own now gets the first
    character's span, as Hugging Face's `NormalizedString::prepend` gives it. That holds for the
    Prepend normalizer, Metaspace and ByteLevel's `add_prefix_space`. Found by tightening the
    harness: reference lines are excused from exact offsets only where NFC-style Unicode
    normalization changed them, no longer for any normalizer.
  - **Metaspace decoder:** Hugging Face drops every `▁` in the first token, not only the one it
    added, so pulsatrix does too.
  - **CI fixtures:** Gemma 3 and TinyLlama, plus variants no target uses: Gemma 3 without byte
    fallback (fused unknown tokens) and the Metaspace form with and without splitting. Each has
    a small alphabet, so byte fallback runs constantly.
  - **Mutation-checked:** removing byte fallback, `fuse_unk` or the Prepend offset rule fails the
    fixtures that use it.
- **TOK-4** (`AggregateToWords`, `word_scores.hpp`) takes words from the text, not from the
  tokenizer, so every tokenizer gives the same words.
  - A token across several words splits its score by the bytes it has inside each. The rule
    itself is our own choice. `Sum` therefore conserves the total, with the words plus
    `unassigned` adding up to the tokens' total. That is checked on every line of the
    tokenizer corpus, through Llama 3.2's pipeline with its BOS token.
  - `MaxAbs` joins the planned sum, mean and maximum. Signed attributions need it, because `Max`
    hides strong negative evidence.
  - On SmolLM2, AttnLRP for " Paris" after "The Eiffel Tower is located in the city of" puts
    +4.09 on "Eiffel"; the next largest word is +0.30.

## XAI: Question-driven explainability framework

This epic builds two frameworks into the library:

- **Four reasons to explain** (Adadi and Berrada, 2018): to *justify* a decision, to *control*
  the model (find its flaws), to *improve* it, and to *discover* something new from it.
- **Nine question categories** (Liao, Gruen and Miller, CHI 2020): Input (data), Output,
  Performance, How (global), Why, Why not, What if, How to be that (a different prediction), and
  How to still be this (the same prediction).

You state the question and the reason. Pulsatrix picks the explainers that answer that question,
runs the checks that reason requires, and returns a report that says what the result does and
doesn't support.

| Question | Primary method | Status in pulsatrix |
|---|---|---|
| Why | LRP composites, Integrated Gradients | Have |
| Why not | Contrastive LRP (relevance of logit P minus logit Q) | Have |
| What if | PDP, ICE, ALE, local and global (Morris, Sobol) sensitivity, occlusion, re-running with an edited input | Have ([CFS](#cfs-counterfactuals-and-sensitivity)) |
| How (global) | Global surrogate tree, PDP, TCAV, CRP | Partial; XAI-4, INT-6 |
| How to be that | Counterfactual search | Have: gradient-based, model-agnostic and diverse sets ([CFS](#cfs-counterfactuals-and-sensitivity)) |
| How to still be this | Anchors | Gap; XAI-4 |
| Performance | Metrics, calibration, per-slice error, uncertainty | Partial; KS-3 |
| Input (data) | Dataset statistics, nearest training examples, data attribution | Partial; INT-1, KS-5 |
| Output | Output schema and label documentation | Template only |

The reason changes the defaults:

- **Justify:** faithfulness and stability metrics are required, and the report carries
  uncertainty.
- **Control:** the model-randomization test and per-slice error are required.
- **Improve:** attributions are aggregated over the dataset to surface spurious features.
- **Discover:** concept and featurizer tools run, each with a random-model baseline.

| ID | Item | Why | Depends on | P | Effort | Status |
|---|---|---|---|---|---|---|
| XAI-1 | `XaiQuestion` (9 values) and `XaiReason` (4 values), and a `capabilities()` declaration on every explainer | Machine-readable "what this explainer answers" | — | P0–P1 | S | |
| XAI-2 | `ExplanationPlanner::plan(question, reason, model_traits)`, returning ranked explainers and the checks that must run. Optional, and never hides the direct explainer APIs | The framework users actually call | XAI-1 | P1 | M | |
| XAI-3 | An `Explanation` report: the attribution plus question, reason, method, settings, metric results and caveats. Caveats are filled in automatically from known failure modes | Explanations that say what they don't support | XAI-2 | P1 | M | |
| XAI-4 | The missing explainers: a global decision-tree surrogate with a fidelity score, local fidelity output for LIME, Anchors, prototypes from nearest neighbors, and pertinent negatives | Covers How to still be this and How (global); counterfactuals moved to CFS | XAI-1, INT-1 | P1 | M each | |
| XAI-5 | Explanation-quality metrics in the Quantus families: deletion and insertion curves (with the ROAD correction), the model-parameter randomization test, sparseness and complexity | Today pulsatrix has conservation and stability checks only | — | P0 | M | Done, [#53](https://github.com/Joshuaweg/pulsatrix/pull/53) |
| XAI-6 | `NullModelBaseline`: re-run any explainer, probe or featurizer on a re-initialized copy of the model and report the difference | The baseline rule above, as one call | FND-1 | P0 | S–M | Done, [#54](https://github.com/Joshuaweg/pulsatrix/pull/54) |

### How the XAI work departed from the plan

- **XAI-5.** The metrics take a prediction function and a finished attribution, so they work with
  any explainer. ROAD's imputation solves every removed pixel jointly as the weighted mean of its
  neighbors, plus seeded noise. Sparseness and complexity score an all-zero attribution as 0,
  where Quantus returns NaN.
- **XAI-6.** `NullModelBaseline` is a template over any analysis and any result type, not just
  explainers. Re-initialization draws each tensor from N(0, σ) with that tensor's own σ, keeps
  frozen flags and leaves buffers alone. XAI-5's randomization test was rebuilt on its
  `ParameterSnapshot` and `ReinitializeParameters`. Both use a portable random generator, so
  results match on Linux and Windows.

### Follow-ups the XAI work surfaced

| Follow-up | Found in | Belongs with |
|---|---|---|
| Per-pixel aggregation across channels; metrics treat each input element as a feature today | XAI-5 | VIZ-2 or XAI-3 |
| Quantus's localisation, robustness and axiomatic families | XAI-5 | KS-4 (robustness), or its own item |
| A built-in plain-linear-probe comparison, the second half of the baseline rule | XAI-6 | INT-3 (probe controls) |
| Running the null-model baseline and the metrics from the report planner | XAI-6 | XAI-2, XAI-3 |

## CFS: Counterfactuals and sensitivity

Two of the nine questions had almost nothing behind them. "What if" had PDP only; the table above
used to claim ICE, which pulsatrix never had. "How to be that" had nothing. This epic covers both,
model-agnostic where it can be (a prediction function, like LIME and PDP), and each method ships a
`pulsatrix.<kind>.v1` document (VIZ-1) and an SVG view (VIZ-2).

| ID | Item | Why | Depends on | P | Effort | Status |
|---|---|---|---|---|---|---|
| CFS-1 | ICE: one curve per instance, centered ICE (c-ICE) and derivative ICE, plus two-feature partial dependence. View: PDP over ICE lines, and a 2-D PDP heatmap. Checked against scikit-learn's `partial_dependence` | PDP averages away heterogeneity; ICE shows it. Two-feature PDP shows interactions | — | P0 | S || Done, [#69](https://github.com/Joshuaweg/pulsatrix/pull/69) (see below) |
| CFS-2 | ALE (accumulated local effects), first order, with quantile bins. Checked against PyALE | PDP reads the model at impossible inputs when features are correlated; ALE doesn't | CFS-1 | P1 | S || Done, [#72](https://github.com/Joshuaweg/pulsatrix/pull/72) (see below) |
| CFS-3 | Local sensitivity: move each input feature by ±δ (absolute or in units of the background's spread) or across its range and record the output change; occlusion with patches for images and spans for sequences. Views: a tornado chart, and the occlusion map through the heatmap renderer | "Which inputs is this prediction sensitive to, and how much?" with no gradients and no surrogate | — | P0 | S || Done, [#70](https://github.com/Joshuaweg/pulsatrix/pull/70) (see below) |
| CFS-4 | Global sensitivity: Morris elementary effects (μ\*, σ) and Sobol first-order and total indices (Saltelli sampling, Jansen estimators) with bootstrap confidence intervals. Views: μ\*–σ scatter, Sobol bars with error bars. Checked against SALib | Which inputs drive the output over the whole input space, and which interact | — | P1 | M || Done, [#73](https://github.com/Joshuaweg/pulsatrix/pull/73) (see below) |
| CFS-5 | Gradient counterfactual (Wachter): the nearest input that reaches a target class or value, distance weighted by each feature's median absolute deviation, with immutable features, bounds, and integer or categorical features. Reports validity, proximity (L1, L2) and sparsity. View: what changed, feature by feature | "How to be that", the most requested missing explainer | — | P0 | M || Done, [#71](https://github.com/Joshuaweg/pulsatrix/pull/71) (see below) |
| CFS-6 | Model-agnostic counterfactual: growing spheres, for models with no gradient | Counterfactuals for any prediction function | CFS-5 | P1 | S || Done, [#74](https://github.com/Joshuaweg/pulsatrix/pull/74) (see below) |
| CFS-7 | Diverse counterfactuals (DiCE: a determinantal diversity term) and a plausibility score (distance to the k nearest background instances). | One counterfactual hides the other ways to change the outcome; implausible ones mislead | CFS-5 | P1 | M || Done, [#75](https://github.com/Joshuaweg/pulsatrix/pull/75) (see below) |

Every item runs on a re-initialized model too (rule 1): sensitivity that looks the same on a random
network says nothing about what the model learned.

### How the CFS work departed from the plan

- **CFS-1.** ICE takes a prediction function and a background set, like `PDP`; the ICE average
  equals `PDP`'s curve exactly. The document holds the raw curves only, and the centered and
  derivative views are computed when drawn, so every renderer agrees on them. The derivative
  uses `numpy.gradient`'s formula, which handles uneven grids such as the percentile grids
  `FeatureGrid` builds. The SVG helpers moved to an internal header shared by the new views.
- **CFS-2.** ALE follows PyALE 1.2.0 (and R's ALEPlot) exactly: its type-1 quantile edges, bins
  closed on the right, and centering by the count-weighted midpoint effect. It draws through the
  partial dependence document, which gained a `method` field (`"ale"`); a document without it
  still reads as partial dependence.
- **CFS-3.** The bounds are separate from the sweep (`DeltaBounds`, `ScaledDeltaBounds`,
  `RangeBounds`), so ±δ and range sensitivity are one function. `Occlusion` follows Captum 0.9's
  rules exactly, including the cropped last window, and returns an ordinary `Attribution`, so the
  existing heatmap view draws it. The tornado chart gives each feature two bars, high and low,
  rather than one two-colored bar, because both values often move the output the same way.
- **CFS-4.** The analyzers are separate from the samplers (`AnalyzeMorris`, `AnalyzeSobol`), so
  they are checked against SALib 1.6 on SALib's own samples, digit for digit. The samplers use
  plain pseudo-random numbers from `std::mt19937_64` (whose output the standard fixes) instead of
  SALib's Sobol sequence, and are checked against the Ishigami function's analytic indices.
  Bootstrap intervals use the same generator, so they match SALib's in size only. Second-order
  Sobol indices are not computed.
- **CFS-5** uses proximal gradient descent (a gradient step on the prediction hinge, then
  soft-thresholding toward the input) rather than plain gradient descent on the L1 distance,
  so unneeded features stay exactly unchanged; on a linear model it finds the L1-optimal
  counterfactual, which the tests check in closed form. The prediction loss is a hinge on the
  outputs (Carlini and Wagner's form) rather than Wachter's squared error, so it is zero once the
  target is reached. Snapping a one-hot group can undo a change the search made only partway, so
  if it does, each category of each group is tried and the nearest valid one kept.
- **CFS-6** refines the growing-spheres point with the paper's own feature selection (features put
  back, smallest change first, while the target still holds) rather than the evolutionary module:
  that already gives sparse answers, and the module would add a second search with its own
  settings. Both searches now share one implementation of the constraints, the final rounding and
  snapping, and the metrics.
- **CFS-7** maximizes the log-determinant of DiCE's kernel instead of the determinant (same
  maximizer, gradients that don't vanish as the set grows) and adds DiCE's post-hoc sparsity step.
  It reports DiCE's set metrics (validity, diversity, count diversity) but was not run against the
  DiCE package itself: DiCE's random restarts and optimizer make digit-level comparison
  meaningless. Its tests check the property instead, and fail when the diversity term is removed.
  The set view is a grid of the counterfactuals, not a new document kind: it takes several
  counterfactual documents, as the beeswarm takes several attributions.

## INT: Embedding and representation analysis

| ID | Item | Why | Depends on | P | Effort |
|---|---|---|---|---|---|
| INT-1 | Nearest neighbors and mutual nearest neighbors over embeddings or activations, with optional mean-centering | The cheapest useful embedding tool; also a cross-model alignment score | FND-3 | P0 | S |
| INT-2 | Linear CKA (debiased) | Compare layers across models, training runs and backends | — | P0 | S |
| INT-3 | Control tasks for `LinearProbe` (selectivity = probe accuracy minus control accuracy) | High probe accuracy can come from the probe, not the representation | — | P0 | S |
| INT-4 | Anisotropy, effective rank and TwoNN intrinsic dimension | Explains why raw cosine similarity misleads; one number per layer | INT-1 | P1 | S |
| INT-5 | PCA projections | 2D and 3D views, steering directions, preprocessing for TDA | FND-4 | P1 | M |
| INT-6 | TCAV (with significance against random concepts) and Concept Relevance Propagation | Concept-level explanations, and CRP reuses pulsatrix's LRP rules | INT-3 | P1 | M |
| INT-7 | RSA and SVCCA | Mostly covered by CKA | FND-4 | P2 | — |
| INT-8 | UMAP and t-SNE: export the data only, don't implement | Expensive, and they distort distances and cluster sizes | VIZ-1 | P3 | — |

## TDA: Topological data analysis

Persistent homology measures the shape of a point cloud: its clusters (H0), loops (H1) and
voids (H2), and how long each one lasts as the scale grows. The plan treats it as a typed
pipeline:

```
PointCloud → DistanceMatrix → Filtration → Diagram → Vectorization
```

The category theory pays off in one place: **stability**. Each stage has a bound on how much its
output can move when its input moves, and those bounds compose. So every result carries a
certified error bar (`Certified<T>{value, bound, provenance}`) that adds up the fp32 distance
error, the subsampling error and the vectorization error. The API doesn't expose `Functor`
classes that nothing checks.

Pulsatrix writes H0 itself (a minimum spanning tree, on CPU and GPU), vendors Ripser (MIT) for H1
and H2, and writes the distances, vectorizations and significance tests itself. GUDHI's
GPL-licensed parts and giotto-ph (AGPL) are avoided.

| ID | Item | Why | Depends on | P | Effort |
|---|---|---|---|---|---|
| TDA-1 | A `PointCloud` built from `ActivationSnapshot`s, with flatten and pooling options, concatenated across batches | The input to everything below | — | P0 | S |
| TDA-2 | Pairwise distances via GEMM on every backend, with an fp32 error bound | Distances are the only part that touches the model's dimension | TDA-1 | P0 | S |
| TDA-3 | H0 as a minimum spanning tree | Exact, fast and GPU-friendly | TDA-2 | P0 | S |
| TDA-4 | Vendored Ripser for H1 and H2, returning critical edges and cocycles | Loops and voids | TDA-2 | P0 | M |
| TDA-5 | `Diagram`, bottleneck and Wasserstein distances, `Certified<T>`, and property tests for stability | Compare layers, models and seeds with certified error | TDA-3, TDA-4 | P0 | M |
| TDA-6 | Betti curves, persistence landscapes, persistence images, persistent entropy | Turn diagrams into numbers for probes and plots | TDA-5 | P0 | S |
| TDA-7 | Farthest-point landmarks, which give the subsampling error bound | Persistent homology can't run on 100,000 activations | FND-3 | P0 | S |
| TDA-8 | Significance tests: a universal null distribution, bootstrap confidence bands, Gaussian and random-network baselines | Every reported loop needs a p-value | TDA-5 | P0 | M |
| TDA-9 | Barcode, persistence diagram and Betti curve views, with JSON and SVG export | See the result | TDA-5, VIZ-1 | P1 | M |
| TDA-10 | **A manifold verifier for featurizer blocks:** decide whether a BSF or SAE block forms a circle or a torus, and recover a circular coordinate to steer along. It passes by reproducing the GPT-2 day-of-week circle (Engels et al.) against a random-network control | Goodfire measures block dimension with a linear method. Nobody has published a topological check, so this is new | TDA-4, TDA-7, TDA-8, FEAT-6 | P1 | M |
| TDA-11 | Representation Topology Divergence next to CKA in one `compare_representations()` report | Catches clusters and loops that CKA misses | TDA-4, INT-2 | P1 | S |
| TDA-12 | Betti numbers layer by layer through a network, with a random-network baseline | Reproduces "topology simplifies through the layers" (Naitzat et al., 2020) | TDA-7, TDA-8 | P1 | M |
| TDA-13 | Spectral and distance-to-measure distances | Raw distances concentrate in high dimensions and hide loops | INT-1, FND-4 | P2 | M |
| TDA-14 | Mapper, with an instability score and a parameter sweep, never a single graph | Mapper output depends heavily on its parameters | FND-4, TDA-3 | P2 | M |
| TDA-15 | A differentiable persistence loss | Topology as a training signal | TDA-4 | P2 | M |
| TDA-16 | Probes on diagram features for out-of-distribution and trojan detection, compared against non-topological features | Only worth it if it beats simpler statistics | TDA-6 | P2 | M |
| TDA-17 to TDA-19 | Image/kernel persistence across layers, zigzag persistence, multiparameter persistence | Research-grade | TDA-4 | P3 | L–XL |

Not planned: "neural persistence" on weight graphs. A 2023 follow-up showed it mostly measures
weight variance.

## KD: Surrogates and student-teacher

| ID | Item | Why | Depends on | P | Effort |
|---|---|---|---|---|---|
| KD-1 | A knowledge-distillation loss (softened teacher outputs, with the T² scaling) | A wrapper around the existing `KLDivergenceLoss` | — | P1 | S |
| KD-2 | A teacher–student explanation-agreement test: compare LRP or IG maps with CKA or rank correlation | Turns distillation into a test of whether the student uses the same evidence | KD-1, INT-2 | P1 | S |
| KD-3 | Feature and attention distillation, soft decision trees, and rule extraction that feeds the Datalog engine | Interpretable students; rules connect to pulsatrix's neuro-symbolic side | KD-1 | P2 | — |
| KD-4 | Born-again networks and a DistilBERT-style recipe | Small and inconsistent gains | KD-1, IO-2 | P3 | — |

## FEAT: Featurizers, sparse autoencoders and Goodfire BSF

Goodfire's **Block-Sparse Featurizers** (BSF, June 2026) are like sparse autoencoders, except
the unit of sparsity is a small block of 2–4 dimensions rather than one direction. That lets one
feature be a curve or a circle instead of a line. BSF is the newest member of a family pulsatrix
can't represent yet, so the plan builds the family first, behind one interface.

Sparse autoencoders have real limits: they lose to linear probes out of distribution, steering
with their features loses to simple difference-of-means vectors, and BSF still splits features.
They ship here as discovery tools with metrics and baselines, not as detectors.

| ID | Item | Why | Depends on | P | Effort | Status |
|---|---|---|---|---|---|---|
| FEAT-1 | A `Featurizer` interface (encode, decode, loss, decoder normalization), with the existing SAE ported to it and given unit-norm decoders, an L0 metric and dead-latent tracking | One interface for the whole family | — | P0 | M | Done, [#102](https://github.com/Joshuaweg/pulsatrix/pull/102) (see below) |
| FEAT-2 | TopK SAE with the auxiliary loss for dead latents | Removes L1 shrinkage; sets sparsity directly | FEAT-1, FND-3 | P0 | M | Done, [#105](https://github.com/Joshuaweg/pulsatrix/pull/105) (see below) |
| FEAT-3 | Core SAEBench metrics: explained variance, loss recovered when the reconstruction is spliced back into the model, dead and dense latents, feature absorption. Random-model and probe baselines on by default | Without these, no featurizer result can be trusted | FEAT-1, XAI-6 | P0 | M | Done, [#106](https://github.com/Joshuaweg/pulsatrix/pull/106) (see below) |
| FEAT-4 | BatchTopK, JumpReLU and Matryoshka SAEs | Fix specific failures of TopK | FEAT-2 | P1 | — | Done, [#107](https://github.com/Joshuaweg/pulsatrix/pull/107) (see below) |
| FEAT-5 | Transcoders and skip transcoders | Reported to be more interpretable than SAEs | FEAT-1 | P1 | M | Done, [#108](https://github.com/Joshuaweg/pulsatrix/pull/108) (see below) |
| FEAT-6 | BSF: the vanilla, Grassmannian and group-lasso variants, then tournament top-k, with MDL and stable-rank metrics | The newest member of the family | FEAT-2, FND-4 | P1 | L |
| FEAT-7 | Steering with difference-of-means by default and featurizer directions as an option, with a reliability report | The simple baseline usually wins | FEAT-1 | P1 | S |
| FEAT-8 | Crosscoders, including the Delta-Crosscoder for fine-tuning diffs | Model diffing across layers and models | FEAT-1, IO-2 | P2 | — |
| FEAT-9 | Parameter decomposition (SPD and VPD) | Interpretability in weight space; few libraries have it | FND-4 | P2 | L–XL |
| FEAT-10 | Attribution graphs with cross-layer transcoders | The existing circuit graph plus LRP may cover most of the value first. VIZ-4's AttnLRP graph (residual-stream nodes) and Neuronpedia export are the starting point: transcoder features would replace the nodes | FEAT-5 | P2 | XL |

Open: no LRP rule exists yet for propagating relevance through a block top-k. FEAT-6 needs one.

### How the FEAT work departed from the plan

- **FEAT-1** adds the `Featurizer` interface (`featurizer.hpp`): `encode`, `decode`,
  `loss_and_backward`, `normalize_decoder` and `decoder_direction`. Alongside it:
  - `TrainFeaturizer`, a step for any featurizer;
  - `MeanL0`;
  - `FeatureActivityTracker`, for dead latents and dense ones (firing rates).
  - **The port.** `SparseAutoencoder` implements the interface, and is now also a `Module`
    (input to reconstruction, with backward, LRP and named parameters), so checkpoints work.
    Its old API is unchanged.
  - **Unit-norm decoders, on by default.** Normalizing moves each direction's length into the
    encoder's row and bias. A ReLU passes a positive scale through, so reconstructions don't
    change; a test checks it.
  - **This changed an existing test.** The L1 control in `sparse_autoencoder_test.cpp` measured
    sparsity by the mean activation, and with free decoders most of that drop was the penalty
    shrinking codes, not silencing features. At l1_lambda 0.05:

    | Decoders | Mean activation | L0 (active features per input) |
    |---|---|---|
    | Free | ×0.19 | ×0.58 |
    | Unit-norm | | ×0.21 |

    Under unit norm 0.05 also collapses reconstruction on that Gaussian data. The control now
    uses 0.01 and checks L0, and a new test pins the comparison above.
  - The SAE recipe's expected output changed with it.
- **FEAT-2** adds `TopKSparseAutoencoder` (`topk_sparse_autoencoder.hpp`), a `Featurizer`
  and a `Module`.
  - **Checked against PyTorch,** not against Gao et al.'s code: OpenAI's `sparse_autoencoder`
    package isn't maintained and needs Triton. A PyTorch rendering of the paper's method gives
    the loss, the gradients and two Adam steps, matched to 1e-5.
  - **AuxK only reaches latents with positive pre-activations,** as in the paper: the ReLU
    applies to the auxiliary codes too. A latent pushed below zero everywhere stays dead.
  - **Unit-norm decoders without moving the scale into the encoder,** as Gao et al. do. Moving
    it, as FEAT-1 does for the L1 SAE, would change which latents win the top k.
  - **`b_dec` starts at the data's mean,** not its geometric median.
  - **Dead-latent counters aren't in checkpoints;** `inputs_since_fired()` and
    `set_inputs_since_fired()` save and restore them.
  - **Selection runs on the host.** `top_k` runs on the device, but the codes, the auxiliary
    selection and the gradient routing are host loops. That is fine at PLM-8's scale; a large
    SAE would want them on the device.
- **FEAT-3** adds `featurizer_metrics.hpp`: `EvaluateReconstruction` (explained variance,
  cosine, norm ratio, L0, dead and dense latents), `MeasureLossRecovered` with `SpliceHook` and
  `AblationHook`, and `FeatureAbsorption`.
  - **Splicing needed a hook.** `CausalLM` and `EncoderLM` gain `set_hidden_state_hook()`,
    called at every position during `forward()`; what it returns replaces the hidden states.
    It doesn't run in `next_token_logits()`'s cached passes.
  - **Absorption works for any binary concept,** not only first letters as in Chanin et al.
    The probe is fit on half the inputs, the main features are picked by SAEBench's k-sparse
    rule (each must raise F1 by 0.03), and the rate is scored on the other half. The probe is
    the baseline and is always reported.
  - **Its unit test plants the answer.** A featurizer with set weights has a parent concept
    and ten rare children that absorb it; every absorbed input is found, and none in a
    featurizer without the hierarchy.
  - **The random-model baseline runs in the tool,** not inside the metrics:
    `pulsatrix_probe_esm --sae` trains the same featurizer on a randomly initialized ESM-2 and
    reports every metric for both. On ESM-2 8M it changed the reading of three results:
    - the random model's SAEs explain more variance;
    - its loss can't be recovered, because ablating the layer doesn't hurt it;
    - for weakly probed concepts it "absorbs" as much as the trained one.

    So `recovered` is NaN unless ablation raises the loss, and the tool prints absorption only
    where the probe's F1 is at least 0.5. The numbers are in the
    [mechanistic-interpretability guide](../mechanistic-interpretability/index.md#measuring-a-featurizer).
  - **`pulsatrix_probe_esm --featurizer topk`.** At the same L0, a TopK SAE beats PLM-8's L1
    SAE on every measure: explained variance 0.78 vs 0.66, loss recovered 0.96 vs 0.90.
- **FEAT-4** adds BatchTopK and Matryoshka as options of `TopKSparseAutoencoder`
  (`batch_topk`, `matryoshka_prefixes`), and `JumpReLUSparseAutoencoder`.
  - **Options, not classes, for the TopK family.** Both variants change only selection and the
    loss, so they share TopK's AuxK, decoder handling and Module code. BatchTopK's inference
    threshold, a moving average of each training batch's smallest kept activation, is a buffer,
    so checkpoints keep it.
  - **Matryoshka sums its prefix losses,** as the paper does: FeaturizerLoss::total is that sum
    plus AuxK, and `reconstruction` is the full dictionary's error.
  - **JumpReLU's λ is per element.** The loss uses the mean squared error over elements, where
    the paper sums over dimensions, so λ is smaller by the input dimension. There is no λ
    warm-up built in; `set_l0_coefficient()` lets a training loop do it.
  - **All three match PyTorch renderings of their papers**
    (`tools/golden/make_sae_variants_golden.py`): loss, gradients (JumpReLU's straight-through
    threshold gradients included) and two Adam steps.
  - **The absorption test needed a tight budget.** At k = 3, with about 1.6 true features per
    input, plain BatchTopK absorbed nothing: merging a parent into its children saved no
    latents. At k = 2 it absorbs in 6.6% to 12.6% of inputs over four seeds, and Matryoshka in
    none.
  - **On ESM-2 8M** Matryoshka absorbs least and finds the best single features for helix and
    transmembrane, at 0.90 loss recovered to TopK's 0.96. JumpReLU matches TopK's loss
    recovered with 12.6 latents instead of 16
    ([table](../mechanistic-interpretability/index.md#batchtopk-matryoshka-and-jumprelu)).
- **FEAT-5** adds `Transcoder` (`transcoder.hpp`), with an optional skip connection, and an MLP
  hook on `EncoderBlock` and `TransformerBlock`.
  - **TopK activation and AuxK,** as Paulo et al. train them, rather than Dunefsky et al.'s L1
    penalty.
  - **The interface grew, compatibly.** `Featurizer` gains `output_dim()`, `predict()` (the
    full output, skip connection included) and `loss_and_backward_with_target()`, which
    autoencoders refuse. `TrainFeaturizer` gains an overload with targets.
    `EvaluateReconstruction` is now `EvaluatePrediction` with the input as its target, and
    uses `predict()`.
  - **Splicing an MLP needed its own hook,** since `HiddenStateHook` sits between blocks.
    `set_mlp_hook()` sees the MLP's input and output, and its return value is what reaches the
    residual stream. It reads the training pairs and splices the transcoder in
    (`MlpSpliceHook`, `MlpAblationHook`, `MeasureMlpLossRecovered`). It doesn't run in
    `forward_cached()`.
  - **Absorption refuses transcoders.** It compares decoder directions with a probe in the
    input space, and a transcoder writes to another space.
  - **Checked against a PyTorch rendering** (`tools/golden/make_transcoder_golden.py`), with
    different input and output sizes, the skip connection and AuxK.
  - **On ESM-2 8M** the skip transcoder beats the plain one on every measure (loss recovered
    0.72 vs 0.59 with the layer-4 MLP replaced). Against a residual-stream TopK SAE its concept
    features are mixed: better on 3 of 6 concepts, worse on 2
    ([table](../mechanistic-interpretability/index.md#transcoders)).

## ARCH: New architectures

These are ranked by how well they fit an explainability-first library.

| ID | Item | Why | Depends on | P |
|---|---|---|---|---|
| ARCH-1 | B-cos layers | Self-explaining networks whose explanation can be compared directly with LRP | — | P1 |
| ARCH-2 | Concept bottleneck models | Supports "what if this concept were true" interventions. Watch for concept leakage | — | P1 |
| ARCH-3 | Mixture of experts with router attribution | The router's choice is a free, discrete explanation | FND-3 | P1 |
| ARCH-4 | Prototype networks (ProtoPNet) | "This looks like that" explanations | INT-1 | P2 |
| ARCH-5 | Graph neural networks with GNN-LRP | A new data type; `gather_rows` and `scatter_add_rows` already exist | — | P2 |
| ARCH-6 | Mamba-2 / SSD | Extends the existing Mamba and MambaLRP | — | P2 |
| ARCH-7 | xLSTM | Extends the recurrent family | — | P2–P3 |
| ARCH-8 | Kolmogorov–Arnold networks | Interpretable by construction, but they lose to MLPs outside symbolic tasks | — | P2 |
| ARCH-9 | Diffusion transformers (adaLN) | The timestep embedding and noise schedules already exist | — | P2 |
| ARCH-10 | Gated DeltaNet and Titans, Hopfield layers, JEPA, self-explaining neural networks, hypernetworks, neural ODEs and liquid networks | Lower fit or poor match with hand-written backward passes | — | P3 |

## VIZ: Visualization pack

Interactive visualization libraries for interpretability tend to go stale, while stable data
formats last. The plan keeps ImGui for live views and puts every view on top of a versioned JSON
format, with static and web renderers next to it.

| ID | Item | Why | Depends on | P | Effort | Status |
|---|---|---|---|---|---|---|
| VIZ-1 | A versioned JSON document model (`pulsatrix.<kind>.v1`) for attributions, heatmaps, token relevance, circuit graphs, training logs and feature dashboards. The ImGui widgets read it too. NaN and infinity encode as null plus a flag | One source of truth for every renderer | — | P0 | M | Done, [#62](https://github.com/Joshuaweg/pulsatrix/pull/62) (see below) |
| VIZ-2 | A dependency-free SVG renderer: bar, waterfall, heatmap, token strip, beeswarm | Publication figures with no GPU or display, including in CI | VIZ-1 | P0 | M | Done, [#63](https://github.com/Joshuaweg/pulsatrix/pull/63) (see below) |
| VIZ-3 | Self-contained Vega-Lite HTML, with the JavaScript inlined or loaded from a CDN | Hover, zoom and export for free, in a browser or notebook | VIZ-1 | P1 | S | Done, [#92](https://github.com/Joshuaweg/pulsatrix/pull/92) (see below) |
| VIZ-4 | Export circuit graphs in the attribution-graph schema that Neuronpedia and circuit-tracer read | Large graphs get a mature viewer for free | VIZ-1 | P1 | M | Done, [#93](https://github.com/Joshuaweg/pulsatrix/pull/93) (see below) |
| VIZ-6 | New views: (a) token relevance for text, (b) feature dashboards, (c) an embedding projector, (d) an attention head grid, (e) SHAP force, decision and dependence plots | Fill the gaps between the current widgets and the reference tools | VIZ-1 | P2 | M each | (a) Done, [#87](https://github.com/Joshuaweg/pulsatrix/pull/87) (see below) |
| VIZ-7 | A node editor for circuit graphs | Only if graphs outgrow the current view; VIZ-4 covers large ones | — | P3 | M | |
| VIZ-8 | Shaped text in the ImGui widgets. Each piece is shaped with HarfBuzz (through FreeType) and right-to-left runs are reordered (FriBidi, or UAX #9 for single-direction pieces); the shaped glyphs are rasterized with FreeType into a glyph atlas and drawn as textured quads in `TokenRelevanceView`. It passes when Arabic (joined, right to left), Hebrew, Devanagari, Bengali and Tamil (vowel signs and conjuncts) and Thai render the way HarfBuzz's own `hb-view` does, checked against reference images, and emoji ligatures (skin tones, ZWJ families, flags) show as one glyph | Every character already shows (ImGui 1.92 plus system fonts), but ImGui places characters one after another, so complex scripts aren't readable in the native windows; today only the SVG shows them correctly | — | P3 | M | |

(VIZ-5, the notebook path, moved to the NB epic.)

### How the VIZ work departed from the plan

- **VIZ-1.** "Null plus a flag" became null plus a top-level `"nonfinite"` object that maps each
  non-finite number's JSON Pointer to `"nan"`, `"inf"` or `"-inf"`, so a round trip loses
  nothing. A reader rejects a null with no entry and an entry that points at no null. The
  documents needed a JSON library, so the work added `pulsatrix/json.hpp`: a strict RFC 8259
  parser, fuzzed, and a deterministic writer whose numbers round-trip bit for bit. Readers
  ignore unknown fields, so v1 can grow without a new version. KS-2's benchmark reports follow
  the same rules.
- **VIZ-2** added the `pulsatrix_svg` command-line tool. Heatmaps above 4,096 cells embed a
  lossless PNG (224 × 224 is about 45 KB, not about 3 MB of rectangles). Non-finite heatmap
  cells and tokens are drawn gray; the bar, waterfall and beeswarm charts refuse non-finite
  values. Every stored test figure was rendered and checked by eye before it became a golden
  file.
- **VIZ-3** (`viz/html.hpp`) covers every SVG view plus two the SVG lacks: the training log
  and the feature dashboard. Each view is also available as a bare Vega-Lite spec for your own
  page.
  - Pages load Vega 6.4.0, Vega-Lite 6.4.3 and vega-embed 7.3.0 from jsDelivr, pinned and
    checked with Subresource Integrity hashes.
  - Or they carry the scripts inline (about 830 KB) from a directory that
    `tools/render/fetch_vega.sh` fills after checking the files' SHA-384 hashes. Nothing is
    downloaded at build time.
  - The token relevance page is plain HTML with no scripts, so the browser shapes the text:
    right to left, joined Arabic, Indic conjuncts.
  - Non-finite values follow the SVG rules: gray heatmap cells and tokens. Training logs skip a
    diverged step instead of refusing the log.
  - Every spec was compiled with Vega-Lite and rendered with Vega in Node with no warnings
    (`tools/render/validate_vega.mjs`). The pages were checked by eye in headless Chromium,
    including a line in ten scripts.
  - Circuit graphs have no HTML view; VIZ-4 exports them to Neuronpedia's viewer instead.
- **VIZ-4** checked the falsifier first. Neuronpedia's `graph-schema.json` needs no
  transcoder-specific fields: `feature_type` is free text and `feature` may be null. But the
  viewer lays nodes out as layer × token, and the circuit graph has no tokens.
  - So VIZ-4 also builds a token-level graph for language models from AttnLRP
    (`BuildRelevanceGraph`). Nodes are the residual stream at each layer and token. Links run
    each block's LRP once per output position, which is exact because the rules are linear in
    the incoming relevance.
  - A node's outgoing links sum to its relevance (tested on the four tiny models). Breaking the
    per-position split fails that test. The embedding row equals the token view (6e-7 on
    Qwen2.5-0.5B).
  - Layer totals aren't equal from layer to layer: LRP through norms and biases isn't
    conservative, the same as in LXT.
  - `CausalLM::propagate_relevance_by_layer` returns the relevance at every block boundary;
    `propagate_relevance` now calls it, so LXT parity covers both.
  - Viewer quirks, found in circuit-tracer's frontend and handled:
    - the output node's label must hold `(p=...)`, which the viewer parses;
    - features seen at more than ⅔ of the positions are hidden, so residual nodes use their
      position as their feature, and circuit graphs get two columns instead of one.
  - Checked: graphs validate against `graph-schema.json` with Python `jsonschema`. They were
    viewed in circuit-tracer's frontend in headless Chromium: the Qwen2.5-0.5B " Paris" graph,
    clicked through, and an exported circuit graph.
- **VIZ-6a** builds token relevance documents from real explanations
  (`MakeTokenRelevanceDocument`, `MakeWordRelevanceDocument`), rather than adding a new
  document kind. `token_relevance.v1` gained optional `granularity`, `scored` (unscored context
  such as the spaces between words) and `unassigned` fields, written only when set, so existing
  documents and golden figures are unchanged.
  - Byte-level tokens that split a character are shown as one piece with their scores summed.
  - It added the `TokenRelevanceView` ImGui widget, `pulsatrix_explain_text` (AttnLRP on a
    Hugging Face model, straight to a figure) and `token_relevance_demo`.
  - The figures and the widget were checked by eye on SmolLM2 explanations: English, and a German,
    French and emoji line.
  - ImGui's default font is Latin-1 only, so the widget needs a loaded font for other scripts;
    the SVG has no such limit.
- **Fonts.** Dear ImGui went from v1.90.9 to v1.92.5 and ImPlot from v0.16 to v0.17. ImGui 1.92
  draws glyphs on first use, so no glyph ranges need building. `VizWindow` now merges a system
  font for every script behind the default font (`viz/fonts.hpp`): one font per script through
  fontconfig on Linux, the standard system fonts on Windows and macOS. ImGui text is 32-bit
  (`IMGUI_USE_WCHAR32`), and FreeType, when installed, draws vector color emoji. Bitmap emoji fonts
  can't be scaled and are skipped. Every widget and the MNIST gallery were rendered again and
  checked by eye. ImGui still doesn't shape text (see the follow-ups).

### Follow-ups the VIZ work surfaced

| Follow-up | Found in | Belongs with |
|---|---|---|
| SVG views for circuit graphs, training logs and feature dashboards | VIZ-2 | Done: training logs and dashboards in HTML (VIZ-3); circuit graphs in Neuronpedia's viewer (VIZ-4) |
| Nothing produces `feature_dashboard.v1` yet; its fields follow SAEDashboard | VIZ-1 | FEAT |
| Byte-level BPE tokens must be decoded to UTF-8 before they go into a document | VIZ-1 | Done in VIZ-6a: pieces are the text their offsets cover |
| SVG text widths are estimated (0.6 em), so very wide scripts such as CJK can overflow labels | VIZ-2 | Done for text: widths count display columns (CJK and emoji 2, combining marks 0), and the HTML token view (VIZ-3, [#92](https://github.com/Joshuaweg/pulsatrix/pull/92)) lets the browser lay out and shape the text, so every script is exact there |
| ImGui doesn't shape text: Arabic and Hebrew show unjoined and left to right, and Indic vowel signs and conjuncts aren't formed. Every character shows (system fonts merged as fallbacks, ImGui 1.92). Shaping the widget's text with HarfBuzz plus bidi reordering, drawn as glyph quads, would fix it | Fonts work | VIZ-8 |

## NB: Notebook layer

Notebook output needs neither Python nor a running kernel. The layers build on each other. ImGui
can't draw inline in a notebook, so it stays the desktop tool and shares the VIZ-1 data.

| ID | Item | Why | Depends on | P | Effort |
|---|---|---|---|---|---|
| NB-1 | `pulsatrix::mime_bundle_repr(const T&)` overloads that return MIME bundles: tensor summaries, attribution heatmaps as SVG, token relevance, circuit graphs and persistence diagrams as Vega-Lite. The C++ Jupyter kernel xeus-cpp finds them automatically, so pulsatrix doesn't depend on it | Rich display in a C++ notebook | VIZ-1, VIZ-2 | P1 | S |
| NB-2 | A C++ `Report` builder that writes `.ipynb` files (nbformat 4.5) and self-contained HTML: markdown, code shown as text, and rich outputs | Notebook-format results with no Python and no kernel; they render on GitHub, in VS Code and in Quarto | NB-1 | P1 | S–M |
| NB-3 | Python `_repr_mimebundle_` on the bound types, delegating to NB-1 so both languages render the same (golden test), then interactive views with anywidget | Notebook users on the Python side | NB-1 | P2 | S → M |
| NB-4 | xeus-cpp support: a `pulsatrix_notebook.hpp` header, example notebooks, and a Linux CI smoke test. Windows and GPU code inside notebook cells are marked experimental. Every precondition reachable from a cell throws instead of aborting, because a crash kills the kernel | Run pulsatrix interactively in C++ | NB-1 | P3 | M |
| NB-5 | A WebAssembly build of the CPU backend for JupyterLite | Notebooks in the browser with nothing installed | NB-4 | Deferred | XL |

## HIP: Training efficiency on AMD GPUs

The bottlenecks are structural, so measure first and fix those before anything exotic. On the
gfx1151 APU almost every non-GEMM kernel in pulsatrix is limited by memory bandwidth at best. In
practice they are limited by launch and sync overhead and by too little parallelism.

| ID | Item | Why | Depends on | P | Effort | Status |
|---|---|---|---|---|---|---|
| HIP-1 | `scripts/profile_hip.sh`: a rocprofv3 wrapper that writes kernel time per op type to CSV | The baseline, and the falsifier for every item below | — | P0 | S | Done, [#55](https://github.com/Joshuaweg/pulsatrix/pull/55) |
| HIP-2 | Reductions with one workgroup or one wave per row or channel (softmax, norms, column sums): wave shuffles using the runtime `warpSize` and 64-bit masks, then a shared-memory stage. Small channel counts get a deterministic two-stage grid reduction, no atomics | BatchNorm at N=64, C=3, 224×224 currently runs 3 threads over 3.2 million elements | HIP-1 | P0 | M | Done, [#57](https://github.com/Joshuaweg/pulsatrix/pull/57) (see below) |
| HIP-3 | A caching allocator: size classes, a pool per stream, and a configurable budget instead of `hipMemGetInfo`, which overstates what the APU can allocate. Don't use `hipMallocAsync` (open corruption bugs on RDNA) | Raw `hipMalloc` and `hipFree` on every tensor | — | P0 | M | Done, [#58](https://github.com/Joshuaweg/pulsatrix/pull/58) |
| HIP-4 | Remove the 56 per-op `hipStreamSynchronize` calls. Sync only when the host reads a result, and add `PULSATRIX_HIP_SYNC_DEBUG=1` to bring them back for debugging | Launch and sync cost more than the kernels on small models | HIP-3, FND-8 | P0 | S | Done, [#59](https://github.com/Joshuaweg/pulsatrix/pull/59) (see below) |
| HIP-5 | `dot` and `sum` across many blocks (partials, then a second pass), deterministic | They run on a single block today | — | P0 | S | Done, [#56](https://github.com/Joshuaweg/pulsatrix/pull/56) |
| HIP-6 | Fused kernels: AdamW across all parameters in one launch, bias plus activation, softmax plus cross-entropy. In explain mode they still write the values LRP needs | Fewer launches and less memory traffic | TRN-2 | P1 | — | |
| HIP-7 | Conv2D that runs im2col and GEMM in batch chunks | The first layer's im2col buffer at N=64 is about 350 MB, which competes with system RAM on an APU | FND-6 | P1 | S | |
| HIP-8 | A hipBLASLt probe on the pinned container, then bf16 GEMM through `hipblasGemmEx` | Reports conflict on whether hipBLASLt works on gfx1151 in ROCm 7.2.4; measure it | HIP-1 | P1 | — | |
| HIP-9 | A ROCm 10.0 evaluation image, the first release that officially lists gfx1151. Also pin the host kernel version (6.18.4 or newer, or the Ubuntu OEM kernel with the VGPR fix) | Known gfx1151 crashes depend on both | — | P1 | S || Done, [#65](https://github.com/Joshuaweg/pulsatrix/pull/65) (see below) |
| HIP-10 | Zero-copy staging buffers on APUs, enabled only when the device reports itself as integrated | Saves a copy on Strix Halo without slowing discrete GPUs | HIP-3 | P2 | — | |
| HIP-11 | Full bf16 training | Halves memory traffic and reaches the matrix cores. Needs a dtype in `Tensor` | IO-6 | P2 | XL | |
| HIP-12 | HIP graphs, WMMA or rocWMMA kernels, FlashAttention for training only, MIOpen | Last: graphs have measured slowdowns on gfx11, and FlashAttention never builds the attention matrix that AttnLRP needs | HIP-4 | P3 | — | |
| HIP-13 | Long-sequence encoder inference: profile ESM-2 650M at 1024 tokens with `scripts/profile_hip.sh`, then speed up the kernels that dominate (attention's `(N, H, L, L)` matmuls and softmax, the MLP GEMMs). An attention path that never builds the attention matrix is allowed here, for forwards that keep no activations | A 1024-token masked pass takes about 3.7 s on gfx1151, so ProteinGym's BRCA1 assay (1,863 residues) takes about 1 h 55 min | HIP-1 | P1 | L | |

The HIP items above are done (2026-10-04). Measurements are in [GPU Profiling](../gpu-profiling.md).

| Workload (gfx1151, Release) | v1.0 step time | After HIP-1 to HIP-5 |
|---|---|---|
| `mlp` | about 1.0 ms | 0.35 ms |
| `tagger` | 7.8 ms | 2.84 ms |
| `cnn` | about 27 ms | 5.24 ms |

The two columns come from different runs, so read them as rough; each item's own interleaved A/B
numbers are in the profiling guide. `dot` and `sum` on 16M floats went from about 17 ms to 0.61 ms and 0.30 ms, which is the measured
memory bandwidth (about 212 GB/s).

### How the HIP work departed from the plan

- **HIP-2** covers BatchNorm only. The profiler showed BatchNorm at 79% of the CNN's kernel time.
  Softmax, LayerNorm, RMSNorm and `column_sums` took about 5 µs per call, which is launch overhead,
  so by the research notes' falsifier they keep their kernels. BatchNorm uses the deterministic
  two-stage grid reduction, not wave shuffles.
- **HIP-3** keeps one pool, not a pool per stream, because the backend uses a single in-order
  stream. It was pulled forward from v1.2 because HIP-4 depends on it.
- **HIP-4** removed 59 syncs, not 56. The host still waits on device-to-host copies, on
  host-to-device copies (the source is often a temporary buffer) and before memory goes back to
  the driver.
- **Measuring under load.** Other programs held the GPU at 100% during this work, which tripled
  raw timings. HIP-3 and HIP-4 were measured with interleaved A/B runs and medians; the profiling
  guide describes the method.
- **HIP-9.** ROCm 10.0.0 passed all 2,402 HIP tests and matched 7.2.4 within 2% on every
  benchmark (six ABBA rounds of KS-2's suite), so it became the default container;
  `PULSATRIX_ROCM_VERSION=7.2.4` selects the old pin. Its image needed one fix: it doesn't
  register `/opt/rocm/lib` with the dynamic loader. "Pin the host kernel" became a check,
  `scripts/check_host_kernel.sh`, that `rocm-build.sh` runs on every call, plus documented
  instructions to hold the package; the repository doesn't change the host.

### Follow-ups the HIP work surfaced

| Follow-up | Found in | Belongs with |
|---|---|---|
| Host-side work in each step: reading the loss back, and losses that validate targets on the host. Small models are about 20% GPU-busy | HIP-4 | HIP-6 |
| Softmax, LayerNorm, RMSNorm and `column_sums` still run one thread per row; re-profile at SmolLM2 widths | HIP-2 | after LLM-1 |
| The CUDA backend still synchronizes after every op (61 calls) and allocates with raw `cudaMalloc` | HIP-3, HIP-4 | its own item |
| A pool per stream, if a second stream is ever added | HIP-3 | HIP-12 |
| Run the GPU tests both asynchronously and with `PULSATRIX_HIP_SYNC_DEBUG=1`, since a kernel fault now surfaces at the next wait | HIP-4 | KS-8 |
| Build the benchmark suite on `hip_profile_workloads` and the interleaved A/B method | HIP-1, HIP-3 | KS-2 (done) |
| CI's compile-only HIP job still uses ROCm 7.2.4: there is no slim 10.0 image, and the full one is 8.2 GB compressed | HIP-9 | KS-8 |
| ROCm 10's hipBLAS links hipBLASLt; check which path gfx1151 GEMMs take | HIP-9 | HIP-8 |

## AGT: Agents, native C++

Pulsatrix can run its own agent orchestrator without Python. There is no official C++ SDK for
the Anthropic API or for MCP, but neither is hard to write against. The plain HTTP and JSON API is
stable, and the 2026-07-28 MCP specification dropped the handshake and sessions, so a stdio server
that only offers tools is small. The genuinely hard parts are narrow:

- TLS certificate trust on Windows and macOS;
- validating JSON Schema 2020-12;
- getting a 135M-parameter model to pick tools reliably.

LLM agents live in `pulsatrix::orch` (target `pulsatrix_orchestrator`), because
`pulsatrix::Agent` already names the reinforcement-learning policy contract.

| ID | Item | Why | Depends on | P | Effort |
|---|---|---|---|---|---|
| AGT-1 | An optional `PULSATRIX_ENABLE_ORCHESTRATOR` target, off by default, with nlohmann/json (MIT). The same JSON library serves VIZ-1 and NB-1 | Keeps the core free of networking dependencies | — | P1 | S |
| AGT-2 | An `HttpTransport` interface. cpp-httplib by default, with an optional libcurl backend that uses the operating system's TLS on Windows. A shared server-sent-events parser that ignores unknown event types | Streaming LLM responses | AGT-1 | P1 | M |
| AGT-3 | Providers and the tool loop: an Anthropic provider and an OpenAI-compatible provider (llama-server, vLLM, Ollama), with one internal message format | The orchestrator itself | AGT-2 | P1 | M |
| AGT-4 | A tool registry: name, description, input schema and a `std::function`. Schemas use only the keywords that mean the same in JSON Schema draft-7 and 2020-12, checked by a small built-in validator. The MCP server and the in-process loop share one registry | Define a tool once, use it everywhere | AGT-1 | P1 | S–M |
| AGT-5 | A native MCP server over stdio (2026-07-28 specification, with a fallback for older clients) that exposes explain, attribute, logit lens, patching and featurizer tools | Lets Claude and other agents drive pulsatrix | AGT-4, IO-1 | P2 | M |
| AGT-6 | A local provider that runs pulsatrix's own LLM, with tool calls constrained by a token-level mask | Agents with no network, and a model pulsatrix can explain | LLM-1 to LLM-5, AGT-4 | P3 | M |
| AGT-7 | **Explain an agent's tool choice:** run AttnLRP from the logit of the first token that distinguishes the chosen tool, back over the prompt, the tool descriptions and the history. It passes if deleting the top-attributed tokens flips the choice more often than deleting random tokens | No other library can explain why an agent picked a tool | AGT-6, LLM-7 | P3 | L |
| AGT-8 | Automated descriptions of featurizer latents: an LLM describes the top activations, then the description is scored by how well it predicts activations | Labels thousands of features | AGT-3, FEAT-1 | P2 | M |
| AGT-9 | An interpretability agent that runs experiments on the model (MAIA-style) | Research-grade | AGT-3, AGT-4 | P3 | L |

AGT-3 has to follow the current Anthropic API rules:

- Forcing a specific tool returns HTTP 400 on the current Claude models; use automatic tool
  choice with strict tool schemas instead.
- Tool results come first in the next user message.
- A 429 from a spend limit has no `retry-after` header and must not be retried.
- Back off with jitter on 5xx, 529 and mid-stream overload errors.
- Cache the system prompt and tool definitions.
- Read the API key from the environment and never log it.

AGT-5 has to follow these security rules:

- Write nothing to stdout except protocol messages.
- Resolve every path and confine it to allowed directories.
- Load safetensors, GGUF and pulsatrix-native formats only, never pickle.
- Cap tensor and request sizes.
- Use opaque handles that expire.
- Treat tool output as untrusted.

## KS: Kitchen sink

| ID | Item | Why | Depends on | P | Effort | Status |
|---|---|---|---|---|---|---|
| KS-1 | CMake `install()` and a package config, so other projects can `find_package(pulsatrix)` | There's no install step today | — | P0 | S | Done, [#61](https://github.com/Joshuaweg/pulsatrix/pull/61) |
| KS-2 | A benchmark suite: step time, explanation time, conservation error | Measures every HIP item and catches regressions | — | P1 | S | Done, [#64](https://github.com/Joshuaweg/pulsatrix/pull/64) (see below) |
| KS-3 | Uncertainty: Monte Carlo dropout, deep ensembles, split conformal prediction | Answers the Performance question | — | P1 | S | |
| KS-4 | FGSM and PGD attacks, also used as a robustness test for explanations | Robustness and the Control reason | — | P1 | S | |
| KS-5 | TracIn data attribution | Answers "which training examples caused this" | FND-1 | P2 | M | |
| KS-6 | Fairness metrics, drift detection and a model-card generator | The Input question, and documentation | — | P2 | S each | |
| KS-7 | Wire the existing thread pool into `DataLoader`'s `num_workers` | Built but not connected | — | P2 | M | |
| KS-8 | GPU CI on a self-hosted gfx1151 runner | The HIP backend is only tested by hand today | HIP-9 | P1 | M | |
| KS-9 | A model zoo: ResNet18, VGG16 and SmolLM2 with reference heatmaps | Reproducible examples on real models | IO-4, FND-6 | P1 | M | |
| KS-10 | Strided views and broadcasting | Removes copies everywhere; touches every kernel | — | P2 | L | |
| KS-11 | Python wheels and a vcpkg port | Easier installation | KS-1 | P2 | — | |
| KS-12 | EK-FAC influence functions, quantization, op-level autograd, distributed training | Large or low priority for an explainability library | — | P3 / v2 | — | |

### How the KS work departed from the plan

- **KS-1** installs the core library as `pulsatrix::core`, its headers, the package config and
  the `pulsatrix_svg` and `pulsatrix_bench` tools. A backend's headers ship only with that
  backend. The viz module and the Python bindings aren't installed. The project now has a version
  number, 1.0.0. A test installs the build and compiles every installed header from a separate
  project; on HIP it caught hip's package config overwriting the one variable pulsatrix's config
  needed.
- **KS-2.** Comparing a build with itself flagged sub-millisecond CPU benchmarks 14 to 20%
  slower: separate processes of one binary land at one of two speeds on this machine, even
  pinned to a core. So a time regresses only when its median slows past 10% *and* a one-sided
  Mann-Whitney U test over at least three rounds per side says the rounds separate.
  `scripts/bench_ab.sh` runs the builds in ABBA order. Restoring HIP-4's syncs was flagged on all
  three training benchmarks at p = 0.014. The conservation gate is 10⁻³, the per-layer tolerance
  the Conv2D LRP tests accept, and runs in CI on the CPU.

### Follow-ups the KS work surfaced

| Follow-up | Found in | Belongs with |
|---|---|---|
| Install `pulsatrix_viz`, which needs the fetched ImGui, ImPlot and GLFW installed too | KS-1 | NB or KS-11 |
| Run `pulsatrix_bench` and its A/B comparison on the GPU in CI | KS-2 | KS-8 |
| The CUDA path of `pulsatrix_bench` has never run; there is no NVIDIA GPU here | KS-2 | KS-8 |
| Benchmarks on real models (ResNet18, SmolLM2) next to the fixed small ones | KS-2 | KS-9 |

## PLM: Protein language models

Protein language models are transformers trained on amino-acid sequences by filling in hidden
residues. Along the way they learn which positions tolerate change, which residues touch in the
fold and which motifs mark functional sites. The epic builds for ESM-2 first: it is MIT-licensed,
the most studied, 8M to 650M parameters (fits in fp32 here), and a BERT-style encoder that
pulsatrix mostly has. Pretraining at ESM-2's scale (about 10²¹ FLOPs) isn't realistic on one
workstation, so the epic loads the published checkpoints, explains and evaluates them, and
fine-tunes small heads; small models are trained from scratch only to check the training code.
Every item is checked against published numbers. The full research, the views and the data
sources are in [Protein language models: research and plan](protein-language-models.md).

| ID | Item | Why | Depends on | P | Effort | Status |
|---|---|---|---|---|---|---|
| PLM-1 | Encoder block options: LayerNorm or RMSNorm, a plain or gated MLP, exact (erf) GELU with its LRP rule, and attention and MLP biases | Every BERT-style encoder needs it, not just ESM | — | P0 | M | Done, [#94](https://github.com/Joshuaweg/pulsatrix/pull/94) (see below) |
| PLM-2 | `EncoderLM` and ESM-2 loading: token-dropout scaling, rotate-half RoPE, final LayerNorm, LM head; `EsmForMaskedLM` configs and weights; ESM `vocab.txt` tokenizer and a FASTA reader. Golden parity of logits, hidden states and attentions with `transformers` on a tiny generated ESM and on `esm2_t6_8M` and `esm2_t33_650M` | The foundation; must match the reference before anything else | PLM-1 | P0 | M | Done, [#95](https://github.com/Joshuaweg/pulsatrix/pull/95) (see below) |
| PLM-3 | Variant scoring: masked-marginal, wild-type-marginal and pseudo-log-likelihood scores, and full single-mutant scans. A ProteinGym runner (Spearman, NDCG, top-10% recall) that matches published ESM-2 numbers per assay on a subset | The headline use and the strongest numerical check | PLM-2 | P0 | M | Done, [#96](https://github.com/Joshuaweg/pulsatrix/pull/96) (see below) |
| PLM-4 | Contacts: ESM's contact head (symmetrize, APC, logistic regression) at parity with `transformers`, the top-K head average, a PDB/mmCIF reader, and precision at L, L/2 and L/5 by sequence separation | Shows whether the model learned the fold; feeds the head grid | PLM-2 | P1 | M | Done, [#97](https://github.com/Joshuaweg/pulsatrix/pull/97) (see below) |
| PLM-5 | Protein views: the mutation map (L × 20), sequence logos, contact maps (predicted and true triangles), residue tracks, and a 3D structure page (Mol* or 3Dmol.js, CDN or inline) colored by any per-residue score | The views biologists read; reuses VIZ-1 to VIZ-3 | PLM-3, PLM-4 | P1 | L | Done, [#98](https://github.com/Joshuaweg/pulsatrix/pull/98) (see below) |
| PLM-6 | Explaining encoders: AttnLRP to a masked position, a mutation's log-odds or a fine-tuned head; per-residue relevance; attribution graphs over residues (VIZ-4); a sanity suite (randomized weights, deletion curves, agreement with DMS sensitivity and conservation) | Residue explanations that are checked, not just drawn | PLM-2, PLM-5 | P1 | M | Done, [#99](https://github.com/Joshuaweg/pulsatrix/pull/99) (see below) |
| PLM-7 | Masked-LM training: the 15% (80/10/10) masking collator with token dropout, cropping and cluster-weighted sampling; an ESM-2-8M-shaped model on a UniRef50 sample, checked against BioNeMo's curve; per-residue and per-protein fine-tuning heads | Training and transfer learning; full pretraining waits on HIP-11 | PLM-2 | P1 | M | Done, [#100](https://github.com/Joshuaweg/pulsatrix/pull/100) (see below) |
| PLM-8 | Probes and features: per-layer linear probes (DSSP, accessibility, binding sites) with control tasks; an InterPLM-style SAE on ESM-2-8M with features matched to Swiss-Prot annotations; a feature dashboard with a structure panel | Concept-level interpretability | PLM-5, FEAT-1 | P2 | L | Done, [#103](https://github.com/Joshuaweg/pulsatrix/pull/103) (see below) |
| PLM-9 | A tutorial on one protein (TEM-1 β-lactamase): scan every mutation against its DMS, draw the contact map against its structure, and show relevance on the 3D structure | The whole path, end to end | PLM-3 to PLM-6 | P1 | S | Done, [#101](https://github.com/Joshuaweg/pulsatrix/pull/101) (see below) |

### How the PLM work departed from the plan

- **PLM-1** added `EncoderBlock` as a new class instead of options on `TransformerBlock`.
  `TransformerBlock` keeps the decoder path (KV cache, Gemma's sandwich norms) and its accessors'
  types. `EncoderBlock` covers the encoder layouts: pre- or post-norm, LayerNorm or RMSNorm, and a
  plain or gated MLP. With pre-RMSNorm and a gated MLP it reproduces `TransformerBlock` exactly,
  which a test checks.
  - Also new: `ActivationModule` (any pointwise activation), `FeedForwardModule`, and exact GELU
    as a backend op (`ElementwiseOp::Gelu`, CPU and GPU kernels, plus `GatedActivation::Gelu`).
  - Checked:
    - ESM-2's and BERT's layers match `transformers` (`EsmLayer`, `BertLayer`) to 2e-5 in output and input gradient.
    - Parameter gradients match finite differences in both layouts.
    - LRP is linear in both layouts.
    - The new GPU cases pass on gfx1151, and so does the full HIP suite (2663 tests).
  - Post-norm relevance can get very large with random weights, because a residual split's
    `a + b` can be near zero. That's the epsilon rule's known behavior, not a defect; trained
    weights keep the sums well away from zero.

- **PLM-2** matches transformers on ESM-2 8M and 650M to float32 precision: logits, every hidden
  state and every attention map, with worst differences of 3.7e-6 relative and 1.3e-5 absolute.
  Getting there took three findings:
  - **Stored RoPE frequencies.** The checkpoints store `inv_freq` rounded to fp16, and
    transformers uses the stored values. With exactly computed frequencies, attention differed
    by 2e-4: 100 times float noise, found by rebuilding layer 0 in float64. `LoadEncoderLM`
    now reads the stored frequencies.
  - **New names in transformers 5.** It saves `LayerNorm` parameters as `gamma`/`beta`, and the
    layers' shared `inv_freq` once under a wildcard name
    (`esm.encoder.layer.*.attention...`). The loader accepts both.
  - **Tokenizer splitting.** `EsmTokenizer` matches every vocabulary token anywhere, then
    makes each leftover run one `<unk>`. The tokenizer reproduces this with added tokens, a
    whitespace split and a word-level lookup, and matches on 12 edge cases.
  - Also: `EncoderLM` on GPU matches the CPU, with padding and masks. Hidden states and
    attention maps are exposed for probes, SAEs and contacts. A FASTA reader was added.

- **PLM-3** matches ProteinGym's published ESM-2 650M results on 14 assays: Spearman to the
  three decimals published, and on 13 of them every variant's score within 9e-5. The metrics
  match scipy, scikit-learn and ProteinGym's own NDCG and top-recall code.
  - **Memory.** The first full run was killed by the out-of-memory killer at 26 GB on the host.
    Every module keeps its forward activations for `backward()` and LRP, and attention keeps
    two `(N, H, L, L)` tensors per layer: over 10 GB per 1024-token sequence for 650M.
    `Module::release_activations()` (all modules on the encoder path) and
    `EncoderLM::set_keep_activations(false)` now let a forward hold one layer's activations at a
    time, with identical outputs. `VariantScorer` uses them, and sizes each batch by a memory
    budget (`max_pass_bytes`), refusing a pass that can't fit. BRCA1 (1863 residues) at batch 8
    then peaks at the model's own 10.2 GB on CPU and 16 GB on GPU.
  - Not done here, now backlog items: the 10 GB it takes to load 650M (FND-9), and the speed of
    1024-token passes on gfx1151, about 3.7 s each (HIP-13).

- **PLM-4** matches transformers' `predict_contacts` within 3.1e-5 on ESM-2 8M and 650M, on CPU
  and GPU, for ten proteins with experimental structures. The structure reader matches biotite
  on 17 chains in both formats. The precision equals ESM's `compute_precisions` on every case.
  - **Streamed attention.** `EncoderLM::set_attention_observer()` hands each layer's attention
    to a callback during `forward()`. `ContactPredictor` folds it into the result there, so a
    pass holds one layer's maps instead of all 660 of 650M's heads.
  - **The top-K average needs no regression.** On five proteins, the ten heads ranked best on
    five others score a mean long-range P@L of 0.642, against 0.637 for ESM's trained head.
    All ten are in layers 22 to 32.
  - **Our own reader, not a dependency.** PDB and mmCIF readers came out small enough to own
    (first model, `ATOM` plus selenomethionine, residues with a Cα, the first alternate
    location). Author chain ids and numbering in both formats make the two agree.
  - **Two choices in precision.** Pairs with an unknown distance are dropped before ranking;
    ESM's code ranks them with a score of minus infinity. Ties keep row order. On real
    structures the two agree exactly.
  - Not done: the full SEQRES sequence. The model sees the observed residues, so a chain with
    unresolved loops is scored on a shortened sequence. AlphaFold DB models have no gaps.

- **PLM-5** adds four viz documents (`mutation_map`, `sequence_logo`, `contact_map`,
  `residue_tracks`), with builders from the model's outputs, SVG figures, HTML pages and a 3D
  structure page. `tools/plm/pulsatrix_protein_views` makes all of them for one protein.
  - **Pages hold the SVG, not Vega-Lite.** A contact map has L² cells (69,000 for TEM-1) and a
    logo needs stretched letters, neither of which Vega-Lite handles well. So each page embeds
    the figure, with a tooltip on every cell, and needs no scripts. Only the structure page loads
    a library.
  - **3Dmol.js, not Mol\*.** 3Dmol.js is one 540 KB file (BSD-3-Clause), easy to pin, hash and
    inline. Mol\* is several megabytes. 3Dmol.js reads author chain ids and numbering from mmCIF,
    as the PLM-4 reader does, so residue ids line up. It doesn't read insertion codes from mmCIF;
    PDB files carry them.
  - **Logo letters are stretched text.** Each letter is bold sans-serif text scaled to its box, so
    a figure needs no glyph outlines. A letter's ink fills its box to within a few percent,
    depending on the installed font (Helvetica, Arial or Liberation Sans).
  - Checked: every figure, by eye, in Chromium and cairosvg, on fixtures (now golden files) and on
    ESM-2 650M's output for TEM-1 (1BTL). The contact map's long-range precision there is 0.665,
    the same value PLM-4 measured. The structure page was driven in headless Chromium: the menu
    switches tracks, and hovering labels a residue with its value.
  - Figures number residues consecutively from `first_position`. A structure whose author
    numbering skips numbers (TEM-1's Ambler numbering skips 239 and 253) is numbered by position
    in the figures, by author id on the structure page.

- **PLM-6** explains ESM-2 with AttnLRP from four targets: a masked residue's logit, a mutation's
  log-odds, and linear heads per protein and per residue. It matches LXT's rules on transformers'
  ESM-2 to 4.8e-5 on 650M, for every token and layer. Residue relevance graphs export to
  Neuronpedia's viewer.
  - **LayerNorm's rule was wrong for AttnLRP.** `LayerNormModule` passed relevance straight
    through, which equals LXT's rule for RMSNorm but not for LayerNorm: LXT holds only the
    standard deviation constant, so the mean subtraction and the bias stay in the explanation.
    The first comparison missed by a factor of 4,000. `LRPRuleConfig::layer_norm_detach_std` adds
    LXT's rule, and `LxtAttnLrpConfig()` turns it on. The default is unchanged, so earlier
    results stand. LLM-7's models use RMSNorm, so their parity was never affected.
  - **No LXT patch for ESM.** LXT 2.1 patches BERT but not ESM, so the reference applies its
    rules to ESM the way the BERT patch does: `make_esm_attnlrp_reference.py`, with a
    plain-gradient negative control.
  - **The checks found what they're for.**
    - Deletion confirms faithfulness on ubiquitin and TEM-1.
    - TEM-1's explanation keeps most of its ranking (0.83) while the top 14 layers are
      randomized: it mostly reflects the lower layers.
    - The relevance profile barely agrees with DMS sensitivity or conservation (Spearman
      0.03-0.10), where those two agree with each other (0.29-0.50).
  - **Two fixes along the way.**
    - Deletion orders residues by support for the value's sign. Otherwise a negative log-odds
      is "deleted" in the wrong direction.
    - Randomization redraws weight matrices only. The shared routine redrew LayerNorm gains from
      their own small spread, which made a pre-LN block nearly a residual connection and hid the
      randomization (similarity stayed at 0.7-0.8).

- **PLM-7** trains ESM-2-shaped models. It has the masking collator, cluster sampling, ESM's
  initialization, a training step, evaluation, and heads per protein and per residue. One step's
  loss and every gradient match transformers, and three AdamW steps match `torch.optim.AdamW`.
  GPU matches CPU.
  - **BioNeMo has no 8M curve.** Its recipe page gives only validation perplexities for 650M and
    3B (7.00, 6.00). The check became the published checkpoints' perplexity on our own UniRef50
    sample: 8M 10.89 (paper 10.33), 650M 5.96 (paper 6.95). The sample isn't held out from
    ESM-2's training data, which likely explains the low 650M figure.
  - **From scratch,** an 8M-shaped model reached perplexity 14.1 in 4,000 steps, against 18.2
    for amino-acid frequencies. It took 3.7 hours, which is HIP-13's speed problem again.
  - **UniRef50 sampling.** UniProt's FASTA is sorted longest first, and the REST stream by
    accession, which clumps organisms. `fetch_uniref50_sample.py` takes up to 150
    representatives from each of about 960 accession prefixes instead.
  - **Heads:**
    - A per-residue burial probe reaches Spearman 0.68 (8M) and 0.82 (650M) on held-out proteins.
    - On ubiquitin's DMS, fine-tuning beats the probe and the zero-shot score (0.56, 0.36, 0.16).
    - On TEM-1's, neither mean-pooled head beats zero-shot (0.27-0.28 against 0.43): one
      substitution barely moves the mean of 286 residues' representations.
  - Not done: training in bf16 (HIP-11) and token-budget batches. Batches here are fixed
    counts of cropped sequences.

- **PLM-9** is a recipe, `protein_tem1_recipe`, with a walkthrough page.
  - **Results with 650M** (82 s on gfx1151):
    - the scan's Spearman is 0.7315 (ProteinGym's 0.731);
    - contacts against 1BTL reach P@L 0.665;
    - the most damaging mutation, K73S on the catalytic lysine, is explained first by S70, the
      catalytic serine, and the deletion check calls it faithful.
  - **The crystal isn't the assay's protein.** 1BTL differs from the scanned TEM-1 at V84I and
    A184V, so the recipe places the structure by the ungapped alignment with the most identical
    residues and reports the differences.
  - **Two numberings.** The scan numbers the precursor from 1 and the structure uses Ambler's
    numbering; residue ids keep them apart.

- **PLM-8** (after FEAT-1, #102) adds the probes, SAE features and dashboards, in
  `protein_concepts.hpp` and `pulsatrix_probe_esm`.
  - **Labels.** Swiss-Prot annotations stand in for DSSP and accessibility: 3,000 random
    reviewed proteins that have 3D structures, so helices and strands come from structures,
    plus binding and active sites, disulfides, signal peptides, transmembrane spans, zinc
    fingers, coiled coils and motifs. Accessibility isn't among UniProt's features and wasn't
    computed.
  - **The control task became a sequence control.** Hewitt and Liang's control task (labels
    fixed per token type) gave balanced accuracy 1.0 at every layer. Twenty amino acids are
    trivially separable, so a linear probe learns it from any representation. The control is
    a linear probe on one-hot local sequence (±3) instead, beside a randomly initialized model.
    - With it, disulfide probes show no gain at all: they find cysteines.
    - Secondary structure gains +0.20 (8M) and +0.25 (650M) balanced accuracy.
  - **The SAE.** 2,560 latents on ESM-2 8M layer 4: held-out L0 17, 66% of the variance
    explained, no dead latents. Inputs are standardized per dimension; unstandardized, a few
    huge dimensions dominated and it explained 33%.
    - Single features beat single neurons on sparse concepts: disulfides 0.69 against 0.18,
      zinc fingers 0.29 against 0.05, signal peptides 0.44 against 0.34.
    - Neurons win on helices and strands.
    - A random model's SAE matches disulfides (0.39) and coiled coils (0.26) too, so matches
      are leads, not proof.
  - **Dashboards** carry a structure panel (`StructurePanelHtml`, factored out of the structure
    page). Protein examples wrap residue by residue.

### Follow-ups the PLM work surfaced

| Follow-up | Found in | Belongs with |
|---|---|---|
| Loading ESM-2 650M takes about 10 GB on the host | PLM-3 | FND-9 |
| 1024-token passes take about 3.7 s on gfx1151, and an 8M training step 3.3 s | PLM-3, PLM-7 | HIP-13 |
| Structures are read as their observed residues: a chain with unresolved loops is scored on a shortened sequence. Reading SEQRES and `_pdbx_poly_seq_scheme` would keep the gaps | PLM-4 | A follow-up to PLM-4 |
| 3Dmol.js doesn't read insertion codes from mmCIF, so the structure page matches them in PDB files only | PLM-5 | PLM-5 follow-up, or a 3Dmol.js release that reads them |
| TEM-1's K73S explanation keeps most of its residue ranking while the top 14 of 33 layers are randomized | PLM-6 | Research: which residue explanations depend on the whole model |
| Training needs bf16 and token-budget batches to scale past toy runs | PLM-7 | HIP-11 |
| Mean-pooled heads don't beat zero-shot on single mutants of long proteins | PLM-7 | A per-residue or difference head, if wanted |
| Solvent-accessibility labels (computed from structures), and probe splits that keep homologs apart. Also, Hewitt and Liang's control task is trivial with few token types: on 20 amino acids a linear probe learns it perfectly, so INT-3 should offer a sequence-window control | PLM-8 | INT-3 (probe controls) |

Later, if wanted: ESM C and AMPLIFY weight mappings, SaProt's structure tokens, autoregressive
pLMs (ProGen2, through CausalLM), MSA-conditioned models and ESMFold.

## Open questions

These came up in the research and aren't settled yet.

- **BSF details.** The exact MDL formula and the block-size defaults beyond the published example
  need a full read of the papers.
- **Missing LRP rules.** No published rule exists for relevance through a block top-k (BSF),
  through a LoReFT intervention, or through LoRA under the gamma rule. The epsilon-rule split for
  LoRA is our own derivation and needs a numerical check.
- **Qwen3 in LXT.** LXT marks Qwen3 as experimental, with relevance skewed toward the first
  token. The cause isn't documented. pulsatrix matches LXT on Qwen3-0.6B to 1.2e-5 (LLM-7), so any
  skew comes from the rules themselves, not from either implementation.
- **hipBLASLt on gfx1151.** Sources disagree on whether it works on ROCm 7.2.4. HIP-8 settles it.
- **rsLoRA scaling.** It conflicts with the claim that 1/r scaling makes the learning rate
  independent of rank. Only an experiment settles it.
- **Licenses.** Llama 3.2's license terms for redistributing converted weights: ship converters,
  not weights. The licenses of PHAT, Dionysus, Hera and Eirene weren't checked; none are needed
  for the plan.
- **Mamba-2 and xLSTM.** No published LRP rules found.
- **MCP security statistics.** The figures cited come from secondary reports, not primary scans.
- **xeus-cpp.** How version 0.10 loads prebuilt libraries, and whether prebuilt HIP kernels can
  be called from notebook cells, needs a hands-on test.
- **Windows TLS.** Whether the open cpp-httplib Windows certificate issue affects
  `api.anthropic.com` needs a CI probe.
- **TDA numerics.** The fp32 distance error between near-duplicate activations hasn't been
  measured. Don't trust loops smaller than that error.
