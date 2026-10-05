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
  embedding table (LLM-2).
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
- CFS-1 to CFS-7 (added 2026-10-05, built first): ICE, ALE, local and global sensitivity, and
  counterfactuals, each with its view
- TOK-1 to TOK-4: the tokenizer interface with offsets, byte-level BPE (SmolLM2, Qwen, Llama),
  SentencePiece-style BPE (Gemma 3), and word-level aggregation of token scores
- HIP-6, HIP-7: fused kernels, bounded-memory Conv2D (HIP-3 landed early, in v1.1)
- VIZ-3, VIZ-6a: Vega-Lite HTML and the token relevance view
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

- FEAT-1 to FEAT-7: the featurizer interface, TopK and its family, metrics, transcoders, BSF,
  steering
- TDA-9, TDA-10, TDA-12: topology views, the manifold verifier for featurizers, layer-wise
  topology
- VIZ-4, VIZ-6b, VIZ-6c: Neuronpedia export, feature dashboards, the embedding projector
- AGT-5, AGT-8: the native MCP server and automated feature descriptions
- NB-3: Python notebook display
- LLM-8, LLM-9: tuned lens, AtP*, Gemma 3
- KS-5: TracIn data attribution
- IO-7: `.npy`/`.npz`, for published SAE dictionaries
- ARCH-1 to ARCH-3: B-cos layers, concept bottleneck models, mixture of experts
- HIP-8: hipBLASLt and bf16 GEMM

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
| IO-4 | A name-mapping manifest with transforms: transpose (pulsatrix `Linear` stores `(in, out)`, PyTorch stores `(out, in)`), RoPE layout permutation, splitting fused QKV, weight tying. Strict mode fails on unmapped or extra keys | Turns Hugging Face names and layouts into pulsatrix modules | FND-1, IO-1 | P1 | M | |
| IO-5 | Read Hugging Face `config.json` and sharded `model.safetensors.index.json` | Every small LLM on the Hub uses these | IO-1 | P1 | S | |
| IO-6 | Upcast bf16 and fp16 weights to fp32 on load (exact) | Most published weights are bf16 | IO-1 | P1 | S | |
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

### Follow-ups the IO work surfaced

| Follow-up | Belongs with |
|---|---|
| Memory-mapped reading; files are read whole for now | IO-5, for 1B-parameter models |
| Checkpointing SGD's momentum buffers (Adam and AdamW are covered) | a small fix |
| Checkpointing Dropout's mask counter; a resumed run with active dropout draws different masks | a small fix |
| Running the safetensors writer's GPU path on hardware | HIP-9 or KS-8 |

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
| LLM-3 | Native tokenizers for the target models: [TOK-1 to TOK-3](#tok-tokenizers) | No Python needed to tokenize | — | P0 | M–L | |
| LLM-4 | Generation: greedy, temperature, top-k and top-p sampling, seeded, with EOS handling | Run the model, not just score it | LLM-1, FND-3 | P1 | S | |
| LLM-5 | A preallocated KV cache | Generation without recomputing the whole prefix | LLM-1 | P1 | M | |
| LLM-6 | Golden-logit harness: compare against Hugging Face on real text, fp32, maximum absolute difference under 1e-3 | Catches layout bugs that still "load" | IO-4 | P0 | S | |
| LLM-7 | AttnLRP parity with LXT on SmolLM2 (per-token relevance correlation above 0.99), including the relevance split across shared key and value heads | The headline result: pulsatrix explains a real LLM and matches the reference | LLM-1, LLM-2, TOK-2, LLM-6 | P1 | M | |
| LLM-8 | Tuned lens and AtP* (corrected attribution patching) | Better versions of the logit lens and patching that already exist | LLM-1 | P2 | S–M | |
| LLM-9 | Gemma 3 support: sliding-window attention, `(1 + w)` RMSNorm, embedding scaling | Unlocks Gemma Scope 2 dictionaries | LLM-1 | P3 | M | |


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

### Follow-ups the LLM work surfaced

| Follow-up | Belongs with |
|---|---|
| Llama 3's `rope_scaling` changes the low RoPE frequencies at every position, so Llama-3.2 logits need it even on short text | LLM-6 |
| `SwiGLUModule` without biases: Llama-family MLPs have none, and IO-4's strict mode fails on unmapped keys | IO-4 |
| Gemma 3's `query_pre_attn_scalar` replaces the `1/sqrt(head_dim)` scale | LLM-9 |
| Grouped-query attention materializes the repeated K and V; index the shared head inside the matmul instead | HIP-6 |

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
| TOK-1 | A `Tokenizer` interface: `encode` returns ids, token strings, character offsets and a special-token mask; `decode` returns text. Added tokens (special tokens such as `<\|im_start\|>`) are split out before anything else runs. The current tokenizer becomes the `WordLevel` model with a whitespace pre-tokenizer, and a byte tokenizer (256 ids plus specials) and a character tokenizer join it | One interface for every model, and offsets for every explanation | — | P0 | S | |
| TOK-2 | Byte-level BPE from `tokenizer.json`, with a small regex engine for the Split pre-tokenizer: alternation, character classes with Unicode categories (`\p{L}`, `\p{Lu}`, `\p{M}`, `\p{N}`…) over a generated category table, ranges, quantifiers, `(?i:…)` and the `(?!\S)` lookahead; `std::regex` can't match Unicode categories. Also digit splitting, an NFC normalizer from generated Unicode tables, both merge formats (`"a b"` strings and `["a", "b"]` pairs), `ignore_merges`, and template post-processing (BOS and EOS). It passes when ids match Hugging Face on a 10,000-line multilingual corpus for SmolLM2, Qwen2.5, Llama 3.2 and gpt-oss | Any byte-level BPE model loads without new code: SmolLM2, Qwen, Llama, gpt-oss, Mistral, DeepSeek, Phi-4 | TOK-1 | P0 | M | |
| TOK-3 | SentencePiece-style BPE: byte fallback (`<0x41>` tokens), `▁` replacement, `fuse_unk`, and the matching decoder chain. It passes the same corpus test for Gemma 3 | Gemma 3 and Gemma Scope 2 (LLM-9, FEAT) | TOK-2 | P1 | S | |
| TOK-4 | Word-level aggregation: merge per-token scores (relevance, attributions, probe outputs) into per-word scores using the offsets, by sum, mean or maximum | Explanations people can read | TOK-1 | P1 | S | |
| TOK-5 | WordPiece (BERT normalizer and pre-tokenizer, `##` continuation) | BERT-family encoders, the most common models in XAI papers and tutorials | TOK-1 | P2 | S | |
| TOK-6 | A byte-level BPE trainer that writes `tokenizer.json`, so a model trained from scratch in pulsatrix gets a real subword vocabulary that Hugging Face can also load | Small models trained on your own corpus | TOK-2 | P2 | M | |
| TOK-7 | Unigram (SentencePiece) with the precompiled charsmap normalizer | T5, ALBERT, XLNet and mBART | TOK-1 | P3 | M | |
| TOK-8 | A `.tiktoken` loader (base64 token and rank per line; merge priority is the rank of the merged token) with OpenAI's `cl100k_base` and `o200k_base` patterns, checked against `tiktoken` | Count and inspect tokens exactly as GPT models see them, with no Python | TOK-2 | P2 | S | |

Not planned: SentencePiece `.model` protobuf files (every target model also ships
`tokenizer.json`) and a local Claude tokenizer (not public).

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
| What if | PDP, ICE, ALE, local sensitivity, occlusion, re-running with an edited input; global sensitivity | Partial; [CFS-4](#cfs-counterfactuals-and-sensitivity) |
| How (global) | Global surrogate tree, PDP, TCAV, CRP | Partial; XAI-4, INT-6 |
| How to be that | Counterfactual search | Gradient-based; [CFS-6, CFS-7](#cfs-counterfactuals-and-sensitivity) for any model and diverse sets |
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
| CFS-4 | Global sensitivity: Morris elementary effects (μ\*, σ) and Sobol first-order and total indices (Saltelli sampling, Jansen estimators) with bootstrap confidence intervals. Views: μ\*–σ scatter, Sobol bars with error bars. Checked against SALib | Which inputs drive the output over the whole input space, and which interact | — | P1 | M | |
| CFS-5 | Gradient counterfactual (Wachter): the nearest input that reaches a target class or value, distance weighted by each feature's median absolute deviation, with immutable features, bounds, and integer or categorical features. Reports validity, proximity (L1, L2) and sparsity. View: what changed, feature by feature | "How to be that", the most requested missing explainer | — | P0 | M || Done, [#71](https://github.com/Joshuaweg/pulsatrix/pull/71) (see below) |
| CFS-6 | Model-agnostic counterfactual: growing spheres, refined by the evolutionary module, for models with no gradient | Counterfactuals for any prediction function | CFS-5 | P1 | S | |
| CFS-7 | Diverse counterfactuals (DiCE: a determinantal diversity term) and a plausibility score (distance to the k nearest background instances). Checked against DiCE's own metrics | One counterfactual hides the other ways to change the outcome; implausible ones mislead | CFS-5 | P1 | M | |

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
- **CFS-5** uses proximal gradient descent (a gradient step on the prediction hinge, then
  soft-thresholding toward the input) rather than plain gradient descent on the L1 distance,
  so unneeded features stay exactly unchanged; on a linear model it finds the L1-optimal
  counterfactual, which the tests check in closed form. The prediction loss is a hinge on the
  outputs (Carlini and Wagner's form) rather than Wachter's squared error, so it is zero once the
  target is reached. Snapping a one-hot group can undo a change the search made only partway, so
  if it does, each category of each group is tried and the nearest valid one kept.

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

| ID | Item | Why | Depends on | P | Effort |
|---|---|---|---|---|---|
| FEAT-1 | A `Featurizer` interface (encode, decode, loss, decoder normalization), with the existing SAE ported to it and given unit-norm decoders, an L0 metric and dead-latent tracking | One interface for the whole family | — | P0 | M |
| FEAT-2 | TopK SAE with the auxiliary loss for dead latents | Removes L1 shrinkage; sets sparsity directly | FEAT-1, FND-3 | P0 | M |
| FEAT-3 | Core SAEBench metrics: explained variance, loss recovered when the reconstruction is spliced back into the model, dead and dense latents, feature absorption. Random-model and probe baselines on by default | Without these, no featurizer result can be trusted | FEAT-1, XAI-6 | P0 | M |
| FEAT-4 | BatchTopK, JumpReLU and Matryoshka SAEs | Fix specific failures of TopK | FEAT-2 | P1 | — |
| FEAT-5 | Transcoders and skip transcoders | Reported to be more interpretable than SAEs | FEAT-1 | P1 | M |
| FEAT-6 | BSF: the vanilla, Grassmannian and group-lasso variants, then tournament top-k, with MDL and stable-rank metrics | The newest member of the family | FEAT-2, FND-4 | P1 | L |
| FEAT-7 | Steering with difference-of-means by default and featurizer directions as an option, with a reliability report | The simple baseline usually wins | FEAT-1 | P1 | S |
| FEAT-8 | Crosscoders, including the Delta-Crosscoder for fine-tuning diffs | Model diffing across layers and models | FEAT-1, IO-2 | P2 | — |
| FEAT-9 | Parameter decomposition (SPD and VPD) | Interpretability in weight space; few libraries have it | FND-4 | P2 | L–XL |
| FEAT-10 | Attribution graphs with cross-layer transcoders | The existing circuit graph plus LRP may cover most of the value first | FEAT-5 | P2 | XL |

Open: no LRP rule exists yet for propagating relevance through a block top-k. FEAT-6 needs one.

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
| VIZ-3 | Self-contained Vega-Lite HTML, with the JavaScript inlined or loaded from a CDN | Hover, zoom and export for free, in a browser or notebook | VIZ-1 | P1 | S | |
| VIZ-4 | Export circuit graphs in the attribution-graph schema that Neuronpedia and circuit-tracer read | Large graphs get a mature viewer for free | VIZ-1 | P1 | M | |
| VIZ-6 | New views: (a) token relevance for text, (b) feature dashboards, (c) an embedding projector, (d) an attention head grid, (e) SHAP force, decision and dependence plots | Fill the gaps between the current widgets and the reference tools | VIZ-1 | P2 | M each | (a) in SVG only, through VIZ-2 |
| VIZ-7 | A node editor for circuit graphs | Only if graphs outgrow the current view; VIZ-4 covers large ones | — | P3 | M | |

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

### Follow-ups the VIZ work surfaced

| Follow-up | Found in | Belongs with |
|---|---|---|
| SVG views for circuit graphs, training logs and feature dashboards | VIZ-2 | VIZ-4, VIZ-6 |
| Nothing produces `feature_dashboard.v1` yet; its fields follow SAEDashboard | VIZ-1 | FEAT |
| Byte-level BPE tokens must be decoded to UTF-8 before they go into a document | VIZ-1 | TOK-2 |
| SVG text widths are estimated (0.6 em), so very wide scripts such as CJK can overflow labels | VIZ-2 | VIZ-3 (the browser lays out text) |

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

## Open questions

These came up in the research and aren't settled yet.

- **BSF details.** The exact MDL formula and the block-size defaults beyond the published example
  need a full read of the papers.
- **Missing LRP rules.** No published rule exists for relevance through a block top-k (BSF),
  through a LoReFT intervention, or through LoRA under the gamma rule. The epsilon-rule split for
  LoRA is our own derivation and needs a numerical check.
- **Qwen3 in LXT.** LXT marks Qwen3 as experimental, with relevance skewed toward the first
  token. The cause isn't documented.
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
