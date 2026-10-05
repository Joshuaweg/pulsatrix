# Roadmap Research Notes

This page holds the evidence behind the [Roadmap](index.md). The research was done on
2026-10-03. For each item it gives:

- **Source:** the primary source and its date.
- **Falsifier:** the result that would tell us not to build the item, or that it doesn't work.
- **Fails when:** known failure modes from papers, issue trackers and post-mortems.

Some claims about AMD's gfx1151 rest on community reports rather than AMD documentation, so they
carry tags:

- **[V]**: verified in a primary source (vendor docs, a spec, a paper).
- **[C]**: a community report (a GitHub issue or benchmark). Real, but not reproduced by us.
- **[I]**: our own inference. Treat every speedup estimate as [I] until HIP-1 measures it.

Sources and tools age quickly in this field. Check the date before relying on one.

## Cross-cutting

- **Random networks give plausible explanations.**
  - Méloux et al., "The Dead Salmons of AI Interpretability", arXiv 2512.18792 (2025-12): attribution, probing, sparse autoencoders and causal analyses all produce plausible output on randomly initialized networks.
  - Adebayo et al., "Sanity Checks for Saliency Maps", NeurIPS 2018: some saliency methods barely change when the model's weights are randomized.
  - Heap et al., arXiv 2501.17727 (2025-01): SAEs on randomly initialized transformers produce features that look interpretable.

  This is the basis for the baseline rule and for XAI-6.
- **Explainers disagree with each other.** Krishna et al., "The Disagreement Problem in Explainable Machine Learning", arXiv 2202.01602 (2022-02).
- **Surrogates can be fooled.** Slack et al., AIES 2020: a model can be built to look innocent to LIME and SHAP while staying biased.

## FND: Foundations

All FND items are done; see the roadmap's FND section for what changed from the plan. The notes
below record the reasoning they were built on.

- **FND-1, FND-2.** `Module::parameters()` returned `std::vector<ParamRef>` with no names. It is now derived from `named_parameters()`.
  - **Falsifier for freezing:** a frozen parameter's bytes change after N steps, or the input gradient no longer matches finite differences.
- **FND-3, FND-4.** The demand for these comes from items in other epics. FEAT-2 (Gao et al., TopK SAE, arXiv 2406.04093, 2024-06) and ARCH-3 need top-k. INT-5, FEAT-6, TRN-9 (Wu et al., ReFT, arXiv 2404.03592, 2024-04) and TRN-11 (Shuttleworth et al., arXiv 2410.21228, 2024-10) need an eigensolver or SVD. `linear_algebra.hpp` had only a dense solve; `matrix_decompositions.hpp` now has the rest.
  - **Fails when:** eigenvector sign and order are arbitrary, so tests must align them before comparing.
- **FND-5.**
  - **Sources:** `plans/lrp_issues.md` #8. Zennit's canonizers fold BatchNorm into the preceding layer before LRP.
  - **Falsifier:** none. Explanations that depend on the rest of the batch are a correctness bug.
- **FND-6.** VGG and ResNet are the standard LRP benchmark models, and both need stride and padding.
- **FND-7.** `plans/gpu_review.md` lists "no atomics, so deterministic" as a current strength. Keep that.
- **FND-8.** `plans/gpu_review.md` #1 (no device check in `Module::forward`) and #2 (`Tensor::Stack` tags the wrong device).

## IO: Serialization and model import

- **IO-1.**
  - **Format:** huggingface/safetensors (Apache-2.0) is an 8-byte length, a JSON header, then a packed data section. The header is at most 100 MB, and `__metadata__` maps strings to strings.
  - **Audit:** Trail of Bits audited the format (published 2023-05).
  - **Option:** the header-only C++ reader safetensors-cpp (MIT) exists, but its README says shape validation and mmap are incomplete. Fuzz both it and a hand-written reader before choosing.
  - **Fails when:** an offset runs past the end of the file, ranges overlap, or the element-count product overflows.
- **IO-2.** **Falsifier:** save, then load, doesn't give a bit-identical forward pass, or resumed training doesn't reproduce the loss curve. Both checks pass (#44).
- **IO-3: pickle is code.**
  - CVE-2025-32434 (GHSA-53q9-r3pm-6pq6, 2025-04, CVSS 9.8): `torch.load(weights_only=True)` was remote code execution before torch 2.6.0.
  - Scanners get bypassed. "nullifAI" (ReversingLabs, 2025-02) used broken pickles and 7z-compressed `.pt` files; CVE-2025-46417 is another bypass; ShadowPickle (arXiv 2607.17503, 2026-07) shows evasion continues.
  - **Format:** `torch.save` writes a zip of `data.pkl` plus storage blobs, and tensors are rebuilt with explicit strides. Non-contiguous tensors must be made contiguous before writing.
- **IO-4: layout bugs load fine and give garbage.**
  - A missing Q/K RoPE permutation gives fluent nonsense (huggingface/transformers #25199). Random-weight tests can't catch it.
  - Llama 3.2's `rope_scaling` factor is 32, not 8; the wrong value only hurts past about 80k tokens (meta-llama/llama-models #241).
  - GPT-2's `c_attn` is a Conv1D stored `[in, out]`, the opposite of `nn.Linear`.
  - Gemma's RMSNorm multiplies by `(1 + w)`.
  - `rms_norm_eps` is 1e-5 for Llama and SmolLM2, 1e-6 for Qwen3 and Gemma 3.
- **IO-5.** These are Hugging Face `config.json` fields, fetched from the Hub on 2026-10-03:

  | Model | Notable config |
  |---|---|
  | SmolLM2-135M | d=576, 9 heads / 3 KV heads, tied, theta 1e5 |
  | Qwen2.5-0.5B | QKV bias, theta 1e6 |
  | Llama-3.2-1B | `rope_scaling` llama3, factor 32 |
  | Qwen3-0.6B | `head_dim` 128 ≠ d/h, QK-norm |
  | Gemma 3 270M | `head_dim` 256, sliding window 512 |

- **IO-6.** bf16 to fp32 is a 16-bit shift, so it's exact. fp16 to fp32 is also exact.
- **IO-7.** NumPy's format spec covers the `.npy` header. `allow_pickle` code execution has existed since CVE-2019-6446, so never accept the object dtype.
- **IO-8.** onnx/onnx (Apache-2.0). Protobuf caps a message at 2 GB, so larger models use external data (onnx #3273). External-data paths had a path-traversal CVE (CVE-2024-27318, fixed in onnx 1.16).
- **IO-9.** The format spec is `docs/gguf.md` in ggml-org/ggml (MIT). Its parsers have a long CVE history:
  - TALOS-2024-1913 and TALOS-2024-1914 (2024-03);
  - CVE-2025-53630 (integer overflow in `gguf_init_from_file_impl`);
  - CVE-2026-27940 (a heap overflow);
  - six more posted to oss-security on 2026-05-15.
- **IO-10.** Keras's `safe_mode` keeps being bypassed: CVE-2025-1550, then CVE-2025-8747 (which bypassed that fix), then CVE-2025-9905 (`.h5` loading ignores `safe_mode`).
- **IO-11.** TorchScript is deprecated as of PyTorch 2.10 (release blog); 2.11 recommends `torch.export`.

## TRN: Training and fine-tuning

- **Overall falsifier:** Shuttleworth et al. (arXiv 2601.22708, 2026-01) found that with a tuned learning rate, plain LoRA matches or beats most of its variants. If a variant doesn't beat tuned vanilla LoRA on our benchmark, don't ship it.
- **"LoRA Without Regret"** (Thinking Machines, 2025-09-29):
  - Attention-only LoRA underperforms; apply it to all layers.
  - The best learning rate is about 10× full fine-tuning's.
  - LoRA tolerates large batches worse.
  - Rank 1 is enough for RL.
- **"LoRA learns less and forgets less"** (Biderman et al., arXiv 2405.09673, 2024-05): LoRA is clearly worse than full fine-tuning on code and math, and it forgets less.
- **TRN-2.** Loshchilov and Hutter, "Decoupled Weight Decay Regularization", arXiv 1711.05101. **Test:** decay is applied as `w -= lr * wd * w`, not added to the gradient. The check passes (#47).
- **TRN-3, TRN-4.** Global-norm clipping and warmup-plus-cosine schedules are the defaults in every recipe cited above (for example "LoRA Without Regret"). Clip after accumulation and before the optimizer step. Done (#48, #49).
- **TRN-6.** Full fine-tuning is the baseline in Biderman et al. and in "LoRA Without Regret". Adam in fp32 needs about 16 bytes per parameter plus activations: about 2.2 GB for 135M parameters, about 16 GB for 1B.
- **TRN-5.** Hugging Face Trainer fixed this gradient-accumulation normalization bug in v4.46 (2024-10). **Falsifier:** 4×8 accumulated doesn't equal 1×32 on ragged batches. The check passes (#50).
- **TRN-7.** Hu et al., "LoRA", arXiv 2106.09685 (2021-06).
  - Merging into a quantized base changes the outputs (huggingface/peft #2321), so merge in fp32.
  - **Falsifier:** merged and unmerged outputs differ by more than 1e-5.
- **TRN-8.** Under the epsilon rule, LRP is linear in the contributions, so relevance splits exactly between W₀x and the adapter term. This is our own derivation and needs a numerical check. Gamma and z+ act on the positive part of the summed weight, so merged and unmerged models give different maps.
  - **Related:** Soligo et al., arXiv 2506.11618 (2025-06), read rank-1 LoRA adapters as interpretable directions.
- **TRN-9.** Wu et al., "ReFT", arXiv 2404.03592 (2024-04).
  - It fails on arithmetic (GSM8K).
  - It fails badly at concept suppression compared with rank-1 steering vectors (arXiv 2505.20809, 2025-05).
- **TRN-10.** Liu et al., "IA³", arXiv 2205.05638 (2022-05).
- **TRN-11.**
  - Shuttleworth et al., "LoRA vs Full Fine-tuning: An Illusion of Equivalence", arXiv 2410.21228 (2024-10): LoRA introduces "intruder" singular vectors.
  - Minder et al., arXiv 2510.13900 (ICLR 2026): narrow fine-tunes leave readable traces in activation differences. Mixing pretraining data into the fine-tune removes the trace, so a null diff doesn't prove nothing changed.
- **TRN-12.** Activation checkpointing conflicts with LRP's cached activations.
- **TRN-13.**
  - rsLoRA: Kalajdzievski, arXiv 2312.03732 (2023-12).
  - LoRA+: Hayou et al., arXiv 2402.12354 (2024-02).
  - DoRA: Liu et al., arXiv 2402.09353 (2024-02).
- **TRN-14.**
  - PiSSA: arXiv 2404.02948.
  - VeRA: arXiv 2310.11454.
  - AdaLoRA: arXiv 2303.10512.
  - GaLore: arXiv 2403.03507.
  - QLoRA: arXiv 2305.14314.
  - Prefix tuning: arXiv 2101.00190.
  - Prompt tuning: arXiv 2104.08691.
  - BitFit: arXiv 2106.10199. Most modern LLMs have no biases, so BitFit is nearly useless for them.
  - Houlsby adapters: arXiv 1902.00751.
- **bf16 updates round away small steps.** Zamirai et al., arXiv 2010.06192. Keep fp32 master weights.

## LLM: Running real language models

- **LLM-1.** SmolLM2 needs grouped-query attention. Qwen3 and Gemma 3 need `head_dim` separate from `d_model / num_heads`.
  - pulsatrix's `RoPEModule` rotates adjacent pairs, while Hugging Face Llama-family checkpoints use "rotate half".
  - **Fails when:** the AttnLRP rules stop conserving relevance under a mask. Re-validate against LXT.
- **LLM-2.** SmolLM2, Qwen3 and Gemma 3 set `tie_word_embeddings`. Without views, an untied copy of Gemma 3 270M's embedding would cost about 680 MB in fp32.
- **LLM-3.** Moved to the TOK epic below.
- **LLM-4.** Sampling must use the seeded RNG from FND-7 so generations are reproducible.
- **LLM-5.** **Falsifier:** cached and uncached logits differ.
- **LLM-6.** The threshold (fp32, max abs diff under 1e-3) follows from IO-4's layout pitfalls: random-weight tests miss them, real weights on real text don't.
- **LLM-7.**
  - Achtibat et al., "AttnLRP", arXiv 2402.05602 (ICML 2024).
  - LXT 2.1 (2025-07-10) supports Gemma 3 and Qwen3, and marks Qwen3 experimental with attribution skewed toward the first token.
  - **Falsifier:** per-token relevance correlation with LXT is below 0.99 on Llama-family models.
- **LLM-8.**
  - Tuned lens: Belrose et al., arXiv 2303.08112 (2023-03).
  - AtP*: Kramár et al., arXiv 2403.00745 (2024-03). Plain attribution patching misses effects through attention saturation.
- **LLM-9.** Gemma 3 technical report, arXiv 2503.19786 (2025-03). Gemma Scope 2 (Google DeepMind, 2025-12) publishes SAEs and transcoders for every layer of Gemma 3 270M and 1B.

## TOK: Tokenizers

- **TOK-1.** The component split (normalizer, pre-tokenizer, model, post-processor, decoder) is the
  one Hugging Face `tokenizers` uses, so a `tokenizer.json` maps onto it one section at a time.
  - Offsets are measured in bytes of the UTF-8 input. Characters would need a second index, and every consumer here (VIZ-6a, LLM-7) slices UTF-8 strings.
  - **Fails when:** a normalizer changes length (NFC, `▁` replacement) and offsets point at the normalized text instead of the original. Hugging Face keeps an alignment map for this; we need one too.
- **TOK-2.**
  - mlc-ai/tokenizers-cpp (Apache-2.0) exists but wraps the Rust tokenizers library, which pulls in a Rust toolchain.
  - A C++ port of the Hugging Face tokenizer found double-space mismatches in 4 of 15 million lines (wangkuiyi/huggingface-tokenizer-in-cxx).
  - SmolLM2 and Qwen2.5 store merges as `"a b"` strings. Qwen3, Llama 3.2 and Gemma 3 store them as `["a", "b"]` pairs (newer `tokenizers` releases).
  - Llama 3.2 sets `ignore_merges`: a pre-token that is already in the vocabulary is used whole, without merging.
  - Qwen's only normalizer is NFC, which needs Unicode composition tables. They are generated, like the category table.
  - Split regexes seen (2026-10-05): GPT-2, Llama 3 (same as `cl100k_base`, also Phi-4), Qwen, `o200k_base` (gpt-oss, Mistral Nemo: case-aware with `\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}`) and DeepSeek-V3 (three Splits, including CJK ranges). Five families in a year argues for an engine over hand-written pre-tokenizers.
  - The engine only needs what these patterns use, and must match the backtracking, leftmost-first semantics of Oniguruma, which Hugging Face `tokenizers` uses by default (`fancy-regex` behind a feature flag).
  - **Falsifier:** id mismatches on the 10,000-line corpus.
- **TOK-3.** Gemma 3's normalizer replaces spaces with `▁`, so its pre-tokenizer split does nothing, and whole lines go to BPE as one piece. BPE has to be linear-ish in the piece length, not quadratic.
- **TOK-6.** Training was out of scope for the first tokenizer ("a project-sized undertaking"). A byte-level BPE trainer with a priority queue over pair counts is a few hundred lines; the hard part is matching Hugging Face's tie-breaking if we want identical vocabularies, and we don't need that.
- **TOK-8.** A `.tiktoken` file has no merge list: BPE merges the adjacent pair whose concatenation has the lowest rank. gpt-oss ships both `tokenizer.json` and the o200k encoding, which gives a cross-check between TOK-2 and TOK-8.
- **TOK-7.** T5's precompiled charsmap is a serialized double-array trie of normalization rules (316 KB base64 for t5-small).

## XAI: Question-driven framework

- **Four reasons to explain.** Adadi and Berrada, "Peeking Inside the Black-Box", IEEE Access 6 (2018): justify, control, improve, discover. The exact wording comes from secondary summaries; the PDF section still needs reading.
- **Nine question categories.** Liao, Gruen and Miller, "Questioning the AI", CHI 2020, arXiv 2001.02478. The categories were checked against the paper's Figure 1.
  - Designers report that "why not" questions are common and poorly served.
- **Question-to-method mapping.** Liao et al., "Question-Driven Design Process for Explainable AI User Experiences", arXiv 2104.03483 (2021-04). It notes an "availability bias": practitioners default to LIME and SHAP.
- **Evaluation levels.** Doshi-Velez and Kim, arXiv 1702.08608 (2017-02): application-grounded, human-grounded and functionally grounded evaluation. A library can only own the functionally grounded level.
- **XAI-1.** The enums mirror Liao et al. (2020) and Adadi and Berrada (2018) exactly, so reports can be compared against the papers' question bank.
- **XAI-2, XAI-3.** **Falsifier:** users ignore the planner and call LRP directly. That's why the planner is optional and never hides the direct APIs.
- **XAI-4.**
  - Counterfactuals: Wachter et al., arXiv 1711.00399 (2017-11). DiCE: Mothilal et al., arXiv 1905.07697. On image models counterfactuals turn into adversarial examples (Freiesleben, arXiv 2009.05487).
  - Anchors: Ribeiro et al., AAAI 2018. Coverage can be tiny, and precision is estimated, not guaranteed.
  - TREPAN surrogates: Craven and Shavlik, NeurIPS 1996. **Falsifier:** fidelity under about 85% at a depth a person can read.
  - Pertinent negatives (CEM): Dhurandhar et al., NeurIPS 2018.
  - ProtoDash: Gurumoorthy et al., ICDM 2019.
- **XAI-5.**
  - Quantus: Hedström et al., JMLR 2023, arXiv 2202.06861. It groups metrics into six families: faithfulness, robustness, localisation, complexity, randomisation and axiomatic.
  - ROAD: Rong et al., ICML 2022.
  - The model-parameter randomization test comes from Adebayo et al., NeurIPS 2018. Done (#53): a model-independent "explainer" scores a similarity of 1.0 at every layer, and gradient × input decorrelates.
- **XAI-6.** See the cross-cutting section above. Done (#54).

## INT: Embedding and representation analysis

- **INT-1.** Huh et al., "Platonic Representation Hypothesis", arXiv 2405.07987 (2024-05), defines mutual-kNN alignment.
  - **Fails when:** a few points become everyone's neighbor in high dimensions (hubness).
- **INT-2.** Kornblith et al., ICML 2019, arXiv 1905.00414.
  - **Falsifier:** CKA between a trained and a random-init network stays high. That would mean it measures the architecture, not the learning.
  - **Fails when:** outliers sway it, and it can be manipulated (Davari et al., arXiv 2210.16156).
- **INT-3.** Hewitt and Liang, EMNLP 2019. MDL probing: Voita and Titov, EMNLP 2020.
- **INT-4.**
  - Anisotropy: Ethayarajh, EMNLP 2019.
  - TwoNN: Facco et al., Scientific Reports 2017; applied to deep networks by Ansuini et al., NeurIPS 2019.
- **INT-6.**
  - TCAV: Kim et al., ICML 2018, arXiv 1711.11279. Concept vectors are unstable across random negative sets.
  - CRP: Achtibat et al., Nature Machine Intelligence 2023, arXiv 2206.03208.
- **INT-7.**
  - RSA: Kriegeskorte et al., 2008.
  - SVCCA: Raghu et al., NeurIPS 2017.
  - PWCCA: Morcos et al., NeurIPS 2018.
- **INT-8.** t-SNE and UMAP distort global distances and cluster sizes (Wattenberg et al., Distill 2016; Chari and Pachter, PLoS Comp Bio 2023).

## TDA: Topological data analysis

- **Core theory.**
  - Persistent homology: Edelsbrunner, Letscher and Zomorodian, DCG 2002; Zomorodian and Carlsson, DCG 2005.
  - Stability: Cohen-Steiner, Edelsbrunner and Harer, DCG 2007.
  - Persistence modules as functors, and the interleaving distance: Chazal, de Silva, Glisse and Oudot, *The Structure and Stability of Persistence Modules* (2016).
  - Single linkage is the unique stable functorial clustering: Carlsson and Mémoli, JMLR 2010.
  - Mapper as a cosheaf: de Silva, Munch and Patel, DCG 2016.
- **TDA-2, TDA-3.** Distances via ‖x‖² + ‖y‖² − 2xyᵀ suffer fp32 cancellation for near-duplicate points; clamp at zero and carry the error as a diagram bound (stability gives d_B ≤ ‖D − D′‖∞). H0 equals single-linkage clustering, which is the minimum spanning tree (Carlsson and Mémoli 2010).
- **TDA-4.** Ripser: Bauer, JACT 2021, arXiv 1908.02518. MIT, about 1000 lines, no dependencies.
  - giotto-ph is AGPLv3, so it can't be vendored.
  - GUDHI's CGAL-dependent modules are GPLv3.
  - Practical limits on CPU [I]: H1 up to about 2–5k points, H2 up to about 1k.
- **TDA-5.**
  - Bottleneck distance: Kerber, Morozov and Nigmetov 2017.
  - Wasserstein stability: Skraba and Turner, arXiv 2006.16824.
- **TDA-6.**
  - Persistence landscapes: Bubenik, JMLR 2015.
  - Persistence images: Adams et al., JMLR 2017.
  - Betti curves are not stable under the bottleneck distance.
- **TDA-7.** The landmark subsampling error is at most twice the covering radius. Witness complexes: de Silva and Carlsson 2004.
- **TDA-8.**
  - Bootstrap confidence sets: Fasy et al., Annals of Statistics 2014.
  - Permutation tests: Robinson and Turner 2017.
  - The universal null: Bobrowski and Skraba 2023.
- **TDA-9.** Barcodes and persistence diagrams follow VIZ-1's JSON format; the significance band from TDA-8 is drawn on the diagram.
- **TDA-10.**
  - Engels et al., "Not All Language Model Features Are Linear", ICLR 2025, arXiv 2405.14860: days of the week and months form circles in GPT-2 and Mistral.
  - Goodfire's BSF vision post (2026-07-07) measures block dimension by stable rank, a linear measure.
  - Circular coordinates: de Silva, Morozov and Vejdemo-Johansson, DCG 2011.
  - **Falsifier:** the dominant H1 bar isn't significant against the null, or it also appears on random-init activations.
- **TDA-11.** RTD: Barannikov, Trofimov et al., ICML 2022.
- **TDA-12.** Naitzat, Zhitnikov and Lim, "Topology of Deep Neural Networks", JMLR 2020, arXiv 2004.06093.
- **TDA-13.**
  - Distance concentration hides loops in high dimensions: Damrich, Berens and Kobak, NeurIPS 2024, arXiv 2311.03087.
  - Distance-to-measure: Chazal, Cohen-Steiner and Mérigot 2011.
- **TDA-14.**
  - Mapper: Singh, Mémoli and Carlsson 2007.
  - Instability and parameter dependence: Belchí et al., JMLR 2020; Carrière, Michel and Oudot, JMLR 2018.
- **TDA-15.** Carrière et al., ICML 2021. Topological autoencoders: Moor et al., ICML 2020.
- **TDA-16.** TopoTroj: Zheng et al., NeurIPS 2021, arXiv 2106.06469.
  - **Falsifier:** a probe on norms or kNN distances matches it.
- **TDA-17 to TDA-19.** Zigzag persistence on LLM layers: Gardinazzi et al., ICML 2025, arXiv 2410.11042. Multiparameter persistence has no complete invariant (Carlsson and Zomorodian 2009).
- **Neural persistence.** Rieck et al., ICLR 2019. It mostly reduces to weight variance (Girrbach et al., arXiv 2307.10865, 2023).
- **Caution on recent LLM-topology papers.** Most from 2025–26 use H0 only and are unreviewed preprints. H0 is a function of the minimum spanning tree, so check those results against plain kNN statistics.

## KD: Surrogates and student-teacher

- **KD-1.** Hinton, Vinyals and Dean, arXiv 1503.02531 (2015-03).
  - Students often don't match teachers even with enough capacity: Stanton et al., "Does Knowledge Distillation Really Work?", NeurIPS 2021.
- **KD-2.** If prediction agreement is high but explanation agreement is low, that is the interesting signal, not a bug.
- **KD-3.**
  - FitNets: arXiv 1412.6550.
  - Attention transfer: arXiv 1612.03928.
  - Soft decision trees: Frosst and Hinton, arXiv 1711.09784.
  - Boolean rule column generation: Dash, Günlük and Wei, NeurIPS 2018.
- **KD-4.**
  - Born-again networks: Furlanello et al., arXiv 1805.04770.
  - DistilBERT: Sanh et al., arXiv 1910.01108.

## FEAT: Featurizers

- **FEAT-6, BSF.**
  - Fel et al., "Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds", arXiv 2606.25234 (2026-06-23).
  - The code is at github.com/goodfire-ai/block-sparse-featurizer. The variants are:
    - **vanilla:** keep the top-k blocks by norm;
    - **Grassmannian:** tied, with orthonormal block frames;
    - **group lasso:** a block soft-threshold.
  - The paper evaluates with minimum description length and stable rank.
  - The follow-up, Jerpelea and Ananthram, "A Deeper Analysis of Block-Sparse Featurizers", arXiv 2608.27515 (2026-08-27), reports that BSF still suffers from feature splitting and composition, and proposes tournament top-k.
  - **Falsifier:** on pulsatrix-scale models, BSF fails to beat a TopK SAE at matched L0.
- **FEAT-2.** Gao et al., arXiv 2406.04093 (2024-06), introduces the auxiliary loss for dead latents.
- **FEAT-3.**
  - SAEBench: Karvonen et al., arXiv 2503.09532 (2025-03). Its authors note that proxy metrics don't reliably predict usefulness.
  - Feature absorption: Chanin et al., arXiv 2409.14507.
- **FEAT-4.**
  - BatchTopK: arXiv 2412.06410.
  - JumpReLU: arXiv 2407.14435.
  - Matryoshka: arXiv 2503.17547.
- **FEAT-5.**
  - Transcoders: Dunefsky et al., arXiv 2406.11944.
  - "Transcoders Beat Sparse Autoencoders for Interpretability": Paulo et al., arXiv 2501.18823.
- **FEAT-7.** AxBench (Wu et al., arXiv 2501.17148, 2025-01): SAE steering underperforms simple baselines. Steering vectors are unreliable in distribution and misgeneralize out of it (Tan et al., NeurIPS 2024).
- **FEAT-8.** Crosscoders: Lindsey et al., Transformer Circuits (2024-10). Delta-Crosscoder: arXiv 2603.04426 (2026-03).
- **FEAT-9.**
  - APD: arXiv 2501.14926.
  - SPD: Bushnaq, Braun and Sharkey, arXiv 2506.20790.
  - VPD: Goodfire (2026-05). Its technical report URL returned 404, so the details are unverified.
- **FEAT-10.** Ameisen, Lindsey et al., "Circuit Tracing", Transformer Circuits (2025-03-27). The authors list limits: QK circuits aren't explained, there is reconstruction error, and inactive features are missed.
- **SAE limits.**
  - Linear probes beat SAE probes out of distribution: DeepMind, "Negative Results for SAEs on Downstream Tasks" (2025-03-26). After this the team deprioritized SAE research.
  - SAEs trained on different seeds don't agree: Paulo and Belrose, arXiv 2501.16615.
  - Position SAEs as discovery tools: Peng et al., arXiv 2506.23845.

## ARCH: New architectures

- **ARCH-1.** B-cos: Böhle, Fritz and Schiele, CVPR 2022, arXiv 2205.10268. B-cos language models: arXiv 2502.12992.
- **ARCH-2.** Koh et al., ICML 2020, arXiv 2007.04612. Concept leakage: Mahinpei et al., arXiv 2106.13314.
- **ARCH-3.** Switch Transformer: arXiv 2101.03961. OLMoE router analysis: arXiv 2409.02060. Experts tend to specialize by token and syntax rather than domain.
- **ARCH-4.** ProtoPNet: arXiv 1806.10574. Human and model similarity don't always match: Hoffmann et al., arXiv 2105.02968.
- **ARCH-5.**
  - GNNExplainer: arXiv 1903.03894.
  - GNN-LRP: Schnake et al., TPAMI 2021.
  - Explainer benchmarks are unreliable: Faber et al. 2021.
- **ARCH-6.** Dao and Gu, arXiv 2405.21060. No SSD LRP reference implementation was found.
- **ARCH-7.** Beck et al., NeurIPS 2024, arXiv 2405.04517.
- **ARCH-8.** KAN: arXiv 2404.19756. In "KAN or MLP: A Fairer Comparison", arXiv 2407.16674, KANs lose outside symbolic formulas.
- **ARCH-9.** Peebles and Xie, arXiv 2212.09748.
- **ARCH-10.**
  - Gated DeltaNet: arXiv 2412.06464.
  - Titans: arXiv 2501.00663.
  - Hopfield layers: arXiv 2008.02217.
  - V-JEPA 2: arXiv 2506.09985.
  - SENN: arXiv 1806.07538.
  - Hypernetworks: arXiv 1609.09106.
  - Neural ODEs: arXiv 1806.07366.
  - Liquid time-constant networks: arXiv 2006.04439.

## VIZ: Visualization pack

- **Tools go stale.** CircuitsVis's last release was about a year ago, and token-hover regressions are still open (#98) [C]. Captum Insights was deprecated in Captum 0.8 (2025-03) [C].
- **VIZ-1.**
  - **Fails when:** NaN and infinity appear in JSON (they're invalid), or the schema churns. Version it from day one and add golden-file tests. Done (#62): versioned from v1, non-finite numbers carried as null plus a JSON Pointer entry, and golden files byte-compared on Linux and Windows.
- **VIZ-2.** ImPlot has no vector export, so publication figures need a separate renderer [I]. Share `colormap.hpp` so SVG and ImGui colors match. Done (#63): tests check the SVG fills against `colormap.hpp` exactly.
- **VIZ-3.** Vega-Lite embedding docs. Inlining the JavaScript adds roughly 800 KB per file [I].
- **VIZ-4.** circuit-tracer (Anthropic open-sourced it, 2025). Neuronpedia publishes an attribution-graph JSON schema and a validator at neuronpedia.org/graph/validator.
  - **Falsifier:** the schema needs transcoder-specific fields pulsatrix can't fill. Check against the validator first.
- **VIZ-6.** Reference designs: SAEDashboard (jbloomAus), Neuronpedia feature pages, the TensorBoard projector, and SHAP's plots.
- **VIZ-7.**
  - imnodes: maintained, but has no zoom.
  - imgui-node-editor: upstream is stale; the imgui_bundle fork is maintained (discussion #428, 2026-01) [C].

## NB: Notebook layer

- **NB-1.** xeus-cpp's `xcpp::display` calls `mime_bundle_repr` without qualification, so argument-dependent lookup finds an overload in the `pulsatrix` namespace. This was read from `include/xcpp/xdisplay.hpp`.
  - **Falsifier:** `xcpp::display(tensor)` in xeus-cpp 0.10 doesn't pick up the overload.
- **NB-2.** The nbformat v4 format description: since 4.5, every cell needs an `id`. MIME types ending in `+json` are stored as objects, and PNG data is base64.
  - **Falsifier:** the output fails `nbformat.validate`, or doesn't render on GitHub.
- **NB-3.** anywidget (MIT, 0.11.0, 2026-04-27) works in Jupyter, JupyterLab, Colab, VS Code and marimo.
- **NB-4.** xeus-cpp 0.10.0 (BSD-3-Clause, released 2026-04-02).
  - **Kernel crashes:** a segfault or failed assert restarts the kernel (#434), and lambdas with xwidgets crash it (#476).
  - **GPU:** CUDA math functions return 0 inside JIT-compiled kernels (#467).
  - **Windows:** "Does not work on Windows" is still open (#144).
  - xeus-cling hasn't been updated since 2025-10, so don't target it.
- **NB-5.** xeus-cpp-lite (Jupyter blog, 2025-06-19) loads libraries as emscripten-forge side modules. Threads need SharedArrayBuffer headers [I].

## HIP: Training efficiency on AMD GPUs

- **Hardware ceiling.**
  - Memory bandwidth: 256 GB/s theoretical (LPDDR5X-8000, 256-bit), about 212 GB/s measured on the GPU [C] (llm-tracker Strix Halo page).
  - fp16/bf16 peak: about 59 TFLOPS from WMMA [V] (GPUOpen, "WMMA on RDNA3"). About 37 TFLOPS bf16 measured with an optimized stack [C].
  - fp32 peak: about 15 TFLOPS without dual issue [I].
  - So nearly every non-GEMM kernel is bandwidth-bound or worse [I].
- **warpSize and shuffles.** In ROCm 7.0, `warpSize` is no longer `constexpr`, and `*_sync` shuffles take 64-bit masks [V] (ROCm 7.0 release notes; ROCm/HIP #3667).
- **gfx1151 ISA.** The LLVM AMDGPU target table lists gfx1151 as an APU with no XNACK [V] (LLVM AMDGPUUsage).
- **HIP-1.**
  - rocprofv3 (ROCprofiler-SDK) is the standard profiler in ROCm 10.0 [V].
  - rocprof-compute support on gfx115x landed in TheRock PR #8299 [C]. ROCm 10.0 fixed its roofline precision list [V]. It's unverified on 7.2.4.
- **HIP-2.**
  - **Falsifier:** rocprofv3 shows these kernels take under 5% of step time on real workloads. It held for softmax, the norms and `column_sums`, which were left alone. It failed for BatchNorm (79% of the CNN's kernel time), which was rewritten: step time fell from about 27 ms to about 10 ms (#57).
  - **Fails when:**
    - a 32-bit mask truncates the shuffle on 64-wide waves;
    - results differ from the CPU in the last bit (use tolerances);
    - a wave64 build changes the reduction order.
- **HIP-3.**
  - **Design:** PyTorch's caching allocator, described by Zach DeVito (2022) and in the PyTorch devlog on fragmentation (2026-06-01).
  - **Avoid `hipMallocAsync` on RDNA** [C]:
    - silent corruption on gfx1030 (CTranslate2 #2090);
    - OOM reported while memory is still consumed (rocm-systems #11642);
    - double free on pool failure (rocm-systems PR #11643).
  - **VRAM on the APU** [C]:
    - ROCm adds VRAM and GTT together when reporting capacity, which overstates what can be allocated (ROCm #6004).
    - AMD recommends a 0.5 GB carve-out on Strix Halo [V] (AMD Strix Halo system optimization guide).
- **HIP-4.** **Falsifier:** the GPU is already more than 90% busy with the syncs in place. It didn't hold: HIP-1 measured 12–13% busy on the small models, and removing the syncs made them about 2× faster (#59).
  - **Fails when:** the host reads stale data, or an error surfaces in a later op.
- **HIP-5.** `dot` and `sum` launch a single 256-thread block today (`src/gpu_kernels.cuh`). Use per-block partials and a second pass; no atomics, to keep results deterministic. Done (#56): both now run at the measured memory bandwidth.
- **HIP-6.** **Fails when:** the fused training path and the unfused explain path drift apart numerically. Gate them with a conservation test.
- **HIP-7.** At N=64, a 3×3 kernel on 224×224 input needs about 350 MB of im2col buffer [I].
  - MIOpen on gfx1151 has a history of trouble [C]: missing Composable Kernel libraries (TheRock #5105, closed), and a Winograd lockup (TheRock #5581).
- **HIP-8.** Sources disagree [C].
  - Against: ROCm #5643 (ROCm 7.1.0) reports "unsupported architecture".
  - For: the changelog and community guides say ROCm 7.2.4 ships gfx1151 kernels.
  - An older run measured hipBLASLt slower than rocBLAS.
  - A 30-line probe settles it.
- **HIP-9.**
  - ROCm 10.0.0 (2026-08-26) is the first release whose notes list the Ryzen AI Max parts as gfx1151 [V] (release notes, compatibility matrix).
  - ROCm 7.2 page faults on host-to-device copies (ROCm #5890, open) [C].
  - A wrong VGPR count crashed gfx1151 through ROCm 7.1.x. The user-mode fix landed in 7.2; the kernel-mode fix needs Ubuntu's OEM kernel 1018 or newer (TheRock #2991) [C].
  - AMD asks for host kernel 6.18.4 or newer on other distributions [V].
  - Done (#65): on gfx1151 with host kernel 7.0.0, ROCm 10.0.0 passed every HIP test and matched 7.2.4 within 2%; it is now the default. ROCm #5890 didn't show up on either version.
- **HIP-10.** HIP's unified-memory documentation [V]: without XNACK, `hipMallocManaged` and `hipHostMalloc` give pinned, zero-copy host memory.
- **HIP-11.** LRP's stabilized divisions and conservation checks degrade in bf16, which has an 8-bit mantissa [I]. That's why the rule is to explain in fp32.
- **HIP-12.** Reports that this work may not pay off [C]:
  - hipGraph is 2–5% slower than eager execution on gfx11 (legacy-rocm-build #6685);
  - HIP graph branches get serialized on gfx1201 (rocm-systems #12330);
  - Composable Kernel's flash attention on RDNA3 is forward-only (CK #1434);
  - rocWMMA flash attention regresses prefill by 41% on gfx1151 (llama.cpp #24437).

## AGT: Agents

- **No official C++ SDKs.**
  - Anthropic's official SDKs are Python, TypeScript, C#, Go, Java, PHP and Ruby [V] (client SDK docs, checked 2026-10-03).
  - MCP's official SDK tiers [V] (modelcontextprotocol.io/docs/sdk):
    - Tier 1: TypeScript, Python, C#, Go, Rust and Ruby.
    - Tier 2: Java.
    - Tier 3: Swift, PHP and Kotlin.
- **Community C++ MCP libraries.**
  - gopher-mcp (Apache-2.0, v0.1.18): its own issue #294 says 2026-07-28 support had only been tested against itself.
  - cpp-mcp (MIT): targets older spec revisions.
  - Neumann-Labs/mcp-cpp: GPL-3.0, so it can't go into an MIT library.
- **AGT-1.** nlohmann/json (MIT, v3.12.0). xeus-cpp's display API also uses it, so one dependency serves AGT, VIZ-1 and NB-1. Throughput doesn't matter at these payload sizes.
- **AGT-2.**
  - cpp-httplib (MIT, v0.59.0): requires OpenSSL 3.0 or newer. Issue #2596 (opened 2026-09-29) reports Windows certificate verification failures.
  - cpr (MIT-style) wraps libcurl, which can use Windows' own TLS stack.
  - **Falsifier:** Windows CI fails to verify `api.anthropic.com`. Then switch the default to libcurl.
- **AGT-3: Anthropic API behavior** [V] (define-tools, handle-tool-calls, rate-limits and streaming docs, 2026-10-03):
  - `tool_choice: any` and `tool_choice: tool` return HTTP 400 on the current 5.x models.
  - `tool_result` blocks must come first in the next user message.
  - A 429 from a spend limit has error code `enforced_spend_limit_reached` and no `retry-after` header.
  - New stream event types can appear at any time.
  - Streamed tool input arrives as `input_json_delta` fragments, parsed at `content_block_stop`.
- **AGT-4.**
  - MCP schemas default to JSON Schema 2020-12, but the main C++ validator (pboettch/json-schema-validator 2.4.0) supports draft-7 only.
  - **Falsifier:** more than two real tools need `$ref` or `oneOf`.
- **AGT-5.**
  - MCP 2026-07-28 changelog [V]: no `initialize` handshake, no sessions, and `server/discover` is required.
  - stdio transport [V]: nothing but protocol messages on stdout, and no embedded newlines.
  - tools spec [V]: servers must validate inputs, apply access controls, rate-limit and sanitize output.
  - Security statistics about popular MCP servers (for example, 43% with command injection) come from secondary vendor reports (Checkmarx), not primary scans.
- **AGT-6.**
  - Grammar-constrained decoding is fragile at the edges of the schema language (llama.cpp #25923, #22314, #29006).
  - XGrammar (Apache-2.0, v0.2.8) has a C++ core.
  - SmolLM2-135M reaches about 22% exact match on xLAM function calling even after full fine-tuning (chloe-mp/smollm2-xlam-function-calling) [C]. Grammar constraints guarantee well-formed calls, not correct ones.
- **AGT-7.** No prior art or benchmark was found for attributing an agent's tool choice. The falsifier is the deletion test in the roadmap.
- **AGT-8.**
  - "Language Models Can Explain Neurons": Bills et al., OpenAI (2023-05).
  - Delphi: Paulo et al., arXiv 2410.13928 (2024-10). Its explanations fail about 38% of the time (arXiv 2605.01555, 2026-05).
  - Output-centric descriptions: arXiv 2501.08319.
- **AGT-9.** MAIA: Shaham et al., arXiv 2404.14394 (ICML 2024). Transluce's Docent (Apache-2.0) analyzes agent transcripts.
- **Prior art.** mozilla-ai/agent.cpp (MIT, 2025-12) is a C++17 agent loop tied to llama.cpp, with no MCP. It's a design reference, not a dependency.

## KS: Kitchen sink

- **KS-1.** The README said "There's no `install()` step yet." Done (#61).
- **KS-2.** The benchmark suite provides the numbers HIP-1 through HIP-8 are judged by, plus conservation error so performance work can't silently break LRP. Done (#64): a 1000× larger LRP stabilizer fails the conservation gate (0.13 against 10⁻³), and restoring HIP-4's syncs is flagged as a regression.
- **KS-3.**
  - Monte Carlo dropout: Gal and Ghahramani 2016.
  - Deep ensembles: Lakshminarayanan et al. 2017.
  - Split conformal prediction: Vovk et al.; Angelopoulos and Bates, arXiv 2107.07511.
- **KS-4.**
  - FGSM: Goodfellow et al., arXiv 1412.6572.
  - PGD: Madry et al., arXiv 1706.06083.
- **KS-5.** TracIn: Pruthi et al., NeurIPS 2020. Captum 0.8 added influence functions [C]. Kronfluence's EK-FAC covers only linear and conv layers [C].
- **KS-6.** `docs/data-pipeline/index.md` currently calls drift detection and fairness metrics out of scope. Model cards: Mitchell et al., FAT* 2019.
- **KS-7.** `DataThreadPool` and `BoundedQueue` exist but aren't wired into `DataLoader` (`data_loader.hpp`).
- **KS-8.** The HIP backend is tested on real gfx1151 hardware but not in CI (README Status).
  - **Risk:** runner flakiness from the gfx1151 driver issues listed under HIP-9.
- **KS-9.** Llama 3.2 is under the Llama 3.2 Community License. Ship converters, not converted weights.
- **KS-11.** Python wheels with GPU backends are the hard part. A CPU-only wheel and a vcpkg port (mlpack and dlib both ship one) come first.
- **KS-10, KS-12.** These close gaps against the other libraries we compared:

  | Library | What it has that pulsatrix lacks |
  |---|---|
  | PyTorch, JAX | Views, broadcasting, op-level autograd |
  | tinygrad | Automatic fusion |
  | ggml and llama.cpp | GGUF and quantization |
  | Captum | Influence functions |
  | Quantus | Evaluation metrics |
  | TransformerLens and nnsight | Hook points |
  | SAELens | SAE training at scale |
