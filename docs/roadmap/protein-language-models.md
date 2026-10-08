# Protein language models: research and plan

Status: **complete** (2026-10-08). Accepted as its own epic (PLM in the [roadmap](index.md#plm-protein-language-models)) and researched on 2026-10-07. What was built, and where it departed from this plan, is in the roadmap's PLM section and the [Protein language models](../protein-models/index.md) guide.

## What a protein language model is

A protein language model (pLM) is a transformer trained on amino-acid sequences the way a text
model is trained on words. Nobody labels anything: the model learns by filling in hidden
residues. In doing so it picks up evolution's constraints. It learns which positions tolerate
change, which residues touch in the folded structure, and which motifs mark a binding or active
site. Those can then be read out without further training, or used as features for small
supervised models.

| Family | Examples | Objective | Typical use |
|---|---|---|---|
| Encoder, masked LM | ESM-2 (8M to 15B), ESM C (300M, 600M, 6B), AMPLIFY, ProtBert, SaProt (adds structure tokens) | Fill in 15% hidden residues | Embeddings, variant effects, contacts, interpretability |
| Encoder-decoder | ProtT5-XL (3B), Ankh | Span denoising | Per-residue embeddings |
| Autoregressive | ProGen2, ProtGPT2 | Next residue | Generating sequences |
| Retrieval or MSA-conditioned | MSA Transformer, PoET-2, MSA Pairformer | Conditioned on homologs | Best variant-effect scores |
| Multimodal or diffusion | ESM3 (sequence, structure, function tracks), DPLM-2 | Masked or diffusion over several tracks | Design |

**Recommendation: build for ESM-2 first.**
- It is MIT-licensed and the most cited and most studied pLM.
- The interpretability literature, including InterPLM's sparse autoencoders and the contact-from-attention work, uses it.
- Its sizes run from 8M to 650M parameters, which fits this machine in fp32.
- Its architecture is a standard BERT-style encoder: rotary position embeddings, pre-LayerNorm and a GELU MLP. Most of that pulsatrix already has.

ESM C and AMPLIFY are architectural cousins (AMPLIFY uses SwiGLU and RMSNorm, which pulsatrix has). They can follow by adding a weight mapping. ESM C's license tiers need checking per size.

## How they're trained

ESM-2's recipe (Lin et al., *Science* 2023, and NVIDIA BioNeMo's reproduction):

- **Data.** UniRef50 clusters (about 65M), each sampled uniformly, with a random UniRef90 member drawn per cluster. That is about 190M sequences.
  - Sampling clusters, not sequences, keeps over-studied families from dominating.
  - Newer models add metagenomic data (MGnify, JGI). Pure UniRef shows overfitting in masked-LM training and diminishing returns for causal training (Cheng et al., ICML 2024). A 2025 study of yearly UniRef snapshots found no saturation yet.
- **Objective.** Pick 15% of residues; replace 80% of those with `<mask>`, 10% with a random residue, and leave 10% unchanged. Cross-entropy is computed on the picked positions only.
  - ESM-2's "token dropout" zeros masked embeddings and rescales the rest by the observed mask ratio. Inference has to reproduce that scaling.
- **Shape.** Crop sequences to 1,024 residues. `<cls>` and `<eos>` mark the ends. The vocabulary is 33 tokens: the 20 amino acids, the rare and ambiguous ones (X B U Z O), and gap and special tokens.
- **Scale.** The 650M model took on the order of 10²¹ FLOPs, with up to 3.2M tokens per batch. That is far beyond one workstation.

**What that means here.**
- Pretraining a useful pLM from scratch isn't realistic on gfx1151 in fp32. bf16 training (HIP-11) isn't done, and even with it a 650M model would take months.
- Small models (8M–35M parameters on a UniRef50 sample) are trainable. They're for learning, ablations and checking the training code against published curves.
- The realistic path is to load the published ESM-2 checkpoints, explain and evaluate them, and fine-tune small heads on top.

## How results are validated

The field's main checks, and the ones pulsatrix should reproduce numerically before claiming
support:

1. **Parity with the reference implementation.** Masked-LM logits, hidden states and attention maps should match Hugging Face `transformers` (`EsmForMaskedLM`) to float precision. This is the same harness as LLM-6 and LLM-7.
2. **Zero-shot variant effects (ProteinGym).** 217 deep-mutational-scanning assays, about 2.5M variants.
   - A variant is scored by the masked-marginal log-odds, log p(mutant) − log p(wild type), at the masked position.
   - Results are reported as Spearman correlation per assay, plus NDCG and top-10% recall.
   - Published ESM-2 numbers give exact targets. The best methods reach about 0.45–0.48 Spearman (MSA Pairformer, VespaG); ESM-2 650M is lower.
3. **Unsupervised contact prediction.** Attention maps from all layers and heads are made symmetric, corrected with APC, and combined by a logistic regression (ESM's contact head). The score is precision@L, L/2 and L/5 for long-range contacts (|i−j| ≥ 24) against residues within 8 Å in a PDB or AlphaFold structure.
   - A 2026 paper (arXiv 2606.21876) shows that averaging the top-K contact heads, chosen with 10 labeled proteins, matches the Categorical Jacobian in a single forward pass.
   - It also warns that leakage-clean test sets score 30–36 points lower than in-distribution ones.
4. **Probing.** A linear probe per layer for secondary structure (DSSP 3- and 8-state), solvent accessibility and binding sites. Control tasks and a randomized-model baseline are needed to show the probe reads the model and isn't learning the task itself.
5. **Explanation sanity checks.** A 2026 study found that residue attributions from a well-performing allergen classifier did not match known epitopes. Protein-level accuracy doesn't make attributions meaningful. Every residue-level explanation should therefore be checked against:
   - a randomized-weights model (Adebayo-style sanity check);
   - deletion and insertion curves;
   - an independent signal: DMS per-position sensitivity, conservation from an MSA, or annotated sites.
6. **Data-leakage-aware splits.** A naive split overstated performance by about 11% relative to a split aware of ESM's pretraining data. Hold out by sequence identity or by date.

## What explains a pLM best

Proteins have three natural axes: the sequence, the 3D structure and evolution. Good views show
model output on all three, not as a text strip.

| View | What it answers | pulsatrix today |
|---|---|---|
| **Mutational landscape**: position × 20 amino acids, colored by log-odds | Which substitutions does the model think are tolerated? This is the DMS-style map biologists read. | The heatmap document and renderers exist; it needs amino-acid labels, the wild-type marked, and position ticks |
| **Sequence logo** of the predicted distribution at each position | What does the model expect here? Matches the familiar MSA logo | Missing: stacked letter glyphs (SVG and HTML) |
| **Contact map**, predicted vs. true, in triangles of one matrix | Has the model learned the fold? Where does it miss? | The heatmap exists; it needs a two-triangle overlay and top-L markers |
| **3D structure colored by a per-residue score**: relevance, log-odds, SAE feature activation, pLDDT | Where in the fold does the signal sit? Active sites, interfaces | Missing: an HTML view with Mol* or 3Dmol.js from a CDN (VIZ-3's CDN and inline pattern), structure embedded from PDB/mmCIF |
| **Residue tracks**: relevance, entropy, conservation and annotations stacked under the sequence (Nightingale-style) | Comparing several per-residue signals against domain and site annotations | Partly: the token relevance strip; tracks are missing |
| **Attention head grid** with each head's contact precision | Which heads track structure? | Missing (VIZ-6d) |
| **Embedding projector** (UMAP or PCA of per-protein embeddings, colored by family or function) | Does the model separate families or enzyme classes? | Missing (VIZ-6c) |
| **SAE feature dashboard**: top-activating proteins, with activations on sequence and structure | What concept does a feature encode? Following InterPLM.ai | Partly: `feature_dashboard.v1` and the HTML dashboard (VIZ-3); needs a structure panel |
| **Attribution graph over residues** | How does information flow between residues and layers? | Works once an encoder exists: VIZ-4's builder is generic over positions |

## What pulsatrix already has

The codebase survey found most of the stack in place.

- **Exists.**
  - Bidirectional attention (`AttentionConfig.causal` defaults to false), with padding masks.
  - RoPE in both layouts. ESM uses rotate-half.
  - Readable attention maps (`last_attention_weights`, N × heads × L × L).
  - `LayerNormModule`.
  - AttnLRP through attention and norms, with LXT parity.
  - A token cross-entropy loss with ignore-index -100, which is the masked-LM loss.
  - AdamW, warmup and cosine schedules, clipping, and safetensors checkpoints.
  - Sharded Hugging Face weight loading with mapping tables.
  - `CharModel` tokenizers.
  - Integrated gradients, saliency, SHAP and LIME.
  - A basic sparse autoencoder and linear probes.
  - Heatmap, token-relevance and attribution-graph documents, with SVG, HTML and Neuronpedia renderers.
- **Missing.**
  - An encoder block: today's `TransformerBlock` is fixed to RMSNorm and a gated MLP. ESM needs LayerNorm and a plain Linear → exact GELU (erf) → Linear MLP with biases.
  - An encoder model class with ESM's token-dropout scaling, a final LayerNorm and an LM head (dense, GELU, LayerNorm, tied decoder plus bias).
  - `HfModelConfig` accepting `EsmForMaskedLM`, plus the weight mapping.
  - A masked-LM masking collator.
  - The contact head.
  - An ESM `vocab.txt` tokenizer.
  - A FASTA reader.
  - A structure (PDB/mmCIF) reader for contacts and 3D views.
  - The protein views listed above.
- **Gating limits.**
  - fp32 only: bf16 training is HIP-11.
  - `DataLoader` workers aren't wired up.
  - Checked 2026-10-07: `facebook/esm2_t6_8M_UR50D` and `esm2_t33_650M_UR50D` ship `model.safetensors`, `config.json` and `vocab.txt` under the MIT license, so no pickle conversion is needed.

## Proposed epic: PLM

Each item is one PR, in dependency order. Effort: S (days), M (about a week), L (several weeks).

| ID | Item | Why | Depends on | P | Effort |
|---|---|---|---|---|---|
| PLM-1 | Encoder block options: LayerNorm or RMSNorm, a plain or gated MLP, exact erf GELU (with its LRP rule: identity, as AttnLRP treats elementwise nonlinearities), and attention and MLP biases | Every BERT-style encoder needs it, not just ESM | — | P0 | M |
| PLM-2 | `EncoderLM` and ESM-2 loading: token-dropout scaling, rotate-half RoPE, final LayerNorm, LM head. Accept `EsmForMaskedLM` configs and map weights. ESM `vocab.txt` tokenizer and FASTA reader. Golden parity of logits, hidden states and attentions with `transformers` on a tiny generated ESM, plus `esm2_t6_8M` and `esm2_t33_650M` | The foundation; it must match the reference to float precision before anything else | PLM-1 | P0 | M |
| PLM-3 | Variant scoring: masked-marginal, wild-type-marginal and pseudo-log-likelihood scores, and a full single-mutant scan. A ProteinGym runner (Spearman, NDCG, top-10% recall) that matches published ESM-2 numbers per assay on a subset (for example BLAT_ECOLX, GB1, and a dozen assays) | The headline use and the strongest numerical check | PLM-2 | P0 | M |
| PLM-4 | Contacts: ESM's contact head (symmetrize, APC, logistic regression) at parity with `transformers`, the top-K head-average method, a PDB/mmCIF reader (Cα/Cβ distances, 8 Å), and precision@L, L/2, L/5 by sequence separation | The structural check; also feeds the head grid | PLM-2 | P1 | M |
| PLM-5 | Protein views: <ul><li>mutational landscape (labeled L × 20 heatmap)</li><li>sequence logo (SVG and HTML)</li><li>contact map with predicted and true triangles</li><li>residue tracks</li><li>a 3D structure HTML view (Mol* or 3Dmol.js, CDN or inline, colored by any per-residue score)</li></ul> | The visuals biologists read; reuses VIZ-1 to VIZ-3 | PLM-3, PLM-4 | P1 | L |
| PLM-6 | Explaining encoders: <ul><li>AttnLRP for the encoder, targeting a masked position's logit, a mutation's log-odds, or a fine-tuned head</li><li>per-residue relevance documents</li><li>an attribution graph over residues (VIZ-4)</li><li>sanity suite: randomized weights, deletion curves, agreement with DMS per-position sensitivity and with conservation</li></ul> | Faithful, checked residue-level explanations, the gap the 2026 epitope study exposed | PLM-2, PLM-5 | P1 | M |
| PLM-7 | Masked-LM training: masking collator (15%, 80/10/10, token dropout), sequence cropping, cluster-weighted sampling. Train an ESM-2-8M-shaped model on a UniRef50 sample and check the loss and perplexity curve against BioNeMo's. Fine-tuning heads: per-residue (secondary structure) and per-protein (mean-pooled regression) | Training support and transfer learning; full-scale pretraining waits on HIP-11 | PLM-2 | P1 | M |
| PLM-8 | Probing and features: per-layer linear probes (DSSP, accessibility, binding sites) with control tasks; an InterPLM-style SAE on ESM-2-8M with features matched to Swiss-Prot annotations (F1 per concept); a feature dashboard with a structure panel | Concept-level interpretability; joins the FEAT epic | PLM-5, FEAT-1 | P2 | L |
| PLM-9 | Recipe and tutorial: on one protein (for example TEM-1 β-lactamase), scan all mutations, compare with its DMS, draw the contact map against its PDB structure, and show relevance on the 3D structure | Shows the whole path end to end | PLM-3 to PLM-6 | P1 | S |

Later, only if wanted: ESM C and AMPLIFY mappings; SaProt structure tokens; autoregressive pLMs
(ProGen2, which uses the existing CausalLM path); MSA-conditioned models; ESMFold. ESMFold is
large and a separate structure-model effort.

## Data and tools

All downloads are scripted, cached under `~/.cache/pulsatrix`, never committed and never needed
to build. CI uses tiny generated fixtures, the same way LLM-6 does.

| Need | Source | Size |
|---|---|---|
| Checkpoints | Hugging Face `facebook/esm2_t6_8M_UR50D` to `esm2_t33_650M_UR50D` (MIT) | 30 MB to 2.5 GB |
| Reference outputs | `transformers` in a container (`tools/golden/`) | small |
| Variant effects | ProteinGym DMS substitution CSVs and published model scores | about 1 GB (start with a subset) |
| Structures | RCSB PDB or AlphaFold DB, mmCIF per protein | KB per protein |
| Probing labels | DSSP from structures; TAPE/PEER secondary-structure sets | small |
| Annotations (SAEs) | UniProt Swiss-Prot features | about 1 GB |
| Training sample | UniRef50 FASTA (subsample) | full file is about 10 GB; use a 1–5% sample |

## Decisions

- PLM is its own epic, started after VIZ-4.
- PLM-1 (the encoder block) comes first, then PLM-2.
- Downloading ProteinGym and UniProt data to `~/.cache/pulsatrix` is fine.

## Sources

- Lin et al., *Science* 2023 (ESM-2, ESMFold); [Hugging Face ESM docs](https://huggingface.co/docs/transformers/en/model_doc/esm); [NVIDIA BioNeMo ESM-2 recipe](https://docs.nvidia.com/bionemo-recipes/latest/models/ESM-2/)
- [Protein Language Models in 2026](https://rewirebio.io/blog/protein-language-models-in-2026/): model landscape, licensing, evaluation pitfalls
- [Scaling and data saturation in pLMs](https://arxiv.org/abs/2507.22210); [Training compute-optimal pLMs](https://icml.cc/virtual/2024/35912)
- [ProteinGym](https://www.biorxiv.org/content/biorxiv/early/2023/12/08/2023.12.07.570727.full.pdf); [MSA Pairformer](https://www.sciencedirect.com/science/article/pii/S009286742600749X); [VespaG](https://academic.oup.com/bioinformatics/article/40/11/btae621/7907184)
- [Contacts from attention in one forward pass](https://arxiv.org/abs/2606.21876)
- [InterPLM, *Nature Methods*](https://www.nature.com/articles/s41592-025-02836-7) ([code](https://github.com/ElanaPearl/interPLM))
- [Residue-level attributions don't recover allergen epitopes](https://arxiv.org/abs/2606.22181); [Toward the explainability of pLMs](https://arxiv.org/abs/2506.19532)
- [ESM-2 mutation scoring walkthrough](https://huggingface.co/blog/AmelieSchreiber/mutation-scoring); [MutAmore](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05610-8)
- [Structure-aligned pLM](https://arxiv.org/pdf/2505.16896); [Predictive and therapeutic applications of pLMs](https://www.sciencedirect.com/science/article/pii/S1323893025000875)
