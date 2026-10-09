# Finding Features

A layer's activations mix many concepts at once, more than it has dimensions (**superposition**).
A **featurizer** pulls them apart without being told what to look for. It rewrites each
activation `x` as a few active **features** over a large learned dictionary, and rebuilds `x`
from them:

- `f = encode(x)`: sparse codes, mostly zero;
- `x ≈ decode(f)`: each active feature adds its decoder direction, scaled by its code.

If training works, individual features line up with interpretable concepts, such as "a
transmembrane residue" in a protein model or "inside a quotation" in a language model. Whether
they do has to be checked: see [Evaluating features](evaluating-features.md).

## Choosing a featurizer

All of these share one interface (`featurizer.hpp`), so training, metrics, steering and
concept matching work the same for each.

| Featurizer | Class | Use it when | Notes |
|---|---|---|---|
| TopK SAE (Gao et al.) | `TopKSparseAutoencoder` | **The default.** You want sparse features with a sparsity you set directly | Exactly k features per input; codes aren't shrunk toward zero |
| BatchTopK SAE (Bussmann et al.) | `TopKSparseAutoencoder`, `batch_topk = true` | Inputs differ in how much they contain | k on average; more for busy inputs, fewer for simple ones |
| Matryoshka SAE (Bussmann et al.) | `TopKSparseAutoencoder`, `matryoshka_prefixes` | You want clean, general features and less **absorption** | Slightly worse reconstruction |
| JumpReLU SAE (Rajamanoharan et al.) | `JumpReLUSparseAutoencoder` | You'd rather the model choose how many features each input needs | Sparsity set indirectly, by a penalty λ |
| L1 SAE | `SparseAutoencoder` | Comparing with older work | Its penalty shrinks every code toward zero; worse than TopK at the same sparsity |
| Block-sparse featurizer (Fel et al.) | `BlockSparseFeaturizer` | A concept may take several dimensions: an angle, a position, a color | Each feature is a small subspace |
| Transcoder (Dunefsky et al.; Paulo et al.) | `Transcoder` | You want to explain what an **MLP computes**, not what a layer holds | Reads the MLP's input, predicts its output |
| Crosscoder (Lindsey et al.) | `Crosscoder` | One dictionary across several layers or two models | See [Model diffing](model-diffing.md) |

Terms used below:
- **Feature** (papers and the API also say **latent**): one dictionary entry, with its code and
  its decoder direction.
- **L0**: the number of active features per input. Lower is sparser.
- **Explained variance**: the share of the activations' variance the reconstructions keep.
- **Dead feature**: one that never fires. It never learns, so it wastes capacity.
- **Dense feature**: one that fires on more than 10% of inputs. It usually encodes a bias, not a
  concept.
- **Shrinkage**: codes smaller than they should be, so reconstructions come out too short. An L1
  penalty causes it.
- **Absorption**: a general feature ("starts with S") goes silent where a more specific one
  ("short") fires instead.

## Training

Every featurizer trains the same way: one optimizer step per batch through `TrainFeaturizer`.

```cpp
#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/featurizer.hpp"
#include "pulsatrix/topk_sparse_autoencoder.hpp"

TopKSaeOptions options;
options.k = 32;                       // active features per input
TopKSparseAutoencoder sae(/*dim=*/576, /*num_features=*/8 * 576, &backend, options);
sae.initialize_bias(first_batch);     // start the decoder bias at the data's mean
AdamOptimizer opt(1e-3f, &backend);                // used by the snippets below too
FeatureActivityTracker activity(sae.num_features());

for (const Tensor& batch : batches) {            // each (N, 576)
    const FeaturizerLoss loss = TrainFeaturizer(sae, batch, opt, /*unit_norm_decoder=*/true, &activity);
    // loss.reconstruction, loss.sparsity, loss.total
}
const std::vector<float> codes = sae.encode(held_out).to_host_vector();   // (N, num_features)
const std::vector<float> direction = sae.decoder_direction(/*feature=*/17);
const double l0 = MeanL0(sae, held_out);
const std::vector<int64_t> dead = activity.dead(/*window=*/1'000'000);
```

Practical advice:
- **Collect enough data.** Activations come from a hook
  ([Reading and changing activations](index.md#reading-and-changing-activations)). Use many
  inputs; a dictionary with thousands of features needs hundreds of thousands of rows.
- **Standardize or center the activations** first, and normalize them the same way at
  evaluation. A few dimensions with huge values (common in language models) otherwise dominate
  the loss.
- **Width.** 4 to 16 times the layer's dimension is the usual range.
- **Track dead and dense features.** `FeatureActivityTracker` counts how often each feature fires
  and when it last did. Report both with every featurizer.
- **Featurizers are Modules.** They map input to reconstruction, with `backward()`, LRP and
  named parameters, so checkpoints, optimizers and `SaveCheckpoint` work as for any model.
  Save the dead-feature counters too (`inputs_since_fired()`, restored with
  `set_inputs_since_fired()`), or every feature starts out alive again when you resume.

## Sparse autoencoders

### TopK and BatchTopK

`TopKSparseAutoencoder` keeps each input's k largest pre-activations and zeroes the rest, so L0
is k by construction, and there is no penalty pulling codes toward zero.

```cpp
TopKSaeOptions o;
o.k = 16;
o.batch_topk = true;               // BatchTopK: the batch's N·k largest, wherever they fall
TopKSparseAutoencoder sae(320, 4096, &backend, o);
```

- **The auxiliary loss (AuxK) keeps features alive.** A feature outside every input's top k
  gets no gradient, and stays dead. AuxK has the largest dead features (`k_aux`, by default half
  the input dimension) reconstruct what the main reconstruction misses, weighted by
  `aux_coefficient` (1/32). A feature counts as dead after `dead_after` inputs without firing.
- **BatchTopK at inference.** Training selects across the batch. At inference each input's
  codes would then depend on its batch-mates, so BatchTopK switches to a threshold learned in
  training (`threshold()`, saved in checkpoints).
- **What it buys.** On synthetic data, BatchTopK uses 1.1 features on one-direction inputs and
  4.6 on five-direction inputs, at an average of 3.

### Matryoshka

`matryoshka_prefixes` trains nested dictionaries at once. The loss adds the reconstruction error
of each prefix of the dictionary you list, for example the first `m/16` features, the first
`m/4`, and all `m`. The first features must work alone, so they learn general concepts, and
later ones refine them.

```cpp
TopKSaeOptions o;
o.k = 16;
o.batch_topk = true;
o.matryoshka_prefixes = {256, 1024};   // 4096, the full size, is added
TopKSparseAutoencoder matryoshka(320, 4096, &backend, o);
```

On a toy hierarchy (a parent concept with ten rare children), plain BatchTopK absorbs the parent
into its children on 6.6% to 12.6% of inputs, and Matryoshka on none. On ESM-2 it also gives the
cleanest concept features ([results](evaluating-features.md#comparing-featurizers-on-esm-2)).

### JumpReLU

`JumpReLUSparseAutoencoder` learns a threshold per feature, and the loss penalizes L0 directly
with weight λ (`l0_coefficient`). The step function's gradient comes from straight-through
estimators.

```cpp
#include "pulsatrix/jumprelu_sparse_autoencoder.hpp"

JumpReLUSaeOptions j;
j.l0_coefficient = 0.01f;      // λ: higher is sparser
JumpReLUSparseAutoencoder jump(320, 4096, &backend, j);
```

You don't set L0; you set λ and measure the result. On ESM-2 8M, λ = 0.005 gave L0 12.6,
0.01 gave 7.6, and 0.02 gave 4.9. Too little penalty spreads each concept over several features.

### L1

`SparseAutoencoder(dim, hidden_dim, l1_lambda, &backend)` is the original: ReLU codes with an
L1 penalty. The penalty shrinks every code toward zero, which costs reconstruction: at the same
L0, TopK beats it on every measure in the ESM-2 comparison. Keep decoders at unit norm
(`unit_norm_decoder = true`, the default). Otherwise the penalty is mostly paid by shrinking
codes and growing decoders, not by using fewer features.

## Block-sparse featurizers

Some concepts aren't one direction. A position on a circle, a hue or an angle lives on a
low-dimensional surface. An SAE has to spread such a concept over several features. A
**block-sparse featurizer** (BSF; Fel et al.) makes each feature a block of b directions:

- the block's norm says how strongly the concept is present;
- its b coordinates say where on the concept's surface the input is.

```cpp
#include "pulsatrix/block_sparse_featurizer.hpp"

BlockSparseOptions o;
o.variant = BsfVariant::Vanilla;    // or Grassmannian, GroupLasso
o.block_size = 4;                   // b
o.k = 4;                            // active blocks per input
BlockSparseFeaturizer bsf(/*dim=*/320, /*num_blocks=*/640, &backend, o);
// Center the activations, and scale them so the mean squared norm is the dimension.
FeaturizerLoss loss = TrainFeaturizer(bsf, batch, opt);

std::vector<float> frame = bsf.block_frame(/*block=*/12);           // b × d
BlockGeometry geometry = MeasureBlockGeometry(bsf, held_out);       // dimensions used per block
DescriptionLength bits = MeasureDescriptionLength(bsf, held_out);   // bits per input
```

| Variant | Encoder | Which blocks fire |
|---|---|---|
| Vanilla | Free | The k of largest norm |
| Grassmannian | Tied to orthonormal frames | The k of largest norm |
| GroupLasso | Free | Those above a learned threshold; a penalty tuned to hold `target_l0` on average |

- **Codes are signed, and a feature is a block.** L0, dead and dense counts work in blocks.
- **Tournament selection** (`o.selection = BlockSelection::Tournament`; Jerpelea and Ananthram)
  skips a block that overlaps one already chosen, to stop one concept splitting over two
  blocks.
- **Two measures from the paper.**
  - `MeasureBlockGeometry`: how many of a block's b dimensions its codes actually use (stable
    rank, participation ratio).
  - `MeasureDescriptionLength`: the bits needed to describe an input at a fixed distortion. A
    featurizer that finds the right structure describes data more compactly.

On planted circles in random planes, a Vanilla BSF with blocks of 2 recovers every plane and
needs 19.9 bits per input, where a TopK SAE with the same 4 active dimensions needs 32.4. On
ESM-2 8M the gain mostly disappears: its blocks use about 1.5 of their 4 dimensions
([results](evaluating-features.md#comparing-featurizers-on-esm-2)).

## Transcoders

An SAE explains what is *in* an activation. A **transcoder** explains what an MLP *does*: it
predicts the MLP's output from the MLP's input through sparse features, so each feature reads
from the input side and writes to the output side. A **skip transcoder** (Paulo et al.) adds a
linear map from input to output, which takes the MLP's linear part; the features then explain
only what is left.

```cpp
#include "pulsatrix/featurizer_metrics.hpp"
#include "pulsatrix/transcoder.hpp"

// Training pairs: the MLP's input and output, read with the block's hook.
std::vector<float> ins, outs;
model->layer(3).set_mlp_hook([&](const Tensor& in, const Tensor& out) {
    const std::vector<float> a = in.to_host_vector(), b = out.to_host_vector();
    ins.insert(ins.end(), a.begin(), a.end());
    outs.insert(outs.end(), b.begin(), b.end());
    return out;                               // unchanged: only read
});
// ... run the model over your data, then remove the hook:
model->layer(3).set_mlp_hook({});

// inputs, targets: (rows, d) Tensors built from ins and outs.
TranscoderOptions o;
o.k = 16;
o.skip = true;
Transcoder t(/*input_dim=*/d, /*output_dim=*/d, /*num_features=*/8 * d, &backend, o);
t.initialize_bias(targets);
FeaturizerLoss loss = TrainFeaturizer(t, inputs, targets, opt);   // inputs → targets

ReconstructionMetrics fit = EvaluatePrediction(t, held_inputs, held_targets);
```

- **`predict(x)` vs `decode(codes)`.** `predict` is the full output, skip included; `decode` is
  the features' part alone.
- **Splicing a transcoder into the model.** `MlpSpliceHook(t)` replaces the MLP's output with the
  transcoder's prediction, and `MeasureMlpLossRecovered` reports how much of the model's loss
  survives ([Evaluating features](evaluating-features.md#loss-recovered)).
- **Transcoders for attribution graphs** are a separate, inference-only class
  (`CrossLayerTranscoder`) that loads pretrained weights: see
  [Attribution graphs](attribution-graphs.md). A `Transcoder` trained here can't be used there
  directly: attribution graphs need JumpReLU or ReLU features, and this one uses TopK.

On ESM-2 8M the skip connection helps everywhere: it predicts the MLP better (explained
variance 0.81 vs 0.67), recovers more of the loss (0.72 vs 0.59) and leaves fewer features dead
(8% vs 25%).

## Inspecting a feature

To see what a feature means, look at the inputs where it fires most strongly.

- **Collect top examples.** `encode()` a held-out set and sort by a feature's code.
- **Save a dashboard.** Fill a `FeatureDashboardDocument`: activation density, a histogram, and
  the top examples with per-token activations. `RenderFeatureDashboardHtml` or
  `pulsatrix_svg feature.json -o feature.html` turns it into a page
  ([Visualization](../visualization/index.md)).
- **Match labeled concepts.** For concepts with labels, `EncodeSparse` and `MatchConcept`
  (`protein_concepts.hpp`) score every feature against a concept by its best F1 over activation
  thresholds. They work on any featurizer, not only protein models.

## On a protein model: pulsatrix_probe_esm

`pulsatrix_probe_esm` runs this whole page on an ESM-2 layer:
1. It embeds Swiss-Prot proteins.
2. It trains a featurizer, with the same featurizer on a randomly initialized ESM-2 as the
   baseline.
3. It reports every measure in [Evaluating features](evaluating-features.md).
4. It matches every feature against per-residue annotations, and writes a dashboard for each
   concept's best feature, drawn on its protein's 3D structure.

```bash
python3 tools/plm/fetch_swissprot_annotations.py data/swissprot
mkdir -p models/esm2_t6_8M_UR50D
for f in config.json model.safetensors vocab.txt; do
  curl -fsSL https://huggingface.co/facebook/esm2_t6_8M_UR50D/resolve/main/$f -o models/esm2_t6_8M_UR50D/$f
done
pulsatrix_probe_esm models/esm2_t6_8M_UR50D --annotations data/swissprot/annotated.jsonl \
    --sae --sae-layer 4 --featurizer topk --k 16 --out sae/
```

| Option | Default | Meaning |
|---|---|---|
| `--featurizer` | `l1` | `l1`, `topk`, `batchtopk`, `matryoshka`, `jumprelu`, `bsf-vanilla`, `bsf-grassmannian`, `bsf-lasso`, `transcoder`, `skip-transcoder`. **Pass `topk`**: `l1`, the default, is the setup of the [protein guide](../protein-models/index.md)'s first SAE results |
| `--sae-layer` | two-thirds up | The layer whose residues are featurized (layer 4 of ESM-2 8M) |
| `--features`, `--k` | 8 × hidden, 32 | Dictionary size, and active features per residue (or blocks, for BSF) |
| `--l0`, `--l1`, `--block-size` | 0.05, 0.003, 4 | JumpReLU's λ, the L1 penalty, BSF's block size |
| `--sae-proteins`, `--sae-epochs` | 2,000, 10 | Training data and length |
| `--probes`, `--layers` | | Linear probes at every layer (or the listed ones), instead of or beside `--sae` |
| `--structures DIR` | | AlphaFold models, to draw the dashboards in 3D |
| `--device` | `cpu` | `hip` for an AMD GPU |

A TopK run on ESM-2 8M takes a few minutes on a GPU.

## How the implementations are checked

Each featurizer matches a PyTorch rendering of its paper through the loss, every gradient and
Adam steps, to about 1e-5:

| Featurizer | Golden script |
|---|---|
| TopK, AuxK | `tools/golden/make_topk_sae_golden.py` |
| BatchTopK, Matryoshka, JumpReLU | `tools/golden/make_sae_variants_golden.py` |
| Block-sparse (all three variants) | `tools/golden/make_bsf_golden.py` |
| Transcoder, skip transcoder | `tools/golden/make_transcoder_golden.py` |

**Choices where a paper and its code differ.**
- **BSF follows its reference code** (github.com/goodfire-ai/block-sparse-featurizer): a scale
  per block, GroupLasso as a block JumpReLU, and QR re-orthonormalization after each step.
- **TopK rescales its decoder directions to unit norm without moving the scale into the
  encoder.** Moving it would change which features win the top k.

## References

- Gao et al., "Scaling and evaluating sparse autoencoders", arXiv 2406.04093 (TopK, AuxK).
- Bussmann et al., "BatchTopK Sparse Autoencoders", arXiv 2412.06410; "Learning Multi-Level
  Features with Matryoshka Sparse Autoencoders", arXiv 2503.17547.
- Rajamanoharan et al., "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse
  Autoencoders", arXiv 2407.14435.
- Fel et al., "Structuring Sparsity: Block-Sparse Featurizers", arXiv 2606.25234; Jerpelea and
  Ananthram, arXiv 2608.27515 (tournament top-k).
- Dunefsky et al., "Transcoders Find Interpretable LLM Feature Circuits", arXiv 2406.11944;
  Paulo et al., "Transcoders Beat Sparse Autoencoders for Interpretability", arXiv 2501.18823.
