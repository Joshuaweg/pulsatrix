# Mechanistic Interpretability

[Interpretability](../interpretability/index.md) tells you which parts of an *input* mattered
for a prediction. Mechanistic interpretability asks how the *network* computes it:

- what its layers represent;
- which directions in its activations stand for concepts;
- which of its weights do the work;
- how information flows from the prompt to the answer.

This section covers the tools pulsatrix has for that, from reading activations to full
attribution graphs of a language model's prediction.

## Which tool should I use?

| Your question | Tool | Page |
|---|---|---|
| Is concept X linearly readable from layer L? | `LinearProbe`, `TrainLinearProbe()`, with `MeasureSelectivity()` or `CompareToBaseline()` as the control | [Probes and concepts](probes-and-concepts.md#probing-a-layer-for-a-concept) |
| How similar are two layers, two models, or a model before and after fine-tuning? | `LinearCka`, `LinearCkaMatrix` | [Representations](representations.md#comparing-representations-cka) |
| Does a model order the inputs as a brain recording or a model of the categories does? Is the agreement chance? | `RepresentationalDissimilarity`, `CompareRdms`, `RsaPermutationTest` (RSA) | [Representations](representations.md#rsa-and-svcca) |
| Which directions do two layers share? Reproduce a result reported with SVCCA. | `Svcca`, `CanonicalCorrelations` | [Representations](representations.md#rsa-and-svcca) |
| Which inputs does a layer treat as alike? Do two models agree on it? | `NearestNeighbors`, `MutualKnnAlignment`, `MutualNearestNeighbors`, `MeasureHubness` | [Representations](representations.md#nearest-neighbors) |
| Is cosine similarity meaningful in this layer? How many dimensions does it use? | `Anisotropy`, `MeasureEffectiveRank`, `TwoNNDimension` | [Representations](representations.md#the-shape-of-a-layer) |
| What does a layer look like in 2D or 3D? Which directions vary most? | `FitPca`, `PcaProject`, `PcaSplitHalfOverlap`, `RenderProjectionSvg` | [Representations](representations.md#pca-projections) |
| How do I look at a layer with UMAP or t-SNE? | `ExportEmbeddings`, `WriteNpy`, `ReadNpy`, `MakeProjectionDocument(coordinates, "umap")` | [Representations](representations.md#umap-and-t-sne) |
| How do I turn a layer's activations into points for topological data analysis? | `PointCloud`, `MakePointCloud` | [Topology](topology.md#from-activations-to-a-point-cloud) |
| What are the distances between those points, and how far off can float32 make them? | `PairwiseDistances`, `DistanceMatrix::error_bound` | [Topology](topology.md#distances-with-an-error-bar) |
| Which clusters does a layer form, and over what range of scales do they stay apart? | `ComputeH0`, `SingleLinkageClusters` | [Topology](topology.md#clusters-across-scales-h0) |
| Does a layer arrange inputs in loops or around voids? Which inputs form them? | `ComputeRipsPersistence`, critical edges, cocycles | [Topology](topology.md#loops-and-voids-h1-h2-and-up) |
| Does a concept, given by examples, push the model towards a class? | `RunTcav`, `TrainCav` | [Probes and concepts](probes-and-concepts.md#concepts-tcav) |
| Which of a layer's channels does an LRP explanation pass through, and where in the input is each? | `ChannelRelevance`, `ConditionalHeatmaps`, `ConceptReferenceSamples` (CRP) | [LRP](../interpretability/lrp.md#concepts-concept-relevance-propagation) |
| Which directions in a layer stand for concepts, found without labels? | Sparse autoencoders: `TopKSparseAutoencoder`, `JumpReLUSparseAutoencoder`, `SparseAutoencoder` | [Finding features](featurizers.md) |
| Which concepts take more than one dimension, such as an angle or a position? | `BlockSparseFeaturizer` | [Finding features](featurizers.md#block-sparse-featurizers) |
| What does an MLP compute, step by step? | `Transcoder` | [Finding features](featurizers.md#transcoders) |
| Are my features any good? | `EvaluateReconstruction`, `MeasureLossRecovered`, `FeatureAbsorption`, with baselines | [Evaluating features](evaluating-features.md) |
| Can a direction push the model's behavior, and on how many inputs does it work? | `SteeringHook`, `MeasureSteering` | [Steering](steering.md) |
| What did fine-tuning change in a model? | `Crosscoder`, `MeasureLatentScaling`, `pulsatrix_diff_lm` | [Model diffing](model-diffing.md) |
| Which pieces of the weights does each input use? | `ParameterDecomposition` (SPD, VPD) | [Parameter decomposition](parameter-decomposition.md) |
| Which features carry a language model's prediction, layer to layer? | `TraceCircuit`, `pulsatrix_explain_text --transcoders` | [Attribution graphs](attribution-graphs.md) |
| Which layers does a small network's output depend on? | `ExplainerContext::build_circuit_graph()` | [Attribution graphs](attribution-graphs.md#ablation-circuit-graphs-for-small-networks) |
| What happens if I overwrite one activation? | `ExplainerContext::forward_pass_with_patch()` | [Below](#reading-and-changing-activations) |
| What would the model predict if layer L were its last? | `ExplainerContext::logit_lens()` | [Below](#reading-and-changing-activations) |
| How do I sample outcomes in proportion to a reward? | GFlowNets | [GFlowNets](gflownets.md) |

## A typical workflow

For a researcher starting on a model, the pieces fit together like this:

1. **Collect activations** from the layer you care about, with a hook (below).
2. **Describe the layer**: how many dimensions it uses, whether cosine similarity means anything
   in it, and how it compares with other layers and models ([Representations](representations.md)).
3. **Probe** for the concepts you already have labels for, with a control. A probe tells you
   whether the information is there ([Probes and concepts](probes-and-concepts.md)).
4. **Train a featurizer** (a TopK or BatchTopK SAE is a good default) to find the directions the
   model uses without being told what to look for.
5. **Evaluate it against baselines**: the same featurizer on a randomly initialized copy of the
   model, and a linear probe. A feature that also shows up in a random model says nothing about
   what the trained one learned.
6. **Test causally.**
   - Splice the reconstructions back in and see how much of the loss you keep (loss
     recovered).
   - Steer along a direction, with a random-direction control.
   - For one prediction, trace an attribution graph.

Each page reports what these steps found on real models (ESM-2 protein models, SmolLM2,
Gemma 3), including results that didn't work out. Read those before relying on a method.

## Building blocks

### Reading and changing activations

Language models expose two hooks, used throughout this section:

- **`HiddenStateHook`** on `CausalLM` and `EncoderLM`. It is called with each residual-stream
  position during `forward()`: 0 for the embeddings, i for the output of block i. It returns the
  tensor the model continues with. Return it unchanged to only read.
- **`MlpHook`** on each `TransformerBlock` and `EncoderBlock`. It sees an MLP's input (after the
  block's norm) and its output, and returns what is added to the residual stream.

```cpp
#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/tokenizer_json.hpp"

using namespace pulsatrix;

CPUBackend backend;
std::unique_ptr<CausalLM> model = LoadCausalLM("models/SmolLM2-135M", &backend);
const TextTokenizer tok = LoadTokenizerJson("models/SmolLM2-135M/tokenizer.json");
const int64_t d = model->config().hidden_size;

// corpus: your texts, a std::vector<std::string>.
// Collect the residual stream after block 6, one row per token.
std::vector<float> rows;
model->set_hidden_state_hook([&](int64_t position, const Tensor& h) {
    if (position == 6) {
        const std::vector<float> v = h.to_host_vector();   // (1, tokens, d)
        rows.insert(rows.end(), v.begin(), v.end());
    }
    return h;                                              // unchanged: only read
});
for (const std::string& text : corpus) {
    const std::vector<int64_t> ids = tok.encode(text).ids;
    const std::vector<float> ids_f(ids.begin(), ids.end());
    (void)model->forward(Tensor(Shape({1, static_cast<int64_t>(ids.size())}), &backend, ids_f));
}
model->set_hidden_state_hook({});
const Tensor activations(Shape({static_cast<int64_t>(rows.size()) / d, d}), &backend, rows);
```

For small networks built from layers, `ExplainerContext` caches every layer's output and offers
the classic tools:
- `activation_snapshot()`: a copy of one forward pass's activations that stays valid after
  later passes.
- `forward_pass_with_patch()`: activation patching, overwriting one activation.
- `logit_lens()`: what the model would predict if a given layer were the last.
- `attention_weights()`.

### Baselines

`NullModelBaseline` re-initializes a model randomly, so the same probe, featurizer or
explanation can run on it ([Interpretability](../interpretability/index.md#the-null-model-baseline)).
It is the single most useful check in this section. In the ESM-2 results on the following
pages, a random model's SAE explains *more* variance than the trained model's, and some
"concept features" appear in it too.

## Command-line tools

| Tool | What it does | Page |
|---|---|---|
| `pulsatrix_probe_esm` | Probes, SAEs, transcoders and block-sparse featurizers on an ESM-2 layer, with concept matching, absorption, loss recovered and random-model baselines | [Finding features](featurizers.md#on-a-protein-model-pulsatrix_probe_esm), [Protein language models](../protein-models/index.md) |
| `pulsatrix_steer_esm` | A steering vector on ESM-2, with the reliability report | [Steering](steering.md) |
| `pulsatrix_diff_lm` | A crosscoder between a base language model and its fine-tune, with Latent Scaling | [Model diffing](model-diffing.md) |
| `pulsatrix_spd_toy` | Parameter decomposition (SPD, VPD) of a toy model of superposition | [Parameter decomposition](parameter-decomposition.md) |
| `pulsatrix_explain_text --graph` | An attribution graph for a prompt, from AttnLRP or (with `--transcoders`) transcoder features, for Neuronpedia's viewer | [Attribution graphs](attribution-graphs.md) |

Each prints its usage with `--help`. Each source file (in `tools/`) starts with a comment that
explains every option.

## Recipes

- [Sparse autoencoder + linear probe](../recipes/mechanistic-interpretability/sparse_autoencoder_probe.md)
- [GFlowNet on HyperGrid](../recipes/mechanistic-interpretability/gflownet_hypergrid.md)

Full API reference: [Doxygen: Mechanistic Interpretability](../api/group__mech__interp.html)
