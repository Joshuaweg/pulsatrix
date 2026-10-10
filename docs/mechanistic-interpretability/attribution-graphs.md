# Attribution Graphs

An attribution graph explains **one prediction** as a graph: nodes for the pieces of the
computation, and weighted edges for how much each piece contributed to the next. pulsatrix
builds three kinds. They answer different questions:

| Graph | Nodes | Edges | Use it for | Cost |
|---|---|---|---|---|
| **Transcoder attribution graph** (circuit tracing) | Interpretable features at each layer and token, plus error nodes | Exact attributions through a frozen, linearized model | *Which concepts* carry a language model's prediction, and how they build on each other | Pretrained transcoders for the model; seconds on a GPU |
| **AttnLRP relevance graph** | The residual stream at each layer and token | AttnLRP relevance passed between them | *Where* information flows: which tokens, through which layers | Any supported language model; no extra weights |
| **Ablation circuit graph** | Each layer of a small network | How much zeroing a layer moves the output | *Which layers* a small network's output depends on | One forward pass per node |

The first two are written in the JSON format that
[Neuronpedia's graph viewer](https://www.neuronpedia.org/graph/validator) and Anthropic's
[circuit-tracer](https://github.com/decoderesearch/circuit-tracer) read. You get an interactive
viewer with no extra code ([Viewing a graph](#viewing-a-graph)).

## Transcoder attribution graphs

Following Ameisen, Lindsey et al. ("Circuit Tracing", 2025) and circuit-tracer:

1. **Replace each MLP with transcoder features.** A transcoder reads an MLP's input and predicts
   its output through sparse, interpretable features. Whatever it misses becomes an **error
   node**, so the model's output is unchanged.
2. **Freeze the rest.** Attention patterns and normalization scales are fixed at their values
   for this prompt. What is left between features is linear.
3. **Attribute.** Because it is linear, the contribution of every node to every later node is
   exact: the source's output, carried through the frozen model, dotted with the target's input
   direction.
4. **Prune** to the nodes and edges that matter most for the prediction.

The nodes are:
- active features, by layer and token;
- one error node per layer and token;
- the token embeddings;
- the likely next tokens (up to 10, until their probabilities sum to 0.95).

### Quick start

You need transcoders trained for your model. For Gemma 3 270M, Google's Gemma Scope 2 has
per-layer transcoders in the layout circuit-tracer reads. (`google/gemma-3-270m` asks you to
accept a license; the `unsloth` copy downloads without a token.)

```bash
hf download mwhanna/gemma-scope-2-270m-pt --include "transcoder_all/width_16k_l0_small/*" --local-dir transcoders/
hf download unsloth/gemma-3-270m --include "*.json" --include "*.safetensors" --local-dir models/gemma-3-270m

pulsatrix_explain_text models/gemma-3-270m "The Eiffel Tower is located in the city of" \
    --graph eiffel.json --viewer-dir graphs/ \
    --transcoders transcoders/transcoder_all/width_16k_l0_small --scan gemma-3-270m --device hip
```

The tool prints:
- how many features were active;
- the graph's **replacement** and **completeness** scores (below);
- the pruned graph's size.

`--node-threshold` (default 0.8) and `--edge-threshold` (default 0.98) set how much is pruned.

In C++:

```cpp
#include "pulsatrix/circuit_tracing.hpp"

// ids: the prompt's token ids (tokenizer.encode(text).ids).
CrossLayerTranscoder clt = LoadTranscoders(dir, model->num_layers(), &backend);
CircuitTraceOptions options;     // which logits to explain, pruning thresholds, the viewer's labels
CircuitTrace trace = TraceCircuit(*model, clt, ids, options);   // every node and edge, unpruned
CircuitScores scores = ScoreCircuit(trace);                     // replacement, completeness

// logit_labels: the explained tokens as text, for the viewer (tokenizer.decode({id})).
AttributionGraph graph = ToAttributionGraph(trace, options, logit_labels);   // pruned
std::string json = ToNeuronpediaJson(graph);
```

`CircuitTrace` holds the full adjacency matrix (`adjacency[target · n + source]`), the active
features with their activations, and each target's value, if you want to analyze the graph
yourself.

### Which transcoders work

`LoadTranscoders` reads circuit-tracer's two layouts:
- **per-layer transcoders**, a set of `layer_<i>.safetensors`;
- **cross-layer transcoders** (CLTs), whose features write to every later MLP, as
  `W_enc_<i>` and `W_dec_<i>` files.

Both may use JumpReLU or ReLU, and a linear skip.

| Model | Transcoders | Kind | Size |
|---|---|---|---|
| Gemma 3 270M | `mwhanna/gemma-scope-2-270m-pt`, `transcoder_all/width_16k_l0_small` | Per-layer | 1.5 GB |
| Gemma 3 270M | `google/gemma-scope-2-270m-pt`, `clt/...` | Cross-layer | ~13 GB |
| Llama 3.2 1B | `mntss/clt-llama-3.2-1b-524k` | Cross-layer | 25 GB |
| Qwen3 0.6B | `mwhanna/qwen3-0.6b-transcoders-lowl0` | Per-layer | ~19 GB |

Only the first was run end to end. Features read the MLP's input *after* the block's norm, where
Gemma's and Qwen3's transcoders read. The Llama 3.2 CLT reads the residual stream *before* the
norm, which isn't supported yet.

### Reading a graph

- **Edges** are signed. A positive edge raised the target, a negative one lowered it. Edges into
  a feature measure its pre-activation; edges into a logit measure the logit minus the mean
  logit.
- **Error nodes** are where the transcoders fall short. Large edges from error nodes mean part of
  the computation isn't explained by features.
- **Influence.** A node's influence is how much it affects the predicted tokens through every
  path, direct or through other nodes. Edges are normalized so each node's incoming weights sum
  to 1; influence then adds up the paths, weighted by the tokens' probabilities.
- **Replacement score:** the share of the prompt tokens' influence on the output that flows
  through features rather than errors.
- **Completeness score:** the influence-weighted share of every node's inputs that come from
  features and tokens rather than errors.
- **What's typical.** Both are 1 for a perfect transcoder. The Gemma 3 graphs below score about
  0.67 and 0.92: most of the computation is explained by features, but a third of the tokens'
  influence still passes through error nodes.
- **Pruning** keeps the nodes holding 80% of the influence on the predicted tokens, and the edges
  holding 98%. It then drops nodes left without edges, as circuit-tracer does.
- **The first token** (BOS) gets no features or errors; its MLP outputs count as constants.

### What we found on Gemma 3 270M

With Gemma Scope 2's 16k per-layer transcoders, on the GPU:

| Prompt | Top prediction | Active features | After pruning | Replacement | Completeness |
|---|---|---|---|---|---|
| "The Eiffel Tower is located in the city of" | " Paris" (0.86) | 4,532 | 621 nodes, 61,219 edges | 0.67 | 0.92 |
| "The capital of the state containing Dallas is" | " a" (0.13) | 3,779 | 634 nodes, 84,485 edges | 0.66 | 0.92 |

Each graph takes about 14 s on a Radeon 8060S GPU, or 31 s on a 16-core CPU with a peak of 5.7 GB
of memory, and the two give the same graph.

- **" Paris" comes from a handful of late features at the last token,** in layers 11 to 17.
  Error nodes and the " Eiffel" token embedding also feed it directly.
- **The 270M model doesn't know the two-hop fact** in circuit-tracer's demo prompt. It
  predicts " a", " the" or " Dallas", so that graph shows the method, not a circuit for
  "Austin". Use a larger model for multi-hop reasoning.

### Limitations

- **Pretrained transcoders only.** Training a transcoder for attribution graphs isn't
  implemented. The `Transcoder` featurizer trains one per layer, but its features are TopK, and
  attribution graphs need JumpReLU or ReLU features.
- **No `max_feature_nodes`.** Every active feature becomes a node. At 270M that is about 4,500
  features and a dense adjacency of 20M entries, which is fine; larger models would need
  circuit-tracer's cap.
- **Gemma's final logit softcap** is outside the linearization.
- **Attention is frozen, not explained.** As in the paper, the graph doesn't say why attention
  looked where it did.

### How it's checked

A PyTorch golden (`tools/golden/make_circuit_golden.py`) writes the tiny Llama and Gemma 3 test
models in PyTorch with circuit-tracer's freezes. It scales every source by a scalar and takes
the edges from autograd. **Every edge matches to 2e-6.** That covers:
- a cross-layer span and a skip;
- Gemma's sandwich norms, QK-norm and sliding window.

The CPU and GPU results agree.

## AttnLRP relevance graphs

Without transcoders, `pulsatrix_explain_text --graph` builds a graph from
[AttnLRP](../interpretability/lrp.md): one node per token for the residual stream after each
layer, and edges for the relevance each block passes back. It shows where information flows,
but its nodes aren't interpretable features. Details:
[Visualization: attribution graphs](../visualization/attribution-graphs.md).

```bash
pulsatrix_explain_text MODEL_DIR "The Eiffel Tower is located in the city of" --graph eiffel.json --viewer-dir graphs/
```

## Ablation circuit graphs for small networks

For a network built from layers, `ExplainerContext::build_circuit_graph()` reruns the forward
pass with each layer's activation zeroed in turn. It records how far the output moves (the L2
distance) as that node's `ablation_effect`.

```cpp
#include "pulsatrix/explainer_context.hpp"

ExplainerContext ctx({&hidden, &relu, &head});
Tensor input(Shape({1, 4}), &backend, {0.2f, -1.0f, 0.5f, 0.9f});
CircuitGraph circuit = ctx.build_circuit_graph(input);
for (const CircuitNode& node : circuit.nodes()) {
    // node.ablation_effect: how far the output moves when this node is zeroed
}
```

- **Drawing it.** `CircuitGraphView` draws it ([Visualization](../visualization/index.md)).
- **Saving it.** `ToJson(ToCircuitGraphDocument(circuit))` saves it.
- **Opening it in the viewer.** `pulsatrix_svg circuit.json --neuronpedia` converts it to the
  viewer's format.

## Viewing a graph

- **Locally, with circuit-tracer.**
  1. `--viewer-dir DIR` adds each graph to `DIR/graph-metadata.json`, the index circuit-tracer's
     viewer reads.
  2. Start the viewer there: `circuit-tracer start-server --graph_file_dir DIR`.
  3. Open `http://127.0.0.1:8041/index.html?slug=SLUG`. Use `127.0.0.1`: on `localhost` the
     viewer loads Anthropic's hosted examples instead.
- **On Neuronpedia.** Upload the file at
  [neuronpedia.org/graph/validator](https://www.neuronpedia.org/graph/validator). Neuronpedia
  uses `--scan` as its model id.

## References

- Ameisen, Lindsey et al., "Circuit Tracing: Revealing Computational Graphs in Language
  Models", Transformer Circuits, 2025.
- circuit-tracer: github.com/decoderesearch/circuit-tracer.
- Achtibat et al., "AttnLRP", ICML 2024.
