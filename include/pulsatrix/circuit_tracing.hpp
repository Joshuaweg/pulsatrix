/** @file circuit_tracing.hpp
 *  @brief Attribution graphs with cross-layer transcoders (FEAT-10), after Ameisen, Lindsey et
 *         al., "Circuit Tracing: Revealing Computational Graphs in Language Models" (Transformer
 *         Circuits, 2025) and its open implementation, circuit-tracer
 *         (github.com/decoderesearch/circuit-tracer). Each MLP is replaced by a transcoder's
 *         features plus an error term, attention patterns and normalization scales are frozen,
 *         and what is left is linear: the graph's edges are exact attributions between features,
 *         errors, token embeddings and output logits.
 *  @ingroup mech_interp
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/tensor.hpp"
#include "pulsatrix/viz/attribution_graph.hpp"

namespace pulsatrix {

/**
 * @brief A cross-layer transcoder (CLT) over a model's L MLPs, or a set of per-layer
 *        transcoders (PLTs, a CLT whose features only write to their own layer).
 *
 * The features of layer ℓ read the MLP's input there (after the block's pre-MLP norm), and
 * write to the MLP outputs of layers ℓ to ℓ + span − 1:
 * - `pre^ℓ = x^ℓ W_enc^ℓᵀ + b_enc^ℓ`, `a^ℓ = pre^ℓ [pre^ℓ > θ^ℓ]` (JumpReLU; ReLU without θ);
 * - `ŷ^ℓ' = Σ_{ℓ ≤ ℓ'} a^ℓ W_dec^{ℓ→ℓ'} + b_dec^ℓ' [+ x^ℓ' W_skip^ℓ'ᵀ]`.
 *
 * Weights are stored as circuit-tracer does: W_enc^ℓ `(F, d)`, each W_dec `(F, d)`, W_skip
 * `(d, d)`, on the backend's device. Inference only: pretrained transcoders are loaded with
 * LoadTranscoders().
 */
class CrossLayerTranscoder {
public:
    /** @throws std::invalid_argument for a size < 1. */
    CrossLayerTranscoder(int64_t num_layers, int64_t d_model, DeviceBackend* backend);

    /**
     * @brief Sets layer @p layer's features.
     * @param w_enc `(F, d)`. @param b_enc F values. @param threshold F values (JumpReLU), or
     *        empty for ReLU.
     * @param w_dec One `(F, d)` matrix per output layer, from @p layer on; at most L − layer.
     * @throws std::invalid_argument for a layer out of range or inconsistent sizes.
     */
    void set_layer(int64_t layer, Tensor w_enc, std::vector<float> b_enc, std::vector<float> threshold, std::vector<Tensor> w_dec);
    /** @brief The decoder bias of layer @p layer's MLP output, d values (zero by default). */
    void set_output_bias(int64_t layer, std::vector<float> b_dec);
    /** @brief A linear skip from layer @p layer's MLP input to its output, `(d, d)`. */
    void set_skip(int64_t layer, Tensor w_skip);

    [[nodiscard]] int64_t num_layers() const { return L_; }
    [[nodiscard]] int64_t d_model() const { return d_; }
    [[nodiscard]] int64_t num_features(int64_t layer) const;
    /** @brief How many layers layer @p layer's features write to (1 for a PLT). */
    [[nodiscard]] int64_t span(int64_t layer) const;
    [[nodiscard]] bool has_skip(int64_t layer) const;

    /** @brief Pre-activations `x W_encᵀ + b_enc`, `(rows, F)` on the host, for @p x `(rows, d)`. */
    [[nodiscard]] std::vector<float> preactivations(int64_t layer, const Tensor& x) const;
    /** @brief The activations: JumpReLU (or ReLU) of the pre-activations. */
    [[nodiscard]] std::vector<float> encode(int64_t layer, const Tensor& x) const;
    /**
     * @brief Layer @p layer's MLP output as the transcoder predicts it, `(rows, d)` on the host:
     *        every earlier layer's decoders into it, its bias and its skip.
     * @param activations activations[ℓ] `(rows, F_ℓ)` for each ℓ ≤ layer (later ones ignored).
     * @param mlp_input Layer @p layer's MLP input `(rows, d)`, for the skip; ignored without one.
     */
    [[nodiscard]] std::vector<float> reconstruct(int64_t layer, const std::vector<std::vector<float>>& activations, const Tensor* mlp_input) const;

    /** @brief Feature @p f of layer @p layer: its encoder row and its decoder into @p out_layer, d each. */
    [[nodiscard]] std::vector<float> encoder_row(int64_t layer, int64_t f) const;
    [[nodiscard]] std::vector<float> decoder_row(int64_t layer, int64_t f, int64_t out_layer) const;
    [[nodiscard]] const Tensor& encoder(int64_t layer) const;
    [[nodiscard]] const Tensor& decoder(int64_t layer, int64_t out_layer) const;
    [[nodiscard]] const Tensor& skip(int64_t layer) const;
    [[nodiscard]] DeviceBackend* backend() const { return backend_; }

private:
    struct Layer {
        Tensor w_enc;
        std::vector<float> b_enc, threshold, b_dec;
        std::vector<Tensor> w_dec;
        std::vector<Tensor> w_skip;  ///< empty or one (d, d)
    };
    int64_t L_, d_;
    DeviceBackend* backend_;
    std::vector<Layer> layers_;
};

/**
 * @brief Loads transcoders from a directory in one of the layouts circuit-tracer reads, with
 *        bf16/fp16 weights upcast:
 * - **Per-layer** (`model_kind: transcoder_set`, as in mwhanna/gemma-scope-2-270m-pt):
 *   `layer_<i>.safetensors` holding `W_enc` `(F, d)`, `W_dec` `(F, d)`, `b_enc`, `b_dec`,
 *   optionally `activation_function.threshold` (or `threshold`) and `W_skip`.
 * - **Cross-layer, circuit-tracer's lazy format**: `W_enc_<i>.safetensors` holding `W_enc_<i>`,
 *   `b_enc_<i>`, `b_dec_<i>`, optionally `threshold_<i>` and `W_skip_<i>`, and
 *   `W_dec_<i>.safetensors` holding `W_dec_<i>` `(F, L − i, d)`.
 * @param num_layers The model's layers; files are read for 0 .. num_layers − 1.
 * @throws std::invalid_argument for a missing file or tensor, or shapes that don't agree.
 */
[[nodiscard]] CrossLayerTranscoder LoadTranscoders(const std::string& directory, int64_t num_layers, DeviceBackend* backend);

struct CircuitTraceOptions {
    /** @brief Logits explained: the most likely, up to this many, until their probabilities sum
     *         to logit_probability (circuit-tracer: 10 and 0.95). */
    int64_t max_logits = 10;
    double logit_probability = 0.95;
    /** @brief Positions whose features and errors are left out, from the start (circuit-tracer
     *         zeroes the first: BOS's MLP outputs become constants). */
    int64_t skip_positions = 1;
    /** @brief Targets per frozen backward pass. */
    int64_t batch = 64;
    /** @brief Node and edge pruning (circuit-tracer's defaults). 1 keeps everything. */
    double node_threshold = 0.8;
    double edge_threshold = 0.98;
    std::string slug = "pulsatrix-clt-graph";
    std::string scan = "pulsatrix";
    std::string prompt;
    /** @brief One per token, for the viewer; empty: the ids as text. */
    std::vector<std::string> prompt_tokens;
};

/**
 * @brief An attribution graph before pruning: every active feature, error and token, the
 *        explained logits, and the full adjacency.
 *
 * Node order (circuit-tracer's): features, then errors (layer-major, one per layer and
 * position), then token embeddings, then logits. `adjacency[t * n + s]` is the attribution of
 * source s to target t: s's output vector, through the frozen model, dotted with t's input
 * vector. A feature's input vector is its encoder row (pre-activation without b_enc); a logit's
 * is its unembedding minus the mean unembedding, at the final norm's output.
 */
struct CircuitTrace {
    struct Feature {
        int64_t layer = 0, position = 0, index = 0;
        float activation = 0;
    };
    std::vector<Feature> features;
    int64_t num_layers = 0, num_positions = 0;
    std::vector<int64_t> tokens;
    /** @brief The explained logits' token ids and probabilities. */
    std::vector<int64_t> logit_tokens;
    std::vector<double> logit_probabilities;
    /** @brief Each target's value with the frozen model: the pre-activation without b_enc, or the
     *         logit minus the mean logit. One per node (0 for errors and tokens). */
    std::vector<double> target_values;
    std::vector<double> adjacency;

    [[nodiscard]] int64_t num_nodes() const;
    [[nodiscard]] int64_t error_node(int64_t layer, int64_t position) const;
    [[nodiscard]] int64_t token_node(int64_t position) const;
    [[nodiscard]] int64_t logit_node(int64_t k) const;
};

/**
 * @brief Runs the model on @p ids (one sequence) with @p transcoder in place of its MLPs, and
 *        attributes the salient logits and every active feature through the frozen model.
 * - The forward pass is the model's own: reconstruction plus error is the MLP's output, so the
 *   logits are unchanged.
 * - Frozen: attention patterns, every RMSNorm's scale (pre- and post-norms, the final norm).
 *   Gradients don't pass through the MLPs; with a skip they pass through it.
 * - Edges come from batched backward passes from each target's input vector, contracted at
 *   every layer's MLP output with the sources' output vectors (`a_s` times decoders, errors) and
 *   at the embeddings with the scaled token embeddings.
 * @note Gemma's final logit softcap is outside the linearization.
 * @throws std::invalid_argument if the transcoder's layers or width don't match the model, or the
 *         prompt is empty.
 */
[[nodiscard]] CircuitTrace TraceCircuit(CausalLM& model, const CrossLayerTranscoder& transcoder, const std::vector<int64_t>& ids,
                                        const CircuitTraceOptions& options = {});

/** @brief Circuit-tracer's scores of a graph. */
struct CircuitScores {
    /** @brief Influence reaching the logits from token embeddings, over that from tokens and errors. */
    double replacement = 0;
    /** @brief Influence-weighted share of each node's incoming attribution that isn't from errors. */
    double completeness = 0;
};
/** @brief Circuit-tracer's replacement and completeness scores (graph.py). */
[[nodiscard]] CircuitScores ScoreCircuit(const CircuitTrace& trace);

/**
 * @brief The graph for Neuronpedia's or circuit-tracer's viewer, pruned as circuit-tracer prunes:
 *        nodes by indirect influence on the logits (node_threshold), then edges by
 *        influence-weighted score (edge_threshold), then nodes left without edges. Node ids,
 *        types and labels follow circuit-tracer (`"<layer>_<feature>_<position>"`,
 *        `"cross layer transcoder"`, `"mlp reconstruction error"`, `"E_<token>_<position>"`,
 *        `"<L+1>_<token>_<position>"` for logits).
 */
[[nodiscard]] AttributionGraph ToAttributionGraph(const CircuitTrace& trace, const CircuitTraceOptions& options,
                                                  const std::vector<std::string>& logit_labels = {});

}  // namespace pulsatrix
