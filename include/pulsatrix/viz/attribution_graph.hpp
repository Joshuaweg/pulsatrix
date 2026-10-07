/** @file attribution_graph.hpp
 *  @brief Attribution graphs in the JSON that Neuronpedia's graph viewer and circuit-tracer's local
 *         viewer read (VIZ-4): nodes on a layer by token-position grid, and weighted links between
 *         them.
 *  @ingroup visualization
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/viz/document.hpp"

namespace pulsatrix {

/**
 * @brief An attribution graph as Neuronpedia's schema (`graph-schema.json`) and circuit-tracer's
 *        `create_graph_files` describe it.
 * @note The viewer places a node by its `layer` (`"E"` for embeddings, then `"0"`, `"1"`, ...,
 *       and the logit layer last) and its `ctx_idx` (token position), and draws a link's weight as
 *       its width and sign.
 */
struct AttributionGraph {
    struct Node {
        /** @brief Unique id, for example `"3_17_5"` (layer, feature, position). Links refer to it. */
        std::string node_id;
        /** @brief The feature's index; the viewer groups nodes with the same layer and feature. */
        std::optional<int64_t> feature;
        /** @brief `"E"` for an embedding, a layer number, or the logit layer. */
        std::string layer;
        int64_t ctx_idx = 0;
        /** @brief `"embedding"` and `"logit"` are special; a type containing `"error"` is drawn as
         *         an error node; anything else is drawn as a feature. */
        std::string feature_type;
        /** @brief The viewer's id for hover state across prompts. */
        std::string js_node_id;
        /** @brief The node's label. A logit's must hold `(p=PROBABILITY)`: the viewer reads it. */
        std::string clerp;
        /** @brief Cumulative influence (0 to 1); the viewer hides nodes above its pruning threshold. */
        std::optional<double> influence;
        std::optional<double> activation;
        double token_prob = 0.0;
        bool is_target_logit = false;
    };
    struct Link {
        std::string source;
        std::string target;
        double weight = 0.0;
    };

    std::string slug;
    /** @brief The model id, for example `"gemma-2-2b"`; Neuronpedia uses it to find the model. */
    std::string scan;
    std::string prompt;
    std::vector<std::string> prompt_tokens;
    /** @brief The viewer's default pruning threshold (circuit-tracer uses 0.8). */
    std::optional<double> node_threshold;
    /** @brief Optional `metadata.info` text. */
    std::string description;
    std::vector<Node> nodes;
    std::vector<Link> links;
};

/**
 * @brief The graph as Neuronpedia/circuit-tracer JSON, including circuit-tracer's optional node
 *        fields (`token_prob`, `is_target_logit`) and an empty `qParams`.
 * @throws std::invalid_argument if a required string is empty, node ids repeat, a link names an
 *         unknown node, a number isn't finite, a node's position is outside the prompt, or a
 *         logit's label lacks `(p=...)`.
 */
[[nodiscard]] std::string ToNeuronpediaJson(const AttributionGraph& graph);

/** @brief Reads a graph written by ToNeuronpediaJson, circuit-tracer or Neuronpedia. Unknown
 *         fields are ignored. @throws std::invalid_argument for a missing required field. */
[[nodiscard]] AttributionGraph ParseNeuronpediaGraph(std::string_view json);

/**
 * @brief Sets each node's influence the way circuit-tracer's pruning does: nodes other than
 *        embeddings and logits, by |score| largest first, get the cumulative fraction of the total
 *        |score| up to and including them. The viewer keeps nodes whose influence is at most its
 *        threshold, so a threshold of 0.8 keeps the fewest nodes holding 80% of the score.
 * @param scores One per node, in node order; embedding and logit entries are ignored.
 * @throws std::invalid_argument if the sizes differ or a score isn't finite.
 */
void ComputeInfluence(AttributionGraph& graph, const std::vector<double>& scores);

/**
 * @brief Keeps the fewest links, largest |weight| first, whose |weight| sums to at least
 *        @p edge_threshold of the total; ties keep their order. Links into a logit node are
 *        always kept.
 * @throws std::invalid_argument unless 0 < edge_threshold <= 1.
 */
void PruneLinks(AttributionGraph& graph, double edge_threshold);

/**
 * @brief A circuit graph (a chain of operations scored by ablation) for the viewer: two columns,
 *        `"input"` holding an embedding node and `"network"` holding each operation one layer up,
 *        with the last operation as the output. Influence comes from the ablation effects.
 * @note Two columns, not one: the viewer hides features seen at more than 2/3 of the positions,
 *       which with a single position is every node.
 * @throws std::invalid_argument for an empty graph or a non-finite effect or weight.
 */
[[nodiscard]] AttributionGraph FromCircuitGraph(const CircuitGraphDocument& doc, const std::string& slug,
                                                const std::string& scan);

}  // namespace pulsatrix
