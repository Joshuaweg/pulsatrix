/** @file relevance_graph.hpp
 *  @brief A language model's attribution graph from AttnLRP (VIZ-4): how relevance for one
 *         predicted token flows from the input tokens through every layer's residual stream, ready
 *         for Neuronpedia's and circuit-tracer's graph viewers.
 *  @ingroup mech_interp
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/attnlrp_parity.hpp"  // LxtAttnLrpConfig
#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/viz/attribution_graph.hpp"

namespace pulsatrix {

struct RelevanceGraphOptions {
    /** @brief The LRP rules: AttnLRP as LXT applies it, by default. */
    LRPRuleConfig config = LxtAttnLrpConfig();
    /** @brief Keep the fewest links holding this share of the total |relevance| (PruneLinks). */
    double edge_threshold = 0.98;
    /** @brief The viewer's default node pruning threshold (ComputeInfluence). */
    double node_threshold = 0.8;
    std::string slug = "pulsatrix-graph";
    /** @brief The model id the viewer shows, for example `"Qwen2.5-0.5B"`. */
    std::string scan = "pulsatrix";
    /** @brief The prompt's text, and each token as text (one per id). */
    std::string prompt;
    std::vector<std::string> prompt_tokens;
    /** @brief The explained token as text, for the output node's label. */
    std::string target_text;
    /** @brief Refuse longer inputs: the graph costs one block pass per layer and position. */
    int64_t max_tokens = 256;
};

/**
 * @brief Explains the model's logit for @p target at the last position with LRP and returns the
 *        relevance flow as an attribution graph.
 *
 * Nodes are the residual stream at each token position: the embeddings (layer `"E"`), then the
 * output of every block (layers `"0"` to `"L-1"`), and the output token. A node's activation is its
 * relevance (summed over the hidden size). A link from position i in one layer to position j in
 * the next is the part of j's relevance that the block passes back to i: the block's LRP run with
 * relevance only at j. Because the LRP rules are linear in the incoming relevance, a node's
 * outgoing links sum to its own relevance, and the embedding nodes hold the per-token relevance
 * that CausalLM::propagate_relevance gives. A layer's total needn't equal the next layer's: LRP
 * through norms and biases isn't strictly conservative.
 *
 * @param ids The prompt's token ids (one sequence).
 * @note Cost: one forward pass, one full LRP pass, then one block LRP pass per layer and position.
 * @throws std::invalid_argument for an empty or too long prompt, a target outside the
 *         vocabulary, or prompt_tokens of the wrong length.
 */
[[nodiscard]] AttributionGraph BuildRelevanceGraph(CausalLM& model, DeviceBackend* backend, const std::vector<int64_t>& ids,
                                                   int64_t target, const RelevanceGraphOptions& options);

}  // namespace pulsatrix
