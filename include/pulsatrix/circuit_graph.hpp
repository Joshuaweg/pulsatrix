/** @file circuit_graph.hpp
 *  @brief Self-contained circuit-graph artifact -- scored nodes and weighted edges.
 *  @ingroup mech_interp
 */
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/node.hpp"
#include "pulsatrix/op_type.hpp"

namespace pulsatrix {

/**
 * @brief One node of a CircuitGraph: a computation-graph node plus its importance score.
 */
struct CircuitNode {
    /** @brief The node's id in the ComputationGraph this circuit was built from. */
    NodeId id;

    /** @brief The node's op-type tag, copied at build time. */
    OpType op_type;

    /** @brief The node's optional human-readable label (e.g. a layer name). */
    std::optional<std::string> label;

    /**
     * @brief Ablation importance: the L2 distance between the network's real output and
     *        the output it produces when this node's activation is replaced by zeros.
     * @note Larger means "zeroing this node changes what the network outputs more", i.e.
     *       this node matters more. Always >= 0. The output node's score is 0.0f by
     *       convention, not by computation -- see ExplainerContext::build_circuit_graph().
     */
    float ablation_effect;
};

/**
 * @brief One directed edge of a CircuitGraph, carrying a scalar weight.
 */
struct CircuitEdge {
    /** @brief Source node id. */
    NodeId from;

    /** @brief Destination node id. */
    NodeId to;

    /**
     * @brief The edge's importance weight.
     * @note As produced by ExplainerContext::build_circuit_graph(), this is the *source*
     *       node's own ablation_effect -- a deliberate, documented simplification that is
     *       only correct because every ComputationGraph this codebase builds today is a
     *       single-parent linear chain, so "node i matters" and "the i -> i+1 connection
     *       matters" name the same causal quantity. A genuinely branching architecture
     *       would need per-edge causal mediation instead; see the mission notes in
     *       how/campaigns/campaign_exai_dl_library_mechanistic_interpretability/missions/
     *       mission_circuit_graph.md.
     */
    float weight;
};

/**
 * @brief A copyable, self-contained circuit graph: every node of one forward pass with an
 *        ablation importance score, plus the weighted edges between them.
 * @note Raw data only. This type deliberately ships **no rendering** -- no plotting, no
 *       DOT/JSON export, no dependency on any visualization stack. That is an explicit
 *       decision recorded by this campaign (Phase 5 scoping, 2026-09-24), not an
 *       oversight: drawing a circuit graph belongs to
 *       campaign_exai_dl_library_phase5_bindings Mission 2's still-planned visualization
 *       work, and is deferred to whenever that activates (or to a later mission if it is
 *       delayed further). Nothing here is committed to a particular renderer.
 * @note Holds no reference to the ComputationGraph or ExplainerContext it was built from,
 *       for the same reason ActivationSnapshot does not: ExplainerContext replaces its
 *       graph wholesale on every forward pass, so a circuit that referred back to either
 *       would be invalidated by the very next one. Copying the small per-node metadata
 *       instead makes the artifact outlive the analysis that produced it.
 * @note Constructed by ExplainerContext::build_circuit_graph(), not assembled by hand.
 */
class CircuitGraph {
public:
    /**
     * @brief Constructs a circuit graph from already-computed nodes and edges.
     * @param nodes Scored nodes, in topological order.
     * @param edges Weighted edges.
     * @note No cross-validation between the two arguments (e.g. that every edge endpoint
     *       names a node in the list): this constructor is reached only from
     *       ExplainerContext::build_circuit_graph(), which builds both from one
     *       consistent graph in a single pass -- the same rationale ActivationSnapshot's
     *       constructor documents. An empty circuit, and a single-node/zero-edge circuit,
     *       are both well-formed values rather than errors; the latter is what a
     *       one-node graph genuinely produces.
     */
    CircuitGraph(std::vector<CircuitNode> nodes, std::vector<CircuitEdge> edges)
        : nodes_(std::move(nodes)), edges_(std::move(edges)) {}

    /** @brief Every scored node, in topological order as of build time. */
    [[nodiscard]] const std::vector<CircuitNode>& nodes() const { return nodes_; }

    /** @brief Every weighted edge, in source-node order as of build time. */
    [[nodiscard]] const std::vector<CircuitEdge>& edges() const { return edges_; }

private:
    std::vector<CircuitNode> nodes_;
    std::vector<CircuitEdge> edges_;
};

}  // namespace pulsatrix
