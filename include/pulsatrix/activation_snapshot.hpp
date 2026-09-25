/** @file activation_snapshot.hpp
 *  @brief Self-contained, enumerable copy of one forward pass's cached activations.
 */
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/node.hpp"
#include "pulsatrix/op_type.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief A copyable, self-contained record of every activation cached during one
 *        forward pass, plus the node ids in topological order and each node's
 *        op_type/label metadata.
 * @note Holds no reference to the ComputationGraph or ExplainerContext it was captured
 *       from -- deliberately. ExplainerContext::forward_pass() replaces its graph and
 *       activation cache wholesale on every call, and ComputationGraph is move-only
 *       (std::vector<std::unique_ptr<Node>>), so a snapshot that referred back to either
 *       would be invalidated by the very next forward pass. Copying the activation
 *       Tensors and the small amount of per-node metadata instead is what makes two
 *       snapshots from two different forward passes simultaneously alive and correct --
 *       the capability activation patching needs (campaign
 *       campaign_exai_dl_library_mechanistic_interpretability, Phase 4).
 * @note NodeIds are comparable across two snapshots taken from the same ExplainerContext:
 *       its module chain is fixed for its lifetime, so every forward pass builds a
 *       structurally identical graph (same node count, same op-type sequence, same ids),
 *       differing only in the cached values.
 * @note Constructed by ExplainerContext::activation_snapshot(), not assembled by hand.
 */
class ActivationSnapshot {
public:
    /** @brief Per-node metadata captured alongside the activation value. */
    struct NodeMetadata {
        /** @brief The node's op-type tag, as of capture time. */
        OpType op_type;

        /** @brief The node's optional human-readable label (e.g. a layer name). */
        std::optional<std::string> label;
    };

    /**
     * @brief Constructs a snapshot from already-captured data.
     * @param activations Deep copy of the cached activation value per node id.
     * @param node_ids Node ids in topological order at capture time.
     * @param metadata Per-node op_type/label, captured at the same moment.
     * @note No cross-validation between the three arguments: this constructor is reached
     *       only from ExplainerContext::activation_snapshot(), which builds all three from
     *       one consistent graph in a single pass. An empty snapshot (all three empty) is
     *       a well-formed value, not an error -- it is what a context that has not yet run
     *       a forward pass produces.
     */
    ActivationSnapshot(std::unordered_map<NodeId, Tensor> activations, std::vector<NodeId> node_ids,
                       std::unordered_map<NodeId, NodeMetadata> metadata)
        : activations_(std::move(activations)), node_ids_(std::move(node_ids)), metadata_(std::move(metadata)) {}

    /**
     * @brief The activation value captured at a node.
     * @param id Node id. Must be one of node_ids().
     * @note Mirrors ExplainerContext::activation()'s not-found behavior (PULSATRIX_ASSERT):
     *       internal-only invariant per
     *       cpp_tdd/context_tdd_adversarial_boundary_testing.md's classification table --
     *       every id a caller can legitimately hold came from node_ids().
     */
    [[nodiscard]] const Tensor& activation(NodeId id) const {
        auto it = activations_.find(id);
        PULSATRIX_ASSERT(it != activations_.end());
        return it->second;
    }

    /**
     * @brief Every captured node id, in topological order as of capture time.
     * @note The enumeration capability ExplainerContext's single-node activation() lookup
     *       lacks -- Phase 2's linear probing iterates this to reach every cached layer.
     */
    [[nodiscard]] const std::vector<NodeId>& node_ids() const { return node_ids_; }

    /**
     * @brief The op-type tag captured at a node.
     * @param id Node id. Must be one of node_ids().
     */
    [[nodiscard]] OpType op_type(NodeId id) const {
        auto it = metadata_.find(id);
        PULSATRIX_ASSERT(it != metadata_.end());
        return it->second.op_type;
    }

    /**
     * @brief The optional label captured at a node.
     * @param id Node id. Must be one of node_ids().
     * @return The node's label, or std::nullopt if it had none.
     */
    [[nodiscard]] std::optional<std::string> label(NodeId id) const {
        auto it = metadata_.find(id);
        PULSATRIX_ASSERT(it != metadata_.end());
        return it->second.label;
    }

private:
    std::unordered_map<NodeId, Tensor> activations_;
    std::vector<NodeId> node_ids_;
    std::unordered_map<NodeId, NodeMetadata> metadata_;
};

}  // namespace pulsatrix
