/** @file explainer_context.hpp
 *  @brief Stable interface every explainer gets, regardless of type (charter Part 2 SS2).
 */
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "exai/activation_snapshot.hpp"
#include "exai/assert.hpp"
#include "exai/autograd.hpp"
#include "exai/computation_graph.hpp"
#include "exai/module.hpp"
#include "exai/tensor.hpp"

namespace exai {

/**
 * @brief Wraps an ordered chain of Modules, running them via Module::forward_traced to
 *        build a real ComputationGraph and Autograd backward wiring -- graph-native
 *        explainers (Missions 2-3) use graph()/backward_pass()/activation(); surrogate
 *        explainers (Phase 3) would use only forward_pass(), per the charter's stated
 *        interface segregation.
 * @note Owns a std::vector<Module*>, not a new Sequential container -- no generic
 *       module-sequence type exists yet in this codebase (XorNetwork hardcodes its own
 *       chain), and this mission doesn't need one beyond what ExplainerContext itself
 *       requires. Modules are not owned; every pointer must outlive this ExplainerContext.
 * @note forward_pass() replaces its ComputationGraph/Autograd members with fresh instances
 *       on every call, rather than mutating a shared graph -- ComputationGraph deliberately
 *       has no clear/reset method (Phase 0's persistence-past-backward guarantee applies
 *       within one forward_pass()'s lifetime), but Integrated Gradients (Mission 2) needs
 *       multiple independent forward passes at different interpolated inputs, each with
 *       its own graph. See mission_explainer_context.md's Recon.
 */
class ExplainerContext {
public:
    /**
     * @brief Constructs a context over an ordered module chain.
     * @param modules The network, in forward-pass order. Not owned; each must outlive
     *        this ExplainerContext.
     * @throws std::invalid_argument if modules is empty or contains a nullptr -- external
     *         boundary (campaign_exai_dl_library_adversarial_hardening.md, Mission 2,
     *         finding 5): without this check, a nullptr element was an unconditional
     *         null-pointer dereference on the next forward_pass() call.
     */
    explicit ExplainerContext(std::vector<Module*> modules) : modules_(std::move(modules)) {
        if (modules_.empty()) {
            throw std::invalid_argument("ExplainerContext: modules must not be empty");
        }
        for (Module* m : modules_) {
            if (m == nullptr) {
                throw std::invalid_argument("ExplainerContext: modules must not contain a null pointer");
            }
        }
    }

    /**
     * @brief Runs the full module chain forward, building a fresh graph and caching every
     *        node's activation value as it goes.
     * @param input Input to the first module in the chain.
     * @return The final module's output.
     */
    Tensor forward_pass(const Tensor& input) {
        last_forward_was_patched_ = false;
        graph_ = ComputationGraph{};
        autograd_ = Autograd{};
        activations_.clear();

        input_node_ = graph_.add_node(OpType::Elementwise, input.shape(), "input");
        activations_.emplace(input_node_, Tensor(input));

        Tensor current = input;
        NodeId current_node = input_node_;
        for (Module* module : modules_) {
            auto [output, node_id] = module->forward_traced(current, current_node, graph_, autograd_);
            activations_.emplace(node_id, Tensor(output));
            current = std::move(output);
            current_node = node_id;
        }
        output_node_ = current_node;
        return current;
    }

    /**
     * @brief Runs the full module chain forward, substituting patch_value for the natural
     *        output of the module that produces patch_node_id -- the causal-intervention
     *        (activation patching) primitive.
     * @param input Input to the first module in the chain.
     * @param patch_node_id Node whose activation is overridden. Node ids are assigned
     *        deterministically per forward pass: 0 is the input node, then one per module
     *        in chain order, so a node id captured from an earlier pass over this same
     *        context names the same logical layer here.
     * @param patch_value Value substituted at that node. Must have the same shape as the
     *        node's natural (unpatched) output.
     * @return The chain's output, computed downstream from the substituted value. Patching
     *         the output node returns patch_value itself -- nothing runs after it.
     * @throws std::invalid_argument if patch_node_id exceeds the largest node id this
     *         forward pass produces, or if patch_value's shape differs from the target
     *         node's natural output shape -- both external boundaries (caller-supplied),
     *         hence throw rather than EXAI_ASSERT, consistent with the constructor.
     * @note Every module *before* the patch point is unaffected (a forward-only computation
     *       has no upstream influence); every module *after* it computes from patch_value.
     *       Patching the input node is a supported degenerate case, equivalent to
     *       forward_pass(patch_value).
     * @note Builds into local graph/autograd/activation state and only commits it on
     *       success, so a throw leaves this context exactly as the previous forward_pass()
     *       left it -- a failed patch attempt must not corrupt a usable context.
     * @note Single-node patch per call by design (campaign
     *       campaign_exai_dl_library_mechanistic_interpretability, Phase 4 Mission 1);
     *       multi-node patching would be an explicit extension, not assumed here.
     * @note backward_pass()'s guard against this method covers Autograd-based gradients
     *       only. ExplainerContext exposes no LRP/propagate_relevance traversal of its own
     *       to guard -- Module::propagate_relevance() is invoked directly by explainer code
     *       outside this class (module.hpp), not through ExplainerContext. If a future
     *       explainer manually chains propagate_relevance() calls across modules using
     *       state left behind by a patched forward pass, the same causal-inconsistency
     *       hazard backward_pass() guards against applies there too, unguarded -- that
     *       explainer's own author is responsible for it, the same way any code bypassing
     *       ExplainerContext's own accessors already is.
     */
    Tensor forward_pass_with_patch(const Tensor& input, NodeId patch_node_id, const Tensor& patch_value) {
        last_forward_was_patched_ = true;
        if (patch_node_id > modules_.size()) {
            throw std::invalid_argument("ExplainerContext::forward_pass_with_patch: patch_node_id out of range");
        }

        ComputationGraph graph;
        Autograd autograd;
        std::unordered_map<NodeId, Tensor> activations;

        NodeId input_node = graph.add_node(OpType::Elementwise, input.shape(), "input");

        Tensor current = input;
        NodeId current_node = input_node;
        if (patch_node_id == input_node) {
            current = check_patch_shape(current, patch_value);
        }
        activations.emplace(input_node, Tensor(current));

        for (Module* module : modules_) {
            auto [output, node_id] = module->forward_traced(current, current_node, graph, autograd);
            if (node_id == patch_node_id) {
                output = check_patch_shape(output, patch_value);
            }
            activations.emplace(node_id, Tensor(output));
            current = std::move(output);
            current_node = node_id;
        }

        graph_ = std::move(graph);
        autograd_ = std::move(autograd);
        activations_ = std::move(activations);
        input_node_ = input_node;
        output_node_ = current_node;
        return current;
    }

    /**
     * @brief Runs Autograd::backward from the most recent forward_pass()'s output node.
     * @param output_grad Gradient w.r.t. the chain's output.
     * @return Gradient w.r.t. the chain's input.
     * @note Must be called after forward_pass().
     * @throws std::logic_error if the most recent forward pass was a
     *         forward_pass_with_patch() call. A patched activation is a *constant
     *         substitution*, not a differentiable function of the input, yet
     *         Module::forward_traced registers each module's ordinary backward closure
     *         regardless -- so without this guard Autograd::backward() would return a
     *         perfectly plausible gradient taken through a link that does not exist in the
     *         computation it claims to differentiate. Rejected loudly rather than answered
     *         wrongly, the same principle as Phase 1.5's device guards; classified external
     *         boundary (a caller-sequencing mistake, like the constructor's checks) so it is
     *         a throw and behaves identically in Debug and Release. Re-arm with an ordinary
     *         forward_pass() -- the guard is per-call history, not a permanent latch.
     */
    Tensor backward_pass(const Tensor& output_grad) {
        if (last_forward_was_patched_) {
            throw std::logic_error(
                "ExplainerContext::backward_pass: the most recent forward pass was "
                "forward_pass_with_patch(); a patched activation is a constant substitution, not a "
                "differentiable function of the input, so gradients through it would be silently wrong. "
                "Run an unpatched forward_pass() before backward_pass().");
        }
        autograd_.backward(graph_, output_node_, output_grad);
        return Tensor(autograd_.gradient(input_node_));
    }

    /** @brief The current graph (from the most recent forward_pass() call). */
    [[nodiscard]] const ComputationGraph& graph() const { return graph_; }

    /**
     * @brief The cached activation value at a node, from the most recent forward_pass().
     * @param id Node id. Must have been produced by the most recent forward_pass() call.
     */
    [[nodiscard]] const Tensor& activation(NodeId id) const {
        auto it = activations_.find(id);
        EXAI_ASSERT(it != activations_.end());
        return it->second;
    }

    /**
     * @brief The cached gradient at a node, from the most recent backward_pass() call.
     * @param id Node id. Must have accumulated a gradient during the most recent
     *        backward_pass() call (i.e. lie on the path between the seeded output and
     *        the input).
     * @note Mirrors activation()'s shape -- Autograd::backward() already populates a
     *       gradient at every node it walks through, not just the input node
     *       backward_pass() itself returns; this exposes that directly, needed for
     *       Grad-CAM's target-conv-layer gradient (Phase 2 Mission 3).
     */
    [[nodiscard]] const Tensor& gradient(NodeId id) const { return autograd_.gradient(id); }

    /**
     * @brief Captures the current activation cache into a self-contained ActivationSnapshot.
     * @return A snapshot of every activation from the most recent forward_pass(), the node
     *         ids in topological order, and each node's op_type/label -- all copied, with no
     *         reference back to this context or its graph.
     * @note Does not run (or require) a new forward pass; it reads what is already cached.
     *       Called before any forward_pass(), it returns an empty snapshot -- there is
     *       nothing cached yet, which is a well-defined state, not an error.
     * @note This is the generalization of activation(): that method is a single-node lookup
     *       into a cache the next forward_pass() overwrites in place, so two runs'
     *       activations could never be held at once. A snapshot survives any number of
     *       subsequent forward passes, which is what activation patching (campaign
     *       campaign_exai_dl_library_mechanistic_interpretability, Phase 4) needs -- a
     *       "clean" and a "corrupted" run alive simultaneously. Since modules_ is fixed for
     *       this context's lifetime, snapshots from different forward passes are
     *       NodeId-comparable.
     */
    [[nodiscard]] ActivationSnapshot activation_snapshot() const {
        std::vector<NodeId> node_ids = graph_.topological_order();

        std::unordered_map<NodeId, Tensor> activations;
        std::unordered_map<NodeId, ActivationSnapshot::NodeMetadata> metadata;
        for (NodeId id : node_ids) {
            auto it = activations_.find(id);
            EXAI_ASSERT(it != activations_.end());
            activations.emplace(id, Tensor(it->second));
            const Node& node = graph_.node(id);
            metadata.emplace(id, ActivationSnapshot::NodeMetadata{node.op_type(), node.label()});
        }

        return ActivationSnapshot(std::move(activations), std::move(node_ids), std::move(metadata));
    }

    /** @brief Passthrough to Node::label() -- e.g. a layer name, for debugging/display. */
    [[nodiscard]] std::optional<std::string> layer_label(NodeId id) const { return graph_.node(id).label(); }

private:
    /**
     * @brief Validates a patch value against the natural output it replaces.
     * @param natural The node's own (unpatched) output, whose shape is the contract.
     * @param patch_value Caller-supplied replacement.
     * @return A copy of patch_value, ready to substitute for natural.
     * @throws std::invalid_argument if the shapes differ -- reported at the patch site,
     *         where the mistake actually is, rather than as a downstream module's
     *         shape-check failure several layers later.
     */
    [[nodiscard]] static Tensor check_patch_shape(const Tensor& natural, const Tensor& patch_value) {
        if (!(patch_value.shape() == natural.shape())) {
            throw std::invalid_argument(
                "ExplainerContext::forward_pass_with_patch: patch_value shape must match the patched node's "
                "natural output shape");
        }
        return Tensor(patch_value);
    }

    std::vector<Module*> modules_;
    ComputationGraph graph_;
    Autograd autograd_;
    std::unordered_map<NodeId, Tensor> activations_;
    NodeId input_node_ = 0;
    NodeId output_node_ = 0;
    /**
     * @brief Whether the most recent forward pass was a patched one -- backward_pass()'s
     *        precondition (campaign_exai_dl_library_mechanistic_interpretability, Phase 4
     *        Mission 2).
     * @note Set at the *top* of forward_pass_with_patch(), not in its commit block, so that
     *       a patch attempt which throws part-way still arms the guard. That is not an
     *       oversight: the commit block protects graph_/autograd_/activations_, but nothing
     *       can un-run the modules that already executed, and each Autograd closure calls
     *       module->backward(), which reads that module's own cache from its most recent
     *       forward(). After an aborted patched forward those caches belong to the aborted
     *       run while the committed graph belongs to the previous one -- exactly the
     *       cross-run mixture this guard exists to refuse.
     */
    bool last_forward_was_patched_ = false;
};

}  // namespace exai
