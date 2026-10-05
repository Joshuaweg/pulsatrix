/** @file explainer_context.hpp
 *  @brief Stable interface every explainer gets, regardless of type (charter Part 2 SS2).
 *  @ingroup interpretability_dl
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pulsatrix/activation_snapshot.hpp"
#include "pulsatrix/assert.hpp"
#include "pulsatrix/autograd.hpp"
#include "pulsatrix/circuit_graph.hpp"
#include "pulsatrix/computation_graph.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/multihead_attention_module.hpp"
#include "pulsatrix/tensor.hpp"

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

namespace pulsatrix {

namespace detail {
/** @brief The dynamic type's readable name (e.g. "pulsatrix::SoftmaxModule"), for error messages. */
inline std::string module_type_name(const Module& module) {
    const char* raw = typeid(module).name();
#if defined(__GNUG__)
    int status = 0;
    char* demangled = abi::__cxa_demangle(raw, nullptr, nullptr, &status);
    std::string name = (status == 0 && demangled != nullptr) ? demangled : raw;
    std::free(demangled);
    return name;
#else
    return raw;
#endif
}
}  // namespace detail

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
     *         hence throw rather than PULSATRIX_ASSERT, consistent with the constructor.
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

    /**
     * @brief Propagates LRP relevance from the network output back to the input through every
     *        module's own propagate_relevance(), in reverse order -- the relevance counterpart of
     *        backward_pass(), and what LRP::explain() runs.
     * @param output_relevance Relevance at the network output, same shape as the most recent
     *        forward_pass()'s output.
     * @param config LRP rule configuration handed to every module.
     * @return Relevance at the input, same shape as the input to that forward_pass().
     * @throws std::logic_error if the most recent forward pass was forward_pass_with_patch(), for
     *         the same reason backward_pass() refuses: the modules' cached state would describe a
     *         computation the input did not produce.
     * @throws std::invalid_argument if a module does not implement config.rule
     *         (Module::supports_lrp_rule()) -- checked for every module before any propagation;
     *         the rule is never silently replaced by epsilon.
     */
    Tensor relevance_pass(const Tensor& output_relevance, const LRPRuleConfig& config) {
        return relevance_pass(output_relevance, std::vector<LRPRuleConfig>(modules_.size(), config));
    }

    /**
     * @brief relevance_pass() with a per-module rule choice: `configs[i]` is handed to the i-th
     *        module (forward order) -- e.g. LRP composites (lrp.hpp).
     * @throws std::invalid_argument if configs.size() != number of modules, or a module does not
     *         implement its rule; std::logic_error as for the uniform overload.
     */
    Tensor relevance_pass(const Tensor& output_relevance, const std::vector<LRPRuleConfig>& configs) {
        if (last_forward_was_patched_) {
            throw std::logic_error(
                "ExplainerContext::relevance_pass: the most recent forward pass was "
                "forward_pass_with_patch(); relevance through a patched activation would describe a "
                "computation the input did not produce. Run an unpatched forward_pass() first.");
        }
        if (configs.size() != modules_.size()) {
            throw std::invalid_argument("ExplainerContext::relevance_pass: need exactly one LRPRuleConfig per module");
        }
        for (size_t i = 0; i < modules_.size(); ++i) {
            if (!modules_[i]->supports_lrp_rule(configs[i].rule)) {
                throw std::invalid_argument("ExplainerContext::relevance_pass: module " + std::to_string(i) + " (" +
                                            detail::module_type_name(*modules_[i]) + ") does not implement the " +
                                            lrp_rule_name(configs[i].rule) + " LRP rule");
            }
        }
        Tensor relevance = output_relevance;
        for (size_t i = modules_.size(); i-- > 0;) {
            relevance = modules_[i]->propagate_relevance(relevance, configs[i]);
        }
        return relevance;
    }

    /** @brief The module chain, in forward order (not owned). */
    [[nodiscard]] const std::vector<Module*>& modules() const { return modules_; }

    /** @brief The current graph (from the most recent forward_pass() call). */
    [[nodiscard]] const ComputationGraph& graph() const { return graph_; }

    /**
     * @brief The cached activation value at a node, from the most recent forward_pass().
     * @param id Node id. Must have been produced by the most recent forward_pass() call.
     */
    [[nodiscard]] const Tensor& activation(NodeId id) const {
        auto it = activations_.find(id);
        PULSATRIX_ASSERT(it != activations_.end());
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
            PULSATRIX_ASSERT(it != activations_.end());
            activations.emplace(id, Tensor(it->second));
            const Node& node = graph_.node(id);
            metadata.emplace(id, ActivationSnapshot::NodeMetadata{node.op_type(), node.label()});
        }

        return ActivationSnapshot(std::move(activations), std::move(node_ids), std::move(metadata));
    }

    /**
     * @brief The "logit lens": runs the chain's *final* module -- its read-out head -- on the
     *        cached activation at node_id, answering "what would the network predict if this
     *        layer's representation were already final?"
     * @param node_id Node whose cached activation is projected through the head. Must come
     *        from the most recent forward pass, and its activation's shape must match what
     *        the head actually consumed in that pass.
     * @return The head's output for that intermediate representation. Raw data only -- no
     *         softmax, no ranking, no plotting (campaign
     *         campaign_exai_dl_library_mechanistic_interpretability, Phase 5 Mission 1, is
     *         deliberately scoped data-only and takes no visualization dependency).
     * @throws std::invalid_argument if node_id has no cached activation (an id no forward
     *         pass produced, or no forward pass has run yet), or if that activation's shape
     *         differs from the head's actual input shape in the most recent forward pass --
     *         both external boundaries (caller-supplied node id), hence throw rather than
     *         PULSATRIX_ASSERT, matching forward_pass_with_patch and the constructor, and so the
     *         behavior is identical in Debug and Release.
     * @note The shape precondition is checked here, before the head runs, so the message
     *       names the mismatch at the point the mistake was made rather than surfacing as a
     *       confusing failure inside the head's own gemm. It is a real precondition of the
     *       technique, not an assumption: only a chain whose hidden width stays constant up
     *       to the head (the analogue of a transformer's fixed-width residual stream) has
     *       compatible earlier nodes at all.
     * @note The head's *expected* input shape is taken from the activation the head actually
     *       received in the most recent forward pass (the node at index modules_.size() - 1),
     *       not from any declared per-module input-shape accessor -- Module exposes no such
     *       accessor, and inventing one across every subclass is far outside this mission.
     * @note Applied to that same node, this returns the real forward pass's output exactly:
     *       it is literally the computation forward_pass() already performed, which is this
     *       method's zero-tolerance correctness oracle.
     * @note const with respect to *this* -- no ExplainerContext state is read-modified. It
     *       does run the head module's own forward(), which overwrites that module's internal
     *       forward cache (so a subsequent head.backward() would refer to this call's input).
     *       Same caveat this campaign's SparseAutoencoder::reconstruct() mission already
     *       documented and accepted for read-only scoring paths.
     * @note The head is the chain's literal last module, not a separately designated
     *       "unembedding" -- this codebase has no such distinct concept. A caller-specifiable
     *       head would be an explicit future extension, not assumed here.
     */
    [[nodiscard]] Tensor logit_lens(NodeId node_id) const {
        auto it = activations_.find(node_id);
        if (it == activations_.end()) {
            throw std::invalid_argument(
                "ExplainerContext::logit_lens: node_id has no cached activation from the most recent "
                "forward pass");
        }

        const NodeId head_input_node = static_cast<NodeId>(modules_.size() - 1);
        auto head_input = activations_.find(head_input_node);
        if (head_input == activations_.end()) {
            throw std::invalid_argument(
                "ExplainerContext::logit_lens: no forward pass has run, so the final module's input "
                "shape is unknown");
        }
        if (!(it->second.shape() == head_input->second.shape())) {
            throw std::invalid_argument(
                "ExplainerContext::logit_lens: the node's cached activation shape does not match the "
                "final module's input shape from the most recent forward pass");
        }

        Module* head = modules_.back();
        return head->forward(it->second);
    }

    /**
     * @brief The "attention lens": the per-head attention pattern produced by the attention
     *        layer at node_id during the most recent forward pass -- "what did each head
     *        attend to", keyed by node id rather than by a direct module reference.
     * @param node_id Node whose attention pattern is returned. Must come from the most recent
     *        forward pass and must name a node whose op_type() is OpType::Attention.
     * @return That layer's (N, num_heads, L, L) softmax output, copied. Raw data only -- no
     *         head aggregation, no ranking, no plotting (campaign
     *         campaign_exai_dl_library_mechanistic_interpretability, Phase 5, is deliberately
     *         scoped data-only and takes no visualization dependency).
     * @throws std::invalid_argument if node_id lies outside the node range the most recent
     *         forward pass produced (including the case where no forward pass has run yet, so
     *         there are no nodes at all), or if it names a node that is not an attention layer
     *         -- both external boundaries (caller-supplied node id), hence throw rather than
     *         PULSATRIX_ASSERT, matching logit_lens and forward_pass_with_patch rather than
     *         activation()'s older pattern, so the behavior is identical in Debug and Release.
     * @note Node id 0 is the input node, which is never an attention layer, so it falls into
     *       the op-type throw rather than needing a case of its own.
     * @note Uses the same node-to-module index correspondence forward_pass_with_patch and
     *       logit_lens already rely on: forward_pass() assigns node id i+1 to modules_[i]'s
     *       output, so modules_[node_id - 1] is the module that produced that node.
     * @note The dynamic_cast is PULSATRIX_ASSERTed, not thrown on: only MultiHeadAttentionModule
     *       reports OpType::Attention anywhere in this codebase, so a failure here would mean
     *       the op-type/module invariant itself broke -- an internal-consistency violation,
     *       not a caller mistake. Checked rather than assumed, per the assert-vs-throw
     *       classification in context_tdd_adversarial_boundary_testing.md.
     * @note Reads the module's own cache, so it reflects that module's most recent forward()
     *       -- which, for a module driven only through this context, is the most recent
     *       forward_pass()/forward_pass_with_patch() call. A patched pass's pattern is the
     *       real pattern that ran, which is exactly what a causal-intervention study wants.
     */
    [[nodiscard]] Tensor attention_weights(NodeId node_id) const {
        if (node_id >= graph_.node_count() || node_id > modules_.size()) {
            throw std::invalid_argument(
                "ExplainerContext::attention_weights: node_id is outside the range of nodes the most "
                "recent forward pass produced (or no forward pass has run yet)");
        }
        if (graph_.node(node_id).op_type() != OpType::Attention) {
            throw std::invalid_argument(
                "ExplainerContext::attention_weights: node_id does not name an attention layer");
        }

        auto* mha = dynamic_cast<MultiHeadAttentionModule*>(modules_[node_id - 1]);
        PULSATRIX_ASSERT(mha != nullptr);
        return Tensor(mha->last_attention_weights());
    }

    /**
     * @brief Builds a CircuitGraph for this chain at the given input: every node scored by
     *        how much zeroing it changes the network's output, plus the chain's edges.
     * @param input Input the circuit is built at. A circuit graph is input-conditional --
     *        ablation importance is "how much does this node matter *for this input*", not
     *        a property of the weights alone.
     * @return A self-contained CircuitGraph: one CircuitNode per graph node (in topological
     *         order), and one CircuitEdge per adjacent pair. Raw data only; viz/document.hpp exports
     *         it as JSON and CircuitGraphView draws it.
     * @note Scoring method: for each non-output node, forward_pass_with_patch() substitutes
     *       a zero-Tensor of that node's own natural shape, and ablation_effect is the L2
     *       distance between that patched output and the real, unpatched one -- the standard
     *       "ablation importance" score, built entirely from Phase 4's patching primitive
     *       with no new causal-inference machinery.
     * @note The zero patch is a copy of the node's own cached activation, filled with 0.0f,
     *       so it matches that node's natural shape, backend and device by construction --
     *       Tensor exposes no backend() accessor to rebuild one from a Shape alone.
     * @note The output node's ablation_effect is 0.0f **by convention, not by computation**:
     *       forward_pass_with_patch() on the output node returns the patch value itself
     *       (nothing runs after it), so self-patching it would score the arbitrary magnitude
     *       of the real output rather than any causal quantity. It is still present in
     *       nodes(), for structural completeness.
     * @note Edges are the chain's inherent adjacency (i -> i+1), weighted by node i's own
     *       ablation_effect. This is exact only because every graph this codebase builds is
     *       a single-parent linear chain (forward_pass()'s loop); see CircuitEdge::weight.
     * @note Cost is one forward pass per node plus two unpatched passes -- O(node_count)
     *       forward passes. Fine for the small chains this codebase builds; a large network
     *       would want a sampled or grouped variant, which is not built here.
     * @note Runs an ordinary forward_pass(input) last, so the context is handed back exactly
     *       as a plain forward pass would leave it: cached activations from the real run,
     *       and backward_pass()'s patched-pass guard disarmed. Without that, every caller
     *       would silently inherit the state of the final ablation run.
     */
    [[nodiscard]] CircuitGraph build_circuit_graph(const Tensor& input) {
        const Tensor baseline = forward_pass(input);
        const NodeId output_node = output_node_;
        // Captured before any patching: every patched pass replaces graph_/activations_
        // wholesale, so the baseline's shapes and per-node metadata are read from this
        // self-contained copy rather than from state the loop itself overwrites.
        const ActivationSnapshot clean = activation_snapshot();

        std::vector<CircuitNode> nodes;
        nodes.reserve(clean.node_ids().size());
        for (NodeId id : clean.node_ids()) {
            float ablation_effect = 0.0f;
            if (id != output_node) {
                Tensor zero_patch(clean.activation(id));
                zero_patch.fill(0.0f);
                const Tensor patched = forward_pass_with_patch(input, id, zero_patch);
                ablation_effect = l2_distance(baseline, patched);
            }
            nodes.push_back(CircuitNode{id, clean.op_type(id), clean.label(id), ablation_effect});
        }

        std::vector<CircuitEdge> edges;
        if (!nodes.empty()) {
            edges.reserve(nodes.size() - 1);
            for (size_t i = 0; i + 1 < nodes.size(); ++i) {
                edges.push_back(CircuitEdge{nodes[i].id, nodes[i + 1].id, nodes[i].ablation_effect});
            }
        }

        (void)forward_pass(input);
        return CircuitGraph(std::move(nodes), std::move(edges));
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

    /**
     * @brief Euclidean distance between two same-shaped tensors, flattened.
     * @param a First tensor.
     * @param b Second tensor; must have the same element count as a.
     * @return sqrt(sum((a_i - b_i)^2)), always >= 0.
     * @note PULSATRIX_ASSERT, not throw: the only caller is build_circuit_graph(), which
     *       compares two outputs of the same module chain -- a size mismatch there is an
     *       internal-consistency violation, not a caller mistake, per
     *       cpp_tdd/context_tdd_adversarial_boundary_testing.md's classification.
     */
    [[nodiscard]] static float l2_distance(const Tensor& a, const Tensor& b) {
        PULSATRIX_ASSERT(a.numel() == b.numel());
        float sum_of_squares = 0.0f;
        for (int64_t i = 0; i < a.numel(); ++i) {
            const float diff = a.data()[i] - b.data()[i];
            sum_of_squares += diff * diff;
        }
        return std::sqrt(sum_of_squares);
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

}  // namespace pulsatrix
