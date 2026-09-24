#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "exai/activation_snapshot.hpp"
#include "exai/adam_optimizer.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/explainer_context.hpp"
#include "exai/linear_module.hpp"
#include "exai/relu_module.hpp"
#include "exai/sparse_autoencoder.hpp"

// ExplainerContext (charter Part 2 SS2) is what every explainer gets, regardless of type --
// graph-native explainers (Missions 2-3) use graph()/backward_pass(); this mission ships
// the interface itself, built directly on Module::forward_traced (Mission 0).
namespace exai {
namespace {

class ExplainerContextTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 2):
// the constructor previously had zero validation -- a nullptr in the module vector was
// an unconditional null-pointer dereference on the very next forward_pass() call.
// External boundary per Mission 0's classification table (this is what any future
// Explainer-authoring Python binding, Phase 5 Mission 1, will construct directly).
TEST_F(ExplainerContextTest, ConstructorThrowsOnEmptyModuleVector) {
    EXPECT_THROW(ExplainerContext(std::vector<Module*>{}), std::invalid_argument);
}

TEST_F(ExplainerContextTest, ConstructorThrowsOnNullModulePointer) {
    LinearModule linear(2, 2, &backend);
    EXPECT_THROW(ExplainerContext({&linear, nullptr}), std::invalid_argument);
}

TEST_F(ExplainerContextTest, ForwardPassSingleModuleMatchesDirectForward) {
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 2.0f, 3.0f, 4.0f});
    linear.set_bias({0.5f, -0.5f});

    LinearModule reference(2, 2, &backend);
    reference.set_weight({1.0f, 2.0f, 3.0f, 4.0f});
    reference.set_bias({0.5f, -0.5f});

    ExplainerContext ctx({&linear});
    Tensor input(Shape({1, 2}), &backend, {1.0f, 1.0f});

    Tensor traced_output = ctx.forward_pass(input);
    Tensor direct_output = reference.forward(input);

    for (int64_t i = 0; i < direct_output.numel(); ++i) {
        EXPECT_FLOAT_EQ(traced_output.data()[i], direct_output.data()[i]);
    }
}

TEST_F(ExplainerContextTest, ForwardPassBuildsCorrectGraphStructureForMultiModuleChain) {
    LinearModule linear1(3, 4, &backend);
    ReluModule relu(&backend);
    LinearModule linear2(4, 2, &backend);

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});
    (void)ctx.forward_pass(input);

    // input node + 3 module nodes.
    EXPECT_EQ(ctx.graph().node_count(), 4u);

    std::vector<NodeId> linear_nodes = ctx.graph().nodes_by_op_type(OpType::Linear);
    EXPECT_EQ(linear_nodes.size(), 2u);
    std::vector<NodeId> activation_nodes = ctx.graph().nodes_by_op_type(OpType::Activation);
    EXPECT_EQ(activation_nodes.size(), 1u);
}

TEST_F(ExplainerContextTest, ActivationReturnsCachedValuePerNode) {
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 0.0f, 0.0f, 1.0f});  // identity-ish
    linear.set_bias({0.0f, 0.0f});

    ExplainerContext ctx({&linear});
    Tensor input(Shape({1, 2}), &backend, {7.0f, 9.0f});
    Tensor output = ctx.forward_pass(input);

    std::vector<NodeId> order = ctx.graph().topological_order();
    ASSERT_EQ(order.size(), 2u);
    NodeId input_node = order[0];
    NodeId output_node = order[1];

    const Tensor& cached_input = ctx.activation(input_node);
    EXPECT_FLOAT_EQ(cached_input.data()[0], 7.0f);
    EXPECT_FLOAT_EQ(cached_input.data()[1], 9.0f);

    const Tensor& cached_output = ctx.activation(output_node);
    EXPECT_FLOAT_EQ(cached_output.data()[0], output.data()[0]);
    EXPECT_FLOAT_EQ(cached_output.data()[1], output.data()[1]);
}

TEST_F(ExplainerContextTest, LayerLabelPassesThroughInputNodeLabel) {
    LinearModule linear(2, 2, &backend);
    ExplainerContext ctx({&linear});
    Tensor input(Shape({1, 2}), &backend, {1.0f, 1.0f});
    (void)ctx.forward_pass(input);

    std::vector<NodeId> order = ctx.graph().topological_order();
    NodeId input_node = order[0];

    ASSERT_TRUE(ctx.layer_label(input_node).has_value());
    EXPECT_EQ(ctx.layer_label(input_node).value(), "input");
}

TEST_F(ExplainerContextTest, BackwardPassMatchesDirectChainedBackward) {
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});
    (void)ctx.forward_pass(input);

    Tensor grad_output(Shape({1, 2}), &backend, {1.0f, -1.0f});
    Tensor traced_grad_input = ctx.backward_pass(grad_output);

    // Reference: the existing, already-proven-correct way to compute this gradient --
    // direct chained backward() calls, the same oracle pattern as
    // mission_module_graph_wiring.md's own integration test.
    Tensor direct_grad_h2 = linear2.backward(grad_output);
    Tensor direct_grad_h1 = relu.backward(direct_grad_h2);
    Tensor direct_grad_input = linear1.backward(direct_grad_h1);

    for (int64_t i = 0; i < direct_grad_input.numel(); ++i) {
        EXPECT_FLOAT_EQ(traced_grad_input.data()[i], direct_grad_input.data()[i]) << "mismatch at index " << i;
    }
}

TEST_F(ExplainerContextTest, GradientReturnsCachedValuePerIntermediateNode) {
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});
    (void)ctx.forward_pass(input);

    Tensor grad_output(Shape({1, 2}), &backend, {1.0f, -1.0f});
    (void)ctx.backward_pass(grad_output);

    // Reference: the same oracle Objective 3 of mission_module_graph_wiring.md and
    // Objective 3 of mission_explainer_context.md already used -- direct chained
    // backward() calls.
    Tensor direct_grad_h2 = linear2.backward(grad_output);
    Tensor direct_grad_h1 = relu.backward(direct_grad_h2);

    std::vector<NodeId> order = ctx.graph().topological_order();
    ASSERT_EQ(order.size(), 4u);
    NodeId h1_node = order[1];
    NodeId h2_node = order[2];

    const Tensor& traced_grad_h1 = ctx.gradient(h1_node);
    for (int64_t i = 0; i < direct_grad_h1.numel(); ++i) {
        EXPECT_FLOAT_EQ(traced_grad_h1.data()[i], direct_grad_h1.data()[i]) << "h1 mismatch at index " << i;
    }

    const Tensor& traced_grad_h2 = ctx.gradient(h2_node);
    for (int64_t i = 0; i < direct_grad_h2.numel(); ++i) {
        EXPECT_FLOAT_EQ(traced_grad_h2.data()[i], direct_grad_h2.data()[i]) << "h2 mismatch at index " << i;
    }
}

TEST_F(ExplainerContextTest, RepeatedForwardPassDoesNotLeakStateFromPriorCall) {
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 0.0f, 0.0f, 1.0f});
    linear.set_bias({0.0f, 0.0f});

    ExplainerContext ctx({&linear});

    Tensor first_input(Shape({1, 2}), &backend, {1.0f, 2.0f});
    (void)ctx.forward_pass(first_input);
    EXPECT_EQ(ctx.graph().node_count(), 2u);

    Tensor second_input(Shape({1, 2}), &backend, {10.0f, 20.0f});
    Tensor second_output = ctx.forward_pass(second_input);

    // Still exactly 2 nodes -- a fresh graph each call, not an ever-growing one.
    EXPECT_EQ(ctx.graph().node_count(), 2u);
    EXPECT_FLOAT_EQ(second_output.data()[0], 10.0f);
    EXPECT_FLOAT_EQ(second_output.data()[1], 20.0f);

    std::vector<NodeId> order = ctx.graph().topological_order();
    const Tensor& cached_input = ctx.activation(order[0]);
    EXPECT_FLOAT_EQ(cached_input.data()[0], 10.0f);  // reflects the second call, not the first
    EXPECT_FLOAT_EQ(cached_input.data()[1], 20.0f);
}

// --- ExplainerContext::activation_snapshot() -------------------------------------
// campaign_exai_dl_library_mechanistic_interpretability, Phase 1 Mission 1, Objective 2:
// generalizes the single-node activation() lookup into a self-contained, enumerable
// ActivationSnapshot that survives subsequent forward_pass() calls.

TEST_F(ExplainerContextTest, ActivationSnapshotMatchesActivationForEveryNode) {
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});
    (void)ctx.forward_pass(input);

    ActivationSnapshot snapshot = ctx.activation_snapshot();

    // Equivalence oracle: the already-proven activation() accessor, at capture time.
    ASSERT_EQ(snapshot.node_ids().size(), 4u);
    for (NodeId id : snapshot.node_ids()) {
        const Tensor& expected = ctx.activation(id);
        const Tensor& actual = snapshot.activation(id);
        ASSERT_EQ(actual.numel(), expected.numel()) << "numel mismatch at node " << id;
        for (int64_t i = 0; i < expected.numel(); ++i) {
            EXPECT_FLOAT_EQ(actual.data()[i], expected.data()[i]) << "node " << id << " index " << i;
        }
    }
}

TEST_F(ExplainerContextTest, ActivationSnapshotNodeIdsMatchGraphTopologicalOrder) {
    LinearModule linear1(3, 4, &backend);
    ReluModule relu(&backend);
    LinearModule linear2(4, 2, &backend);

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});
    (void)ctx.forward_pass(input);

    EXPECT_EQ(ctx.activation_snapshot().node_ids(), ctx.graph().topological_order());
}

TEST_F(ExplainerContextTest, ActivationSnapshotMetadataMatchesGraphNodes) {
    LinearModule linear1(3, 4, &backend);
    ReluModule relu(&backend);
    LinearModule linear2(4, 2, &backend);

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});
    (void)ctx.forward_pass(input);

    ActivationSnapshot snapshot = ctx.activation_snapshot();
    for (NodeId id : snapshot.node_ids()) {
        EXPECT_EQ(snapshot.op_type(id), ctx.graph().node(id).op_type()) << "op_type mismatch at node " << id;
        EXPECT_EQ(snapshot.label(id), ctx.graph().node(id).label()) << "label mismatch at node " << id;
    }

    // Spot-check the two ends against what the graph is known to be built from: the
    // input node is labeled, module nodes are not.
    EXPECT_EQ(snapshot.op_type(snapshot.node_ids().front()), OpType::Elementwise);
    ASSERT_TRUE(snapshot.label(snapshot.node_ids().front()).has_value());
    EXPECT_EQ(snapshot.label(snapshot.node_ids().front()).value(), "input");
    EXPECT_EQ(snapshot.op_type(snapshot.node_ids().back()), OpType::Linear);
    EXPECT_FALSE(snapshot.label(snapshot.node_ids().back()).has_value());
}

// THE capability this mission exists to add. forward_pass() clears and rebuilds
// activations_ in place, so before this method there was no way to hold two runs'
// activations at once -- exactly what activation patching (Phase 4) needs: a "clean"
// and a "corrupted" run, both alive, both correct, NodeId-comparable.
TEST_F(ExplainerContextTest, SnapshotsFromTwoForwardPassesStayIndependentAndCorrect) {
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 0.0f, 0.0f, 1.0f});  // identity
    linear.set_bias({0.0f, 0.0f});

    ExplainerContext ctx({&linear});

    Tensor first_input(Shape({1, 2}), &backend, {1.0f, 2.0f});
    (void)ctx.forward_pass(first_input);
    ActivationSnapshot first = ctx.activation_snapshot();

    Tensor second_input(Shape({1, 2}), &backend, {10.0f, 20.0f});
    (void)ctx.forward_pass(second_input);
    ActivationSnapshot second = ctx.activation_snapshot();

    // Same node ids across both passes -- the module chain is fixed, so the graphs are
    // structurally identical and the two snapshots are directly comparable.
    ASSERT_EQ(first.node_ids(), second.node_ids());
    ASSERT_EQ(first.node_ids().size(), 2u);

    NodeId input_node = first.node_ids()[0];
    NodeId output_node = first.node_ids()[1];

    // The first snapshot still holds the FIRST pass's values -- taking the second
    // snapshot (and running the second forward pass) did not disturb it.
    EXPECT_FLOAT_EQ(first.activation(input_node).data()[0], 1.0f);
    EXPECT_FLOAT_EQ(first.activation(input_node).data()[1], 2.0f);
    EXPECT_FLOAT_EQ(first.activation(output_node).data()[0], 1.0f);
    EXPECT_FLOAT_EQ(first.activation(output_node).data()[1], 2.0f);

    EXPECT_FLOAT_EQ(second.activation(input_node).data()[0], 10.0f);
    EXPECT_FLOAT_EQ(second.activation(input_node).data()[1], 20.0f);
    EXPECT_FLOAT_EQ(second.activation(output_node).data()[0], 10.0f);
    EXPECT_FLOAT_EQ(second.activation(output_node).data()[1], 20.0f);

    // And the live context reflects only the most recent pass, as it always has.
    EXPECT_FLOAT_EQ(ctx.activation(input_node).data()[0], 10.0f);
}

// Adversarial/boundary (policy item 2, sequencing violation): snapshotting before any
// forward_pass(). Well-defined, not an error -- there is simply nothing cached yet, so
// the result is an empty snapshot. No throw, no assert, no undefined behavior; this
// test must therefore pass identically in Debug and Release.
TEST_F(ExplainerContextTest, ActivationSnapshotBeforeAnyForwardPassIsEmpty) {
    LinearModule linear(2, 2, &backend);
    ExplainerContext ctx({&linear});

    ActivationSnapshot snapshot = ctx.activation_snapshot();

    EXPECT_TRUE(snapshot.node_ids().empty());
}

// --- ExplainerContext::forward_pass_with_patch() ---------------------------------
// campaign_exai_dl_library_mechanistic_interpretability, Phase 4 Mission 1: the causal
// intervention primitive. Runs the same forward loop as forward_pass(), but substitutes
// patch_value for the natural output of the module producing patch_node_id -- everything
// downstream computes from the substituted value, everything upstream is untouched.
// Node ids are assigned deterministically: 0 = input, then one per module in order.

TEST_F(ExplainerContextTest, ForwardPassWithPatchLeavesUpstreamActivationsUntouched) {
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});

    (void)ctx.forward_pass(input);
    ActivationSnapshot clean = ctx.activation_snapshot();
    ASSERT_EQ(clean.node_ids().size(), 4u);
    const NodeId input_node = clean.node_ids()[0];
    const NodeId h1_node = clean.node_ids()[1];   // linear1 output
    const NodeId h2_node = clean.node_ids()[2];   // relu output (the patch target)
    const NodeId out_node = clean.node_ids()[3];  // linear2 output

    Tensor patch(Shape({1, 4}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor patched_output = ctx.forward_pass_with_patch(input, h2_node, patch);

    // Upstream of the patch point: bit-identical to the unpatched run. A forward-only
    // computation cannot be influenced downstream-to-upstream -- confirmed, not assumed.
    for (NodeId id : {input_node, h1_node}) {
        const Tensor& before = clean.activation(id);
        const Tensor& after = ctx.activation(id);
        ASSERT_EQ(after.numel(), before.numel()) << "numel mismatch at node " << id;
        for (int64_t i = 0; i < before.numel(); ++i) {
            EXPECT_FLOAT_EQ(after.data()[i], before.data()[i]) << "node " << id << " index " << i;
        }
    }

    // The patched node caches the substituted value, not its natural output.
    const Tensor& cached_patch = ctx.activation(h2_node);
    ASSERT_EQ(cached_patch.numel(), 4);
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(cached_patch.data()[i], patch.data()[i]) << "index " << i;
    }

    // Downstream: linear2 applied to the patched value, not to the natural relu output.
    LinearModule reference(4, 2, &backend);
    reference.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    reference.set_bias({0.05f, -0.05f});
    Tensor expected = reference.forward(patch);
    ASSERT_EQ(patched_output.numel(), expected.numel());
    for (int64_t i = 0; i < expected.numel(); ++i) {
        EXPECT_FLOAT_EQ(patched_output.data()[i], expected.data()[i]) << "index " << i;
        EXPECT_FLOAT_EQ(ctx.activation(out_node).data()[i], expected.data()[i]) << "index " << i;
    }
}

// Degenerate-but-coherent base case: patching the input node is just running the chain on
// a different input. Explicitly supported, not an error (mission Exit Gate bullet 2).
TEST_F(ExplainerContextTest, ForwardPassWithPatchAtInputNodeMatchesForwardPassOnThatInput) {
    LinearModule linear1(2, 3, &backend);
    linear1.set_weight({0.5f, -0.25f, 1.5f, 2.0f, 0.75f, -1.0f});
    linear1.set_bias({0.1f, 0.2f, -0.3f});
    ReluModule relu(&backend);

    ExplainerContext ctx({&linear1, &relu});
    Tensor original(Shape({1, 2}), &backend, {1.0f, -2.0f});
    Tensor replacement(Shape({1, 2}), &backend, {3.0f, 0.5f});

    Tensor reference_output = ctx.forward_pass(replacement);

    std::vector<NodeId> order = ctx.graph().topological_order();
    const NodeId input_node = order[0];

    Tensor patched_output = ctx.forward_pass_with_patch(original, input_node, replacement);

    ASSERT_EQ(patched_output.numel(), reference_output.numel());
    for (int64_t i = 0; i < reference_output.numel(); ++i) {
        EXPECT_FLOAT_EQ(patched_output.data()[i], reference_output.data()[i]) << "index " << i;
    }
    // The cached input activation is the replacement, not the original.
    EXPECT_FLOAT_EQ(ctx.activation(input_node).data()[0], 3.0f);
    EXPECT_FLOAT_EQ(ctx.activation(input_node).data()[1], 0.5f);
}

// Degenerate at the other end: nothing runs after the output node, so the patched value
// is returned verbatim.
TEST_F(ExplainerContextTest, ForwardPassWithPatchAtOutputNodeReturnsPatchValueExactly) {
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 0.0f, 0.0f, 1.0f});
    linear.set_bias({0.0f, 0.0f});

    ExplainerContext ctx({&linear});
    Tensor input(Shape({1, 2}), &backend, {7.0f, 9.0f});
    (void)ctx.forward_pass(input);
    const NodeId output_node = ctx.graph().topological_order().back();

    Tensor patch(Shape({1, 2}), &backend, {-1.5f, 4.25f});
    Tensor patched_output = ctx.forward_pass_with_patch(input, output_node, patch);

    ASSERT_EQ(patched_output.numel(), 2);
    EXPECT_FLOAT_EQ(patched_output.data()[0], -1.5f);
    EXPECT_FLOAT_EQ(patched_output.data()[1], 4.25f);
    EXPECT_FLOAT_EQ(ctx.activation(output_node).data()[0], -1.5f);
    EXPECT_FLOAT_EQ(ctx.activation(output_node).data()[1], 4.25f);
}

// Adversarial (external boundary -- patch_value is caller-supplied): a shape mismatch
// against the target node's natural output shape throws at the patch site, rather than
// failing confusingly deep inside some unrelated downstream module. Must pass identically
// in Debug and Release (throw, not EXAI_ASSERT).
TEST_F(ExplainerContextTest, ForwardPassWithPatchThrowsOnShapeMismatchedPatchValue) {
    LinearModule linear1(2, 3, &backend);
    linear1.set_weight({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
    linear1.set_bias({0.0f, 0.0f, 0.0f});
    ReluModule relu(&backend);

    ExplainerContext ctx({&linear1, &relu});
    Tensor input(Shape({1, 2}), &backend, {1.0f, 2.0f});
    Tensor reference_output = ctx.forward_pass(input);
    const NodeId h1_node = ctx.graph().topological_order()[1];  // natural shape (1, 3)

    Tensor wrong_shape(Shape({1, 2}), &backend, {1.0f, 1.0f});
    EXPECT_THROW((void)ctx.forward_pass_with_patch(input, h1_node, wrong_shape), std::invalid_argument);

    // The failed patch attempt must leave the context fully usable: a subsequent normal
    // forward_pass() still produces the correct result and a correct graph.
    Tensor after = ctx.forward_pass(input);
    EXPECT_EQ(ctx.graph().node_count(), 3u);
    ASSERT_EQ(after.numel(), reference_output.numel());
    for (int64_t i = 0; i < reference_output.numel(); ++i) {
        EXPECT_FLOAT_EQ(after.data()[i], reference_output.data()[i]) << "index " << i;
    }
}

// Adversarial (external boundary): a node id larger than any node this forward pass will
// produce. Node ids run 0 (input) .. modules.size().
TEST_F(ExplainerContextTest, ForwardPassWithPatchThrowsOnOutOfRangePatchNodeId) {
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 0.0f, 0.0f, 1.0f});
    linear.set_bias({0.0f, 0.0f});

    ExplainerContext ctx({&linear});
    Tensor input(Shape({1, 2}), &backend, {1.0f, 2.0f});
    Tensor patch(Shape({1, 2}), &backend, {0.0f, 0.0f});

    // Valid ids here are 0 (input) and 1 (the single module's output).
    EXPECT_THROW((void)ctx.forward_pass_with_patch(input, 2, patch), std::invalid_argument);
    EXPECT_THROW((void)ctx.forward_pass_with_patch(input, 99, patch), std::invalid_argument);

    // Still usable afterwards.
    Tensor after = ctx.forward_pass(input);
    EXPECT_FLOAT_EQ(after.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(after.data()[1], 2.0f);
}

// --- backward_pass() after a patched forward (Phase 4 Mission 2, Objective 2) ------
//
// A patched activation is a *constant substitution*, not a differentiable function of the
// real input. Every module still registers its ordinary backward closure during
// forward_pass_with_patch (forward_traced does that unconditionally), so Autograd::backward
// would happily walk the chain and return a number -- a number computed through a link that
// does not exist in the computation it claims to differentiate. Silently wrong gradients are
// the worst failure mode this codebase can ship, so the sequencing is rejected outright,
// following the same "turn silent UB into a loud failure" principle as Phase 1.5's device
// guards. External boundary (a caller-sequencing mistake) -> throw, not EXAI_ASSERT, so the
// behavior is identical in Debug and Release.

TEST_F(ExplainerContextTest, BackwardPassThrowsAfterAPatchedForwardPass) {
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});
    Tensor patch(Shape({1, 4}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor grad_output(Shape({1, 2}), &backend, {1.0f, -1.0f});

    (void)ctx.forward_pass_with_patch(input, /*patch_node_id=*/2, patch);

    EXPECT_THROW((void)ctx.backward_pass(grad_output), std::logic_error);
}

// The guard is per-call history, not a one-way latch: an ordinary forward_pass() re-arms
// backward_pass() completely. Verified against the same direct-chained-backward oracle
// BackwardPassMatchesDirectChainedBackward uses, so this asserts the gradient is *right*,
// not merely that no exception escaped.
TEST_F(ExplainerContextTest, BackwardPassSucceedsAfterAnUnpatchedForwardPassFollowingAPatchedOne) {
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});
    Tensor patch(Shape({1, 4}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor grad_output(Shape({1, 2}), &backend, {1.0f, -1.0f});

    (void)ctx.forward_pass_with_patch(input, /*patch_node_id=*/2, patch);
    (void)ctx.forward_pass(input);
    Tensor traced_grad_input = ctx.backward_pass(grad_output);

    Tensor direct_grad_h2 = linear2.backward(grad_output);
    Tensor direct_grad_h1 = relu.backward(direct_grad_h2);
    Tensor direct_grad_input = linear1.backward(direct_grad_h1);

    ASSERT_EQ(traced_grad_input.numel(), direct_grad_input.numel());
    for (int64_t i = 0; i < direct_grad_input.numel(); ++i) {
        EXPECT_FLOAT_EQ(traced_grad_input.data()[i], direct_grad_input.data()[i]) << "mismatch at index " << i;
    }
}

// Adversarial: a patched forward that *threw* still arms the guard, and deliberately so.
// forward_pass_with_patch only commits its graph/autograd/activation state on success, so a
// throw leaves those exactly as the prior forward_pass() left them -- but it cannot un-run
// the modules that already executed, and Module::forward_traced's registered closure calls
// module->backward(), which reads the module's *own* internal cache from its most recent
// forward(). Those caches now belong to the aborted patched run. A backward through the
// stale graph would therefore mix one run's topology with another run's cached operands.
// Re-arming requires a real forward_pass(), which this test also confirms.
TEST_F(ExplainerContextTest, BackwardPassThrowsAfterAPatchedForwardPassThatItselfThrew) {
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 0.0f, 0.0f, 1.0f});
    linear.set_bias({0.0f, 0.0f});

    ExplainerContext ctx({&linear});
    Tensor input(Shape({1, 2}), &backend, {1.0f, 2.0f});
    Tensor wrong_shape_patch(Shape({1, 3}), &backend, {0.0f, 0.0f, 0.0f});
    Tensor grad_output(Shape({1, 2}), &backend, {1.0f, -1.0f});

    (void)ctx.forward_pass(input);
    EXPECT_THROW((void)ctx.forward_pass_with_patch(input, /*patch_node_id=*/1, wrong_shape_patch),
                 std::invalid_argument);

    EXPECT_THROW((void)ctx.backward_pass(grad_output), std::logic_error);

    (void)ctx.forward_pass(input);
    EXPECT_NO_THROW((void)ctx.backward_pass(grad_output));
}

// --- Hand-derivable causal-effect verification (Phase 4 Mission 1, Objective 2) ---
// The mission's actual acceptance criterion: not "patching changes the output," but
// "patching produces the exact output causal-patching methodology predicts."
//
// Fixed network, all 1x1 so every intermediate is a single number:
//     Linear1: h_pre = x * 3.0 + (-1.0)
//     Relu:    h     = max(h_pre, 0)
//     Linear2: y     = h * 2.0 + 0.5
namespace {

constexpr float kW1 = 3.0f;
constexpr float kB1 = -1.0f;
constexpr float kW2 = 2.0f;
constexpr float kB2 = 0.5f;

}  // namespace

// Arithmetic, derived by hand before the test was run:
//   clean     x = 2.0   -> h_pre = 2.0*3 - 1 =  5.0 -> h =  5.0 -> y =  5.0*2 + 0.5 = 10.5
//   corrupted x = 4.0   -> h_pre = 4.0*3 - 1 = 11.0 -> h = 11.0 -> y = 11.0*2 + 0.5 = 22.5
//   clean run, hidden patched with the corrupted run's h = 11.0:
//                          y = 11.0*2 + 0.5 = 22.5
// The patched output equals the corrupted output exactly because the patched node is the
// single bottleneck on the clean run's only path -- full causal-effect restoration.
TEST_F(ExplainerContextTest, ForwardPassWithPatchReproducesHandDerivedCausalEffect) {
    LinearModule linear1(1, 1, &backend);
    linear1.set_weight({kW1});
    linear1.set_bias({kB1});
    ReluModule relu(&backend);
    LinearModule linear2(1, 1, &backend);
    linear2.set_weight({kW2});
    linear2.set_bias({kB2});

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor clean_input(Shape({1, 1}), &backend, {2.0f});
    Tensor corrupted_input(Shape({1, 1}), &backend, {4.0f});

    Tensor clean_output = ctx.forward_pass(clean_input);
    ActivationSnapshot clean = ctx.activation_snapshot();
    ASSERT_EQ(clean.node_ids().size(), 4u);
    const NodeId hidden_node = clean.node_ids()[2];  // post-ReLU
    EXPECT_FLOAT_EQ(clean.activation(hidden_node).data()[0], 5.0f);
    EXPECT_FLOAT_EQ(clean_output.data()[0], 10.5f);

    Tensor corrupted_output = ctx.forward_pass(corrupted_input);
    ActivationSnapshot corrupted = ctx.activation_snapshot();
    EXPECT_FLOAT_EQ(corrupted.activation(hidden_node).data()[0], 11.0f);
    EXPECT_FLOAT_EQ(corrupted_output.data()[0], 22.5f);

    // The intervention: clean run, hidden activation replaced by the corrupted run's.
    Tensor patched_output = ctx.forward_pass_with_patch(clean_input, hidden_node, corrupted.activation(hidden_node));

    EXPECT_FLOAT_EQ(patched_output.data()[0], 22.5f);
    // Upstream is untouched: the clean run's own pre-ReLU value still stands.
    EXPECT_FLOAT_EQ(ctx.activation(clean.node_ids()[1]).data()[0], 5.0f);
    EXPECT_FLOAT_EQ(ctx.activation(clean.node_ids()[0]).data()[0], 2.0f);
}

// Second case, with the ReLU's zero-clamp engaged in the clean run but not the corrupted
// one -- so the patch has to carry a value the clean run's own ReLU would never produce:
//   clean     x = 0.25 -> h_pre = 0.25*3 - 1 = -0.25 -> h = 0.0 (clamped) -> y = 0.5
//   corrupted x = 2.0  -> h_pre = 2.0*3  - 1 =  5.00 -> h = 5.0           -> y = 10.5
//   clean run, hidden patched with 5.0 -> y = 5.0*2 + 0.5 = 10.5
TEST_F(ExplainerContextTest, ForwardPassWithPatchSubstitutesPostReluValueWhenClampDiffers) {
    LinearModule linear1(1, 1, &backend);
    linear1.set_weight({kW1});
    linear1.set_bias({kB1});
    ReluModule relu(&backend);
    LinearModule linear2(1, 1, &backend);
    linear2.set_weight({kW2});
    linear2.set_bias({kB2});

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor clean_input(Shape({1, 1}), &backend, {0.25f});
    Tensor corrupted_input(Shape({1, 1}), &backend, {2.0f});

    Tensor clean_output = ctx.forward_pass(clean_input);
    ActivationSnapshot clean = ctx.activation_snapshot();
    const NodeId pre_relu_node = clean.node_ids()[1];
    const NodeId hidden_node = clean.node_ids()[2];
    EXPECT_FLOAT_EQ(clean.activation(pre_relu_node).data()[0], -0.25f);
    EXPECT_FLOAT_EQ(clean.activation(hidden_node).data()[0], 0.0f);  // clamp engaged
    EXPECT_FLOAT_EQ(clean_output.data()[0], 0.5f);

    Tensor corrupted_output = ctx.forward_pass(corrupted_input);
    ActivationSnapshot corrupted = ctx.activation_snapshot();
    EXPECT_FLOAT_EQ(corrupted.activation(pre_relu_node).data()[0], 5.0f);
    EXPECT_FLOAT_EQ(corrupted.activation(hidden_node).data()[0], 5.0f);  // clamp not engaged
    EXPECT_FLOAT_EQ(corrupted_output.data()[0], 10.5f);

    Tensor patched_output = ctx.forward_pass_with_patch(clean_input, hidden_node, corrupted.activation(hidden_node));
    EXPECT_FLOAT_EQ(patched_output.data()[0], 10.5f);
    // Upstream untouched: the clean run's pre-ReLU value is still its own negative value.
    EXPECT_FLOAT_EQ(ctx.activation(pre_relu_node).data()[0], -0.25f);
}

// The decisive check that the patch substitutes the *post*-ReLU value rather than feeding
// it back through ReLU a second time: patch the post-ReLU node with a NEGATIVE value, which
// no ReLU output can ever be.
//   patched h = -3.0 -> y = -3.0*2 + 0.5 = -5.5
// If ReLU were (incorrectly) re-applied to the patched value, h would clamp to 0.0 and y
// would be 0.5 -- a value this test explicitly rejects.
TEST_F(ExplainerContextTest, ForwardPassWithPatchDoesNotReapplyReluToThePatchedValue) {
    LinearModule linear1(1, 1, &backend);
    linear1.set_weight({kW1});
    linear1.set_bias({kB1});
    ReluModule relu(&backend);
    LinearModule linear2(1, 1, &backend);
    linear2.set_weight({kW2});
    linear2.set_bias({kB2});

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({1, 1}), &backend, {2.0f});
    (void)ctx.forward_pass(input);
    const NodeId hidden_node = ctx.graph().topological_order()[2];

    Tensor negative_patch(Shape({1, 1}), &backend, {-3.0f});
    Tensor patched_output = ctx.forward_pass_with_patch(input, hidden_node, negative_patch);

    EXPECT_FLOAT_EQ(patched_output.data()[0], -5.5f);
    EXPECT_NE(patched_output.data()[0], 0.5f);  // what a re-applied ReLU would have produced
    EXPECT_FLOAT_EQ(ctx.activation(hidden_node).data()[0], -3.0f);
}

// --- SAE-reconstruction patching, end to end ---------------------------------------
//
// campaign_exai_dl_library_mechanistic_interpretability, Phase 4 Mission 3 (follow-up
// remediation 2). SparseAutoencoder::reconstruct() (Phase 4 Mission 2) and
// forward_pass_with_patch() (Phase 4 Mission 1) were each proven correct in isolation;
// nothing proved they compose, which is exactly what Phase 4's exit gate claims ("swap a
// cached activation ... or substitute an SAE reconstruction ... and re-run the downstream
// computation"). This is that proof: real network -> real cached hidden activation -> SAE
// trained on activations from that same layer -> reconstruction patched back in.
//
// Deliberately *not* a hand-derived exact-value test, unlike the three above. Those fix
// every weight to a round number precisely so the expected output can be computed on paper;
// here the patched value comes out of a trained SAE, so the only honest assertions are
// structural (shape, finiteness) and directional. The directional one is the load-bearing
// check: a lossy-but-real reconstruction, patched in, must move the output *less* than a
// garbage patch (an all-zero tensor of the same shape) does. That distinguishes "the
// composition transports real information" from "the composition merely doesn't crash" --
// a test that only asserted no-throw would pass just as happily if reconstruct() returned
// noise.
TEST_F(ExplainerContextTest, SaeReconstructionPatchedIntoAForwardPassStaysCloserThanAGarbagePatch) {
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.9f, -0.4f, 0.2f, 0.7f, -0.3f, 0.8f, 0.5f, -0.6f, 0.4f, 0.1f, -0.7f, 0.3f});
    linear1.set_bias({0.1f, -0.2f, 0.3f, 0.0f});
    ReluModule relu(&backend);
    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.2f, -0.4f, 0.6f, 0.3f, 0.9f, 0.8f, -0.1f});
    linear2.set_bias({0.05f, -0.05f});

    ExplainerContext ctx({&linear1, &relu, &linear2});

    // A small set of related inputs -- the SAE's training distribution is the hidden-layer
    // activations this network actually produces, not synthetic data.
    const std::vector<std::vector<float>> inputs = {{1.00f, 0.50f, -0.30f},  {0.90f, 0.60f, -0.20f},
                                                    {1.10f, 0.40f, -0.35f},  {1.05f, 0.55f, -0.25f},
                                                    {0.95f, 0.45f, -0.30f},  {1.00f, 0.50f, -0.28f}};

    // Probe input: the run whose hidden activation is reconstructed and patched back in.
    Tensor probe_input(Shape({1, 3}), &backend, inputs[0]);
    Tensor unpatched_output = ctx.forward_pass(probe_input);
    ASSERT_EQ(unpatched_output.shape(), Shape({1, 2}));

    ASSERT_EQ(ctx.graph().topological_order().size(), 4u);
    const NodeId hidden_node = ctx.graph().topological_order()[2];  // post-ReLU
    const Tensor probe_activation(ctx.activation(hidden_node));
    ASSERT_EQ(probe_activation.shape(), Shape({1, 4}));

    // Collect one hidden activation per input into a single (N, 4) training batch. The
    // copies matter: activation() reads a cache the next forward_pass() overwrites.
    std::vector<float> batch_data;
    for (const std::vector<float>& values : inputs) {
        Tensor in(Shape({1, 3}), &backend, values);
        (void)ctx.forward_pass(in);
        const Tensor& hidden = ctx.activation(hidden_node);
        for (int64_t i = 0; i < hidden.numel(); ++i) {
            batch_data.push_back(hidden.data()[i]);
        }
    }
    const int64_t batch_n = static_cast<int64_t>(inputs.size());
    Tensor activation_batch(Shape({batch_n, 4}), &backend, batch_data);

    // Overcomplete SAE sized to the hidden layer's width, trained briefly on those
    // activations. Short training on purpose: the reconstruction should be lossy -- a
    // perfect one would make the comparison below trivially true for the wrong reason.
    SparseAutoencoder sae(/*dim=*/4, /*hidden_dim=*/16, /*l1_lambda=*/0.001f, &backend, /*seed=*/7);
    AdamOptimizer optimizer(0.05f, &backend);
    for (int epoch = 0; epoch < 200; ++epoch) {
        (void)sae.train_step(activation_batch, optimizer);
    }

    const Tensor reconstruction = sae.reconstruct(probe_activation);
    ASSERT_EQ(reconstruction.shape(), probe_activation.shape());
    // Lossy, not bit-identical -- otherwise the comparison below is not testing anything.
    bool differs_from_the_real_activation = false;
    for (int64_t i = 0; i < reconstruction.numel(); ++i) {
        ASSERT_TRUE(std::isfinite(reconstruction.data()[i]));
        if (reconstruction.data()[i] != probe_activation.data()[i]) {
            differs_from_the_real_activation = true;
        }
    }
    EXPECT_TRUE(differs_from_the_real_activation);

    // The composition itself: reconstruct -> patch -> re-run downstream.
    Tensor reconstruction_patched_output = ctx.forward_pass_with_patch(probe_input, hidden_node, reconstruction);
    EXPECT_EQ(reconstruction_patched_output.shape(), Shape({1, 2}));
    for (int64_t i = 0; i < reconstruction_patched_output.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(reconstruction_patched_output.data()[i]));
    }

    // The control: a same-shaped patch carrying no information about this activation.
    Tensor garbage_patch(probe_activation.shape(), &backend);
    garbage_patch.fill(0.0f);
    Tensor garbage_patched_output = ctx.forward_pass_with_patch(probe_input, hidden_node, garbage_patch);
    ASSERT_EQ(garbage_patched_output.shape(), Shape({1, 2}));

    float reconstruction_distance = 0.0f;
    float garbage_distance = 0.0f;
    for (int64_t i = 0; i < unpatched_output.numel(); ++i) {
        const float reconstruction_diff = reconstruction_patched_output.data()[i] - unpatched_output.data()[i];
        const float garbage_diff = garbage_patched_output.data()[i] - unpatched_output.data()[i];
        reconstruction_distance += reconstruction_diff * reconstruction_diff;
        garbage_distance += garbage_diff * garbage_diff;
    }

    EXPECT_LT(reconstruction_distance, garbage_distance)
        << "reconstruction-patched distance " << reconstruction_distance << " vs. garbage-patched "
        << garbage_distance;
    // The reconstruction is lossy, so it must not land exactly on the unpatched output
    // either -- the assertion above is a *bound*, not an equality in disguise.
    EXPECT_GT(reconstruction_distance, 0.0f);
}

// --- ExplainerContext::logit_lens() (Phase 5 Mission 1, Objective 1) ----------------
//
// The logit lens: run the chain's *final* module (its read-out head) directly on an
// intermediate cached activation, answering "what would the network predict if this
// layer's representation were already final?" Data only -- a raw Tensor, no plotting.

// The free, exact correctness oracle. The node immediately before the final module is,
// by construction, exactly the input the final module actually received during the real
// forward_pass(). So logit_lens() there is not an approximation of the real output -- it
// is literally the same computation forward_pass() already performed, which makes this an
// element-wise EXPECT_FLOAT_EQ with no tolerance, not a bound.
TEST_F(ExplainerContextTest, LogitLensAtTheNodeBeforeTheFinalModuleReproducesTheRealOutputExactly) {
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});

    Tensor real_output = ctx.forward_pass(input);
    const NodeId node_before_head = ctx.graph().topological_order()[2];  // post-ReLU, the head's input

    Tensor lens = ctx.logit_lens(node_before_head);

    ASSERT_EQ(lens.shape(), real_output.shape());
    for (int64_t i = 0; i < real_output.numel(); ++i) {
        EXPECT_FLOAT_EQ(lens.data()[i], real_output.data()[i]) << "index " << i;
    }
}

// Degenerate single-module chain: the "node before the final module" is the input node
// itself, and the same exact-oracle identity must hold there.
TEST_F(ExplainerContextTest, LogitLensAtTheInputNodeOfASingleModuleChainReproducesTheRealOutputExactly) {
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 2.0f, 3.0f, 4.0f});
    linear.set_bias({0.5f, -0.5f});

    ExplainerContext ctx({&linear});
    Tensor input(Shape({1, 2}), &backend, {1.0f, 1.0f});

    Tensor real_output = ctx.forward_pass(input);
    Tensor lens = ctx.logit_lens(ctx.graph().topological_order()[0]);

    ASSERT_EQ(lens.shape(), real_output.shape());
    for (int64_t i = 0; i < real_output.numel(); ++i) {
        EXPECT_FLOAT_EQ(lens.data()[i], real_output.data()[i]) << "index " << i;
    }
}

// The meaningful-divergence case, on a constant-hidden-width chain -- the structural
// analogue of a transformer's fixed-width residual stream with an unembedding head, which
// is what makes the logit lens interpretable at *every* depth rather than only at the end.
//
//   Linear1(2,2): W1 = [[1, 0], [0, 2]],      b1 = [ 0.5, -0.5]
//   Relu
//   Linear2(2,2): W2 = [[2, 0], [0, 1]],      b2 = [-1.0,  1.0]
//   Relu
//   Linear3(2,2): W3 = [[1, -1], [0.5, 2]],   b3 = [ 0.25, -0.25]    <- the read-out head
// (weight layout is (in_features, out_features), row-major: y_j = sum_i x_i * W[i][j] + b_j)
//
// Hand-derived, before the test was run, for x = [1.0, 2.0]:
//   h1_pre = [1*1 + 2*0 + 0.5,   1*0 + 2*2 - 0.5]   = [1.5, 3.5]
//   h1     = relu(h1_pre)                            = [1.5, 3.5]   (no clamp)
//   h2_pre = [1.5*2 + 3.5*0 - 1,  1.5*0 + 3.5*1 + 1] = [2.0, 4.5]
//   h2     = relu(h2_pre)                            = [2.0, 4.5]   (no clamp)
//   y      = [2.0*1 + 4.5*0.5 + 0.25,  2.0*-1 + 4.5*2 - 0.25]       = [4.50, 6.75]
// Applying the head at earlier depths:
//   lens(input = [1.0, 2.0]) = [1*1 + 2*0.5 + 0.25,  1*-1 + 2*2 - 0.25]     = [2.25, 2.75]
//   lens(h1    = [1.5, 3.5]) = [1.5*1 + 3.5*0.5 + 0.25, 1.5*-1 + 3.5*2 - 0.25] = [3.50, 5.25]
//   lens(h2    = [2.0, 4.5]) = the real output                                  = [4.50, 6.75]
// Three genuinely different read-outs at three depths: the lens shows the prediction
// *moving* through the network, it does not echo the final answer everywhere.
TEST_F(ExplainerContextTest, LogitLensAtEarlierHiddenNodesDivergesFromTheRealOutputByHandDerivedAmounts) {
    LinearModule linear1(2, 2, &backend);
    linear1.set_weight({1.0f, 0.0f, 0.0f, 2.0f});
    linear1.set_bias({0.5f, -0.5f});
    ReluModule relu1(&backend);
    LinearModule linear2(2, 2, &backend);
    linear2.set_weight({2.0f, 0.0f, 0.0f, 1.0f});
    linear2.set_bias({-1.0f, 1.0f});
    ReluModule relu2(&backend);
    LinearModule head(2, 2, &backend);
    head.set_weight({1.0f, -1.0f, 0.5f, 2.0f});
    head.set_bias({0.25f, -0.25f});

    ExplainerContext ctx({&linear1, &relu1, &linear2, &relu2, &head});
    Tensor input(Shape({1, 2}), &backend, {1.0f, 2.0f});

    Tensor real_output = ctx.forward_pass(input);
    const std::vector<NodeId> nodes = ctx.graph().topological_order();
    ASSERT_EQ(nodes.size(), 6u);
    const NodeId input_node = nodes[0];
    const NodeId h1_node = nodes[2];  // post-relu1
    const NodeId h2_node = nodes[4];  // post-relu2, the head's actual input

    // The forward pass itself matches the hand derivation.
    EXPECT_FLOAT_EQ(ctx.activation(h1_node).data()[0], 1.5f);
    EXPECT_FLOAT_EQ(ctx.activation(h1_node).data()[1], 3.5f);
    EXPECT_FLOAT_EQ(ctx.activation(h2_node).data()[0], 2.0f);
    EXPECT_FLOAT_EQ(ctx.activation(h2_node).data()[1], 4.5f);
    EXPECT_FLOAT_EQ(real_output.data()[0], 4.5f);
    EXPECT_FLOAT_EQ(real_output.data()[1], 6.75f);

    Tensor lens_input = ctx.logit_lens(input_node);
    EXPECT_FLOAT_EQ(lens_input.data()[0], 2.25f);
    EXPECT_FLOAT_EQ(lens_input.data()[1], 2.75f);

    Tensor lens_h1 = ctx.logit_lens(h1_node);
    EXPECT_FLOAT_EQ(lens_h1.data()[0], 3.5f);
    EXPECT_FLOAT_EQ(lens_h1.data()[1], 5.25f);

    Tensor lens_h2 = ctx.logit_lens(h2_node);
    EXPECT_FLOAT_EQ(lens_h2.data()[0], 4.5f);
    EXPECT_FLOAT_EQ(lens_h2.data()[1], 6.75f);

    // ...and those three read-outs are actually distinct, which is the whole point.
    EXPECT_NE(lens_input.data()[0], lens_h1.data()[0]);
    EXPECT_NE(lens_h1.data()[0], lens_h2.data()[0]);
    EXPECT_NE(lens_input.data()[1], lens_h1.data()[1]);
    EXPECT_NE(lens_h1.data()[1], lens_h2.data()[1]);
}

// Adversarial (external boundary -- node_id is caller-supplied): the probed node's cached
// activation shape must match what the final module actually consumed. A shape-changing
// first layer makes the raw input node incompatible with the head, which is a real, expected
// failure mode of this technique, not an edge case to paper over. Checked up front, so the
// caller gets a message naming the shape mismatch rather than a confusing failure from deep
// inside the head's own gemm. Throw, not EXAI_ASSERT -- identical in Debug and Release.
TEST_F(ExplainerContextTest, LogitLensThrowsOnAShapeIncompatibleNode) {
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});
    ReluModule relu(&backend);
    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});

    Tensor reference_output = ctx.forward_pass(input);
    const NodeId input_node = ctx.graph().topological_order()[0];  // shape (1, 3); head wants (1, 4)

    EXPECT_THROW((void)ctx.logit_lens(input_node), std::invalid_argument);

    // Exception safety, mirroring forward_pass_with_patch's established pattern: the failed
    // call must leave the context fully usable -- a subsequent normal forward_pass() and a
    // subsequent valid logit_lens() both still produce correct results.
    Tensor after = ctx.forward_pass(input);
    EXPECT_EQ(ctx.graph().node_count(), 4u);  // input + 3 modules
    ASSERT_EQ(after.numel(), reference_output.numel());
    for (int64_t i = 0; i < reference_output.numel(); ++i) {
        EXPECT_FLOAT_EQ(after.data()[i], reference_output.data()[i]) << "index " << i;
    }

    Tensor lens = ctx.logit_lens(ctx.graph().topological_order()[2]);
    ASSERT_EQ(lens.numel(), after.numel());
    for (int64_t i = 0; i < after.numel(); ++i) {
        EXPECT_FLOAT_EQ(lens.data()[i], after.data()[i]) << "index " << i;
    }
}

// Adversarial (external boundary): a node id no forward pass ever produced, and the
// called-before-any-forward-pass case. Both are well-defined throws rather than a
// Debug-only EXAI_ASSERT on the activation cache (which would be UB in Release).
TEST_F(ExplainerContextTest, LogitLensThrowsOnAnUnknownNodeIdOrBeforeAnyForwardPass) {
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 0.0f, 0.0f, 1.0f});
    linear.set_bias({0.0f, 0.0f});

    ExplainerContext ctx({&linear});

    // No forward pass has run: nothing is cached at any node.
    EXPECT_THROW((void)ctx.logit_lens(0), std::invalid_argument);

    Tensor input(Shape({1, 2}), &backend, {1.0f, 2.0f});
    (void)ctx.forward_pass(input);

    // Valid ids here are 0 (input) and 1 (the single module's output).
    EXPECT_THROW((void)ctx.logit_lens(2), std::invalid_argument);
    EXPECT_THROW((void)ctx.logit_lens(99), std::invalid_argument);

    // Still usable afterwards.
    Tensor lens = ctx.logit_lens(0);
    EXPECT_FLOAT_EQ(lens.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(lens.data()[1], 2.0f);
}

}  // namespace
}  // namespace exai
