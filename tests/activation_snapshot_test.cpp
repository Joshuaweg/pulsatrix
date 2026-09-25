#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pulsatrix/activation_snapshot.hpp"
#include "pulsatrix/cpu_backend.hpp"

// ActivationSnapshot (campaign_exai_dl_library_mechanistic_interpretability, Phase 1
// Mission 1) is the self-contained, enumerable generalization of ExplainerContext's
// single-node activation cache: a copyable value type holding a deep copy of the cached
// activations, the node ids in topological order at capture time, and the per-node
// op_type/label metadata -- with no reference back to any ComputationGraph or
// ExplainerContext, so two snapshots from two different forward passes can be alive
// (and correct) at the same time. This file covers the value type itself; the capture
// path lives in explainer_context_test.cpp (Objective 2).
namespace pulsatrix {
namespace {

class ActivationSnapshotTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // A hand-built two-node snapshot standing in for what ExplainerContext::
    // activation_snapshot() produces -- construction is ExplainerContext's job in
    // production, so this fixture is the only place snapshots are assembled by hand.
    ActivationSnapshot make_snapshot() {
        std::unordered_map<NodeId, Tensor> activations;
        activations.emplace(0u, Tensor(Shape({2}), &backend, {1.0f, 2.0f}));
        activations.emplace(1u, Tensor(Shape({2}), &backend, {3.0f, 4.0f}));

        std::unordered_map<NodeId, ActivationSnapshot::NodeMetadata> metadata;
        metadata.emplace(0u, ActivationSnapshot::NodeMetadata{OpType::Elementwise, std::string("input")});
        metadata.emplace(1u, ActivationSnapshot::NodeMetadata{OpType::Linear, std::nullopt});

        return ActivationSnapshot(std::move(activations), std::vector<NodeId>{0u, 1u}, std::move(metadata));
    }
};

TEST_F(ActivationSnapshotTest, ActivationReturnsConstructedValuePerNode) {
    ActivationSnapshot snapshot = make_snapshot();

    const Tensor& first = snapshot.activation(0u);
    EXPECT_FLOAT_EQ(first.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(first.data()[1], 2.0f);

    const Tensor& second = snapshot.activation(1u);
    EXPECT_FLOAT_EQ(second.data()[0], 3.0f);
    EXPECT_FLOAT_EQ(second.data()[1], 4.0f);
}

TEST_F(ActivationSnapshotTest, NodeIdsReturnsConstructedIdsInConstructedOrder) {
    ActivationSnapshot snapshot = make_snapshot();

    const std::vector<NodeId>& ids = snapshot.node_ids();
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 0u);
    EXPECT_EQ(ids[1], 1u);
}

TEST_F(ActivationSnapshotTest, OpTypeReturnsConstructedMetadataPerNode) {
    ActivationSnapshot snapshot = make_snapshot();

    EXPECT_EQ(snapshot.op_type(0u), OpType::Elementwise);
    EXPECT_EQ(snapshot.op_type(1u), OpType::Linear);
}

TEST_F(ActivationSnapshotTest, LabelReturnsConstructedLabelOrNulloptWhenUnlabeled) {
    ActivationSnapshot snapshot = make_snapshot();

    ASSERT_TRUE(snapshot.label(0u).has_value());
    EXPECT_EQ(snapshot.label(0u).value(), "input");
    EXPECT_FALSE(snapshot.label(1u).has_value());
}

// The point of the whole type: a snapshot is a value, independent of whatever produced
// it. Copying one and destroying the original must leave the copy fully intact -- this
// is what lets Phase 4 hold a "clean" and a "corrupted" run's activations at once.
TEST_F(ActivationSnapshotTest, CopyOutlivesAndIsIndependentOfItsSource) {
    std::optional<ActivationSnapshot> copy;
    {
        ActivationSnapshot original = make_snapshot();
        copy = original;  // copy-construct into the outer scope
    }

    ASSERT_TRUE(copy.has_value());
    EXPECT_FLOAT_EQ(copy->activation(0u).data()[0], 1.0f);
    EXPECT_FLOAT_EQ(copy->activation(1u).data()[1], 4.0f);
    EXPECT_EQ(copy->node_ids().size(), 2u);
    EXPECT_EQ(copy->op_type(1u), OpType::Linear);
}

// Boundary (degenerate empty collections, adversarial policy item 3/4): an empty
// snapshot is a well-defined value, not an error -- ExplainerContext::
// activation_snapshot() produces exactly this before any forward_pass() has run.
TEST_F(ActivationSnapshotTest, EmptySnapshotIsWellFormedAndHasNoNodeIds) {
    ActivationSnapshot snapshot({}, {}, {});

    EXPECT_TRUE(snapshot.node_ids().empty());
}

// Adversarial/boundary, unknown-NodeId lookup. Classified internal-only invariant ->
// PULSATRIX_ASSERT (cpp_tdd/context_tdd_adversarial_boundary_testing.md's classification
// table): ActivationSnapshot is constructed only by ExplainerContext, and every id a
// caller can legitimately pass comes from node_ids() -- the same already-validated
// internal-call-chain origin as ExplainerContext::activation(), whose not-found
// behavior this deliberately mirrors (mission Exit Gate). Not pybind11-bound; if it
// ever is, these three re-classify to throw.
class ActivationSnapshotDeathTest : public ActivationSnapshotTest {};

TEST_F(ActivationSnapshotDeathTest, ActivationAbortsOnUnknownNodeId) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#else
    ActivationSnapshot snapshot = make_snapshot();
    ASSERT_DEATH((void)snapshot.activation(99u), "PULSATRIX_ASSERT failed");
#endif
}

TEST_F(ActivationSnapshotDeathTest, OpTypeAbortsOnUnknownNodeId) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#else
    ActivationSnapshot snapshot = make_snapshot();
    ASSERT_DEATH((void)snapshot.op_type(99u), "PULSATRIX_ASSERT failed");
#endif
}

TEST_F(ActivationSnapshotDeathTest, LabelAbortsOnUnknownNodeId) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#else
    ActivationSnapshot snapshot = make_snapshot();
    ASSERT_DEATH((void)snapshot.label(99u), "PULSATRIX_ASSERT failed");
#endif
}

}  // namespace
}  // namespace pulsatrix
