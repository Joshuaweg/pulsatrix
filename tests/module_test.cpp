#include <gtest/gtest.h>

#include <stdexcept>

#include "exai/autograd.hpp"
#include "exai/computation_graph.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/module.hpp"
#include "exai/op_type.hpp"

namespace exai {
namespace {

// Minimal concrete subclass -- Module is abstract. This deliberately does NOT itself
// validate input, so passing tests prove the NVI wrapper's precondition check runs
// regardless of what forward_impl() does, not that forward_impl() happens to check too.
class TestDoubleModule : public Module {
public:
    Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) override {
        return Tensor(relevance_out);  // pass-through, not a real rule -- this is a test double
    }

    // Identity, matching forward_impl()'s own identity behavior -- not a real gradient rule,
    // this is a test double. Phase 2 Mission 0: backward() promoted to virtual on Module so
    // graph-wiring code can call it polymorphically without knowing the concrete subclass.
    Tensor backward(const Tensor& grad_output) override { return Tensor(grad_output); }

    // Arbitrary but fixed choice for a test double -- Elementwise is as reasonable as any
    // other tag for an identity op. Phase 2 Mission 0: op_type() lets ComputationGraph
    // node-tagging work generically through a Module* without a per-subclass switch.
    [[nodiscard]] OpType op_type() const override { return OpType::Elementwise; }

protected:
    Tensor forward_impl(const Tensor& input) override {
        return Tensor(input);  // identity, no validation of its own
    }
};

class ModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
    TestDoubleModule module;
};

TEST_F(ModuleTest, ForwardDelegatesToForwardImpl) {
    Tensor input(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor output = module.forward(input);

    EXPECT_FLOAT_EQ(output.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(output.data()[2], 3.0f);
}

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 2):
// escalated from EXAI_ASSERT (was a death test) to a real throw -- the single most
// external-facing check in the whole system (every Module::forward() call, including
// from Phase 5's Python bindings, passes through this NVI wrapper first).
TEST_F(ModuleTest, ForwardThrowsOnEmptyInputEvenThoughForwardImplDoesNotCheck) {
    Tensor empty(Shape({0}), &backend);
    EXPECT_THROW({ (void)module.forward(empty); }, std::invalid_argument);
}

TEST_F(ModuleTest, PropagateRelevanceIsCallableThroughBasePointer) {
    Module& base = module;
    Tensor relevance_out(Shape({2}), &backend, {0.5f, 0.5f});
    LRPRuleConfig config;

    Tensor relevance_in = base.propagate_relevance(relevance_out, config);

    EXPECT_FLOAT_EQ(relevance_in.data()[0], 0.5f);
    EXPECT_FLOAT_EQ(relevance_in.data()[1], 0.5f);
}

TEST_F(ModuleTest, BackwardIsCallableThroughBasePointer) {
    Module& base = module;
    Tensor grad_output(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});

    Tensor grad_input = base.backward(grad_output);

    EXPECT_FLOAT_EQ(grad_input.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(grad_input.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(grad_input.data()[2], 3.0f);
}

TEST_F(ModuleTest, OpTypeIsCallableThroughBasePointer) {
    const Module& base = module;
    EXPECT_EQ(base.op_type(), OpType::Elementwise);
}

// Phase 2 Mission 0's actual deliverable: an opt-in path that builds a real
// ComputationGraph node (correctly op-type-tagged, correctly parented) and registers a
// real Autograd backward function, reusing the module's own backward() rather than
// reimplementing gradient math. Every existing forward()/backward() call site is
// unaffected -- this is a wholly separate, additive entry point.
TEST_F(ModuleTest, ForwardTracedAddsOneNodeWithCorrectOpTypeAndParent) {
    ComputationGraph graph;
    Autograd autograd;
    NodeId input_node = graph.add_node(OpType::Elementwise, Shape({3}), "input");

    Tensor input(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    auto [output, output_node] = module.forward_traced(input, input_node, graph, autograd);

    EXPECT_EQ(graph.node_count(), 2u);
    EXPECT_EQ(graph.node(output_node).op_type(), OpType::Elementwise);  // TestDoubleModule's op_type()
    EXPECT_EQ(graph.node(output_node).shape(), output.shape());
    ASSERT_EQ(graph.node(output_node).parents().size(), 1u);
    EXPECT_EQ(graph.node(output_node).parents()[0]->id(), input_node);
}

TEST_F(ModuleTest, ForwardTracedRegisteredBackwardMatchesDirectBackward) {
    ComputationGraph graph;
    Autograd autograd;
    NodeId input_node = graph.add_node(OpType::Elementwise, Shape({3}), "input");

    Tensor input(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    auto [output, output_node] = module.forward_traced(input, input_node, graph, autograd);
    (void)output;

    Tensor grad_output(Shape({3}), &backend, {10.0f, 20.0f, 30.0f});
    autograd.backward(graph, output_node, grad_output);
    Tensor direct_grad = module.backward(grad_output);

    ASSERT_TRUE(autograd.has_gradient(input_node));
    const Tensor& traced_grad = autograd.gradient(input_node);
    for (int64_t i = 0; i < direct_grad.numel(); ++i) {
        EXPECT_FLOAT_EQ(traced_grad.data()[i], direct_grad.data()[i]);
    }
}

TEST_F(ModuleTest, DefaultParametersIsEmpty) {
    // TestDoubleModule doesn't override parameters() -- the base default (no parameters)
    // is what a parameterless module type like ReluModule relies on too.
    EXPECT_TRUE(module.parameters().empty());
}

// Phase 6 Mission 5 (DropoutModule): plain, non-virtual training/eval toggle added to the
// base Module class -- defaults to training, matching every mainstream framework's Module.
TEST_F(ModuleTest, DefaultsToTrainingMode) {
    EXPECT_TRUE(module.is_training());
}

TEST_F(ModuleTest, SetTrainingTogglesIsTraining) {
    module.set_training(false);
    EXPECT_FALSE(module.is_training());
    module.set_training(true);
    EXPECT_TRUE(module.is_training());
}

}  // namespace
}  // namespace exai
