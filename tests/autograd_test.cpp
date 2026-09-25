#include <gtest/gtest.h>

#include "pulsatrix/autograd.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/computation_graph.hpp"

namespace pulsatrix {
namespace {

class AutogradTest : public ::testing::Test {
protected:
    CPUBackend backend;
    ComputationGraph graph;
    Autograd autograd;
};

// Smallest possible case: one edge, x -> y = neg(x). dz/dy = 1 (seed), dy/dx = -1.
TEST_F(AutogradTest, SingleOpBackwardComputesCorrectGradient) {
    NodeId x_id = graph.add_node(OpType::Elementwise, Shape({3}), "x");
    NodeId y_id = graph.add_node(OpType::Elementwise, Shape({3}), "neg", {x_id});

    autograd.register_backward(y_id, [this](const Tensor& grad_output) {
        Tensor grad_input(grad_output.shape(), &backend);
        backend.elementwise(ElementwiseOp::Neg, grad_output.data(), grad_input.data(), grad_output.numel());
        return grad_input;
    });

    Tensor seed(Shape({3}), &backend, {1.0f, 1.0f, 1.0f});
    autograd.backward(graph, y_id, seed);

    ASSERT_TRUE(autograd.has_gradient(x_id));
    const Tensor& grad_x = autograd.gradient(x_id);
    EXPECT_FLOAT_EQ(grad_x.data()[0], -1.0f);
    EXPECT_FLOAT_EQ(grad_x.data()[1], -1.0f);
    EXPECT_FLOAT_EQ(grad_x.data()[2], -1.0f);
}

TEST_F(AutogradTest, RootGradientIsTheSeedItself) {
    NodeId x_id = graph.add_node(OpType::Elementwise, Shape({2}), "x");
    NodeId y_id = graph.add_node(OpType::Elementwise, Shape({2}), "neg", {x_id});
    autograd.register_backward(y_id, [this](const Tensor& g) {
        Tensor out(g.shape(), &backend);
        backend.elementwise(ElementwiseOp::Neg, g.data(), out.data(), g.numel());
        return out;
    });

    Tensor seed(Shape({2}), &backend, {5.0f, 6.0f});
    autograd.backward(graph, y_id, seed);

    ASSERT_TRUE(autograd.has_gradient(y_id));
    EXPECT_FLOAT_EQ(autograd.gradient(y_id).data()[0], 5.0f);
    EXPECT_FLOAT_EQ(autograd.gradient(y_id).data()[1], 6.0f);
}

// Chain: x -> relu -> neg. z = -relu(x). dz/dy = -1 (y = relu(x)); dy/dx = 1 if x>0 else 0.
// Hand-derived: for x = [-1, 2, 3], relu'(x) = [0, 1, 1], so dz/dx = [-1*0, -1*1, -1*1] = [0, -1, -1].
TEST_F(AutogradTest, MultiNodeChainMatchesHandDerivedGradient) {
    Tensor x(Shape({3}), &backend, {-1.0f, 2.0f, 3.0f});

    NodeId x_id = graph.add_node(OpType::Elementwise, x.shape(), "x");

    Tensor relu_out(x.shape(), &backend);
    backend.elementwise(ElementwiseOp::Relu, x.data(), relu_out.data(), x.numel());
    NodeId y_id = graph.add_node(OpType::Activation, relu_out.shape(), "relu", {x_id});

    Tensor neg_out(relu_out.shape(), &backend);
    backend.elementwise(ElementwiseOp::Neg, relu_out.data(), neg_out.data(), relu_out.numel());
    NodeId z_id = graph.add_node(OpType::Elementwise, neg_out.shape(), "neg", {y_id});

    // dy/dx = relu'(x): 1 where x > 0, else 0. Captures a copy of x -- backward functions
    // need the forward-pass input values, which Phase 1's Module::forward() will save when
    // it registers a real op's backward function.
    autograd.register_backward(y_id, [this, x_copy = x](const Tensor& grad_output) {
        Tensor grad_input(grad_output.shape(), &backend);
        for (int64_t i = 0; i < grad_output.numel(); ++i) {
            grad_input[i] = grad_output.data()[i] * (x_copy.data()[i] > 0.0f ? 1.0f : 0.0f);
        }
        return grad_input;
    });
    // dz/dy = -1
    autograd.register_backward(z_id, [this](const Tensor& grad_output) {
        Tensor grad_input(grad_output.shape(), &backend);
        backend.elementwise(ElementwiseOp::Neg, grad_output.data(), grad_input.data(), grad_output.numel());
        return grad_input;
    });

    Tensor seed(Shape({3}), &backend, {1.0f, 1.0f, 1.0f});
    autograd.backward(graph, z_id, seed);

    ASSERT_TRUE(autograd.has_gradient(x_id));
    const Tensor& grad_x = autograd.gradient(x_id);
    EXPECT_FLOAT_EQ(grad_x.data()[0], 0.0f);   // x=-1: relu'=0
    EXPECT_FLOAT_EQ(grad_x.data()[1], -1.0f);  // x=2:  relu'=1, dz/dy=-1
    EXPECT_FLOAT_EQ(grad_x.data()[2], -1.0f);  // x=3:  relu'=1
}

// Fan-out: x feeds two independent children, y1=relu(x) and y2=neg(x). Seeding both in one
// backward() call should accumulate both contributions into x's gradient -- this is the
// real, concrete need that resolved the binary-op design decision deferred since Mission 0.
TEST_F(AutogradTest, GradientsFromMultipleChildrenAccumulateOnSharedParent) {
    NodeId x_id = graph.add_node(OpType::Elementwise, Shape({2}), "x");
    NodeId y1_id = graph.add_node(OpType::Activation, Shape({2}), "relu_branch", {x_id});
    NodeId y2_id = graph.add_node(OpType::Elementwise, Shape({2}), "neg_branch", {x_id});

    // dy1/dx = 1 (identity, to keep the arithmetic simple and the test focused on
    // accumulation rather than another relu derivation)
    autograd.register_backward(y1_id, [this](const Tensor& grad_output) { return Tensor(grad_output); });
    // dy2/dx = -1
    autograd.register_backward(y2_id, [this](const Tensor& grad_output) {
        Tensor grad_input(grad_output.shape(), &backend);
        backend.elementwise(ElementwiseOp::Neg, grad_output.data(), grad_input.data(), grad_output.numel());
        return grad_input;
    });

    std::vector<std::pair<NodeId, Tensor>> seeds;
    seeds.emplace_back(y1_id, Tensor(Shape({2}), &backend, {1.0f, 1.0f}));
    seeds.emplace_back(y2_id, Tensor(Shape({2}), &backend, {1.0f, 1.0f}));
    autograd.backward(graph, std::move(seeds));

    // grad_x = dy1/dx * seed_y1 + dy2/dx * seed_y2 = 1*1 + (-1)*1 = 0
    ASSERT_TRUE(autograd.has_gradient(x_id));
    EXPECT_FLOAT_EQ(autograd.gradient(x_id).data()[0], 0.0f);
    EXPECT_FLOAT_EQ(autograd.gradient(x_id).data()[1], 0.0f);
}

TEST_F(AutogradTest, SingleNodeGraphBackwardIsNoOp) {
    NodeId root_id = graph.add_node(OpType::Elementwise, Shape({2}), "root");
    // No backward function registered, no parents -- there is nothing to propagate to.

    Tensor seed(Shape({2}), &backend, {3.0f, 4.0f});
    EXPECT_NO_THROW(autograd.backward(graph, root_id, seed));

    ASSERT_TRUE(autograd.has_gradient(root_id));
    EXPECT_FLOAT_EQ(autograd.gradient(root_id).data()[0], 3.0f);
    EXPECT_FLOAT_EQ(autograd.gradient(root_id).data()[1], 4.0f);
}

TEST_F(AutogradTest, NodeDisconnectedFromSeedNeverGetsAGradient) {
    // Two independent, unconnected single-node graphs sharing one ComputationGraph.
    NodeId reachable_id = graph.add_node(OpType::Elementwise, Shape({1}), "reachable");
    NodeId unreachable_id = graph.add_node(OpType::Elementwise, Shape({1}), "unreachable");

    Tensor seed(Shape({1}), &backend, {1.0f});
    autograd.backward(graph, reachable_id, seed);

    EXPECT_TRUE(autograd.has_gradient(reachable_id));
    EXPECT_FALSE(autograd.has_gradient(unreachable_id));
}

}  // namespace
}  // namespace pulsatrix
