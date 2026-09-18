#include <gtest/gtest.h>

#include "exai/autograd.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/computation_graph.hpp"

namespace exai {
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

}  // namespace
}  // namespace exai
