#include <gtest/gtest.h>

#include "exai/cpu_backend.hpp"
#include "exai/flatten_module.hpp"
#include "exai/lrp_rule_config.hpp"

// FlattenModule (Phase 2 Mission 3) is a reshape-only Module -- needed to chain
// Conv2DModule's rank-3 output into a LinearModule's rank-1 input, the first real
// classifier-head network this codebase has built. No dedicated Reshape OpType exists
// (op_type.hpp's closed set) -- Elementwise is the closest fit (identity over the same
// values, just re-viewed), documented as a deliberate choice, not an ideal one.
namespace exai {
namespace {

class FlattenModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
    FlattenModule flatten{&backend};
};

TEST_F(FlattenModuleTest, ForwardFlattensToRankOneWithSameValuesInOrder) {
    Tensor input(Shape({2, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    Tensor output = flatten.forward(input);

    EXPECT_EQ(output.shape(), Shape({8}));
    for (int64_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(output.data()[i], static_cast<float>(i + 1));
    }
}

TEST_F(FlattenModuleTest, BackwardReshapesGradientBackToOriginalShape) {
    Tensor input(Shape({2, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    (void)flatten.forward(input);

    Tensor grad_output(Shape({8}), &backend, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f});
    Tensor grad_input = flatten.backward(grad_output);

    EXPECT_EQ(grad_input.shape(), Shape({2, 2, 2}));
    for (int64_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(grad_input.data()[i], static_cast<float>((i + 1) * 10));
    }
}

TEST_F(FlattenModuleTest, PropagateRelevanceReshapesBackToOriginalShapeUnchanged) {
    Tensor input(Shape({2, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    (void)flatten.forward(input);

    Tensor relevance_out(Shape({8}), &backend, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});
    LRPRuleConfig config;
    Tensor relevance_in = flatten.propagate_relevance(relevance_out, config);

    EXPECT_EQ(relevance_in.shape(), Shape({2, 2, 2}));
    for (int64_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(relevance_in.data()[i], relevance_out.data()[i]);
    }
}

}  // namespace
}  // namespace exai
