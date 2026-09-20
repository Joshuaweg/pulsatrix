#include <gtest/gtest.h>

#include "exai/cpu_backend.hpp"
#include "exai/flatten_module.hpp"
#include "exai/lrp_rule_config.hpp"

// FlattenModule (Phase 2 Mission 3) is a reshape-only Module -- needed to chain
// Conv2DModule's batched (N,C,H,W) output into a LinearModule's batched (N, in_features)
// input, the first real classifier-head network this codebase has built. No dedicated
// Reshape OpType exists (op_type.hpp's closed set) -- Elementwise is the closest fit
// (identity over the same values, just re-viewed), documented as a deliberate choice, not
// an ideal one. Migrated to flatten only non-batch dims by
// campaign_exai_dl_library_batch_dimension_support -- a genuine behavior change (the old
// semantics flattened the batch dim away too, which is wrong once N carries real
// per-example meaning).
namespace exai {
namespace {

class FlattenModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
    FlattenModule flatten{&backend};
};

// (N=2, 2, 2) -> (N=2, 4). Row-major flatten of the trailing dims leaves the underlying
// buffer bytes identical to before the migration -- only the Shape metadata changes, a
// real confirmation this migration is a reinterpretation, not a data reshuffle.
TEST_F(FlattenModuleTest, ForwardFlattensNonBatchDimsWithSameValuesInOrder) {
    Tensor input(Shape({2, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    Tensor output = flatten.forward(input);

    EXPECT_EQ(output.shape(), Shape({2, 4}));
    for (int64_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(output.data()[i], static_cast<float>(i + 1));
    }
}

// The real acceptance criterion for the batch migration -- rank-4 (N,C,H,W) input, the
// actual shape Conv2DModule's output now has, proving the batch dim survives distinctly
// from the flattened feature dims rather than being folded in with them.
TEST_F(FlattenModuleTest, ForwardFlattensChannelsAndSpatialDimsButNotBatch) {
    // N=2, C=2, H=1, W=2 -> (2, 4).
    Tensor input(Shape({2, 2, 1, 2}), &backend,
                 {1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f});
    Tensor output = flatten.forward(input);

    EXPECT_EQ(output.shape(), Shape({2, 4}));
    EXPECT_FLOAT_EQ(output.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(output.data()[3], 4.0f);
    EXPECT_FLOAT_EQ(output.data()[4], 10.0f);
    EXPECT_FLOAT_EQ(output.data()[7], 40.0f);
}

TEST_F(FlattenModuleTest, BackwardReshapesGradientBackToOriginalShape) {
    Tensor input(Shape({2, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    (void)flatten.forward(input);

    Tensor grad_output(Shape({2, 4}), &backend, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f});
    Tensor grad_input = flatten.backward(grad_output);

    EXPECT_EQ(grad_input.shape(), Shape({2, 2, 2}));
    for (int64_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(grad_input.data()[i], static_cast<float>((i + 1) * 10));
    }
}

TEST_F(FlattenModuleTest, PropagateRelevanceReshapesBackToOriginalShapeUnchanged) {
    Tensor input(Shape({2, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    (void)flatten.forward(input);

    Tensor relevance_out(Shape({2, 4}), &backend, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});
    LRPRuleConfig config;
    Tensor relevance_in = flatten.propagate_relevance(relevance_out, config);

    EXPECT_EQ(relevance_in.shape(), Shape({2, 2, 2}));
    for (int64_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(relevance_in.data()[i], relevance_out.data()[i]);
    }
}

}  // namespace
}  // namespace exai
