#include <gtest/gtest.h>

#include <cmath>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/cross_entropy_loss.hpp"

namespace pulsatrix {
namespace {

class CrossEntropyLossTest : public ::testing::Test {
protected:
    CPUBackend backend;
    CrossEntropyLoss loss{&backend};
};

// logits=[1,2,0.5], target=1. Hand-computed via a numerically stable softmax
// (subtract max logit) -- see this mission's Recon for the exact Python cross-check.
TEST_F(CrossEntropyLossTest, ForwardComputesHandVerifiedValue) {
    Tensor logits(Shape({3}), &backend, {1.0f, 2.0f, 0.5f});

    float value = loss.forward(logits, /*target_class=*/1);

    EXPECT_NEAR(value, 0.46436878f, 1e-5f);
}

TEST_F(CrossEntropyLossTest, ForwardApproachesZeroWhenTargetLogitDominates) {
    Tensor logits(Shape({3}), &backend, {-100.0f, 100.0f, -100.0f});

    float value = loss.forward(logits, /*target_class=*/1);

    EXPECT_NEAR(value, 0.0f, 1e-4f);
}

TEST_F(CrossEntropyLossTest, BackwardComputesHandVerifiedGradient) {
    Tensor logits(Shape({3}), &backend, {1.0f, 2.0f, 0.5f});
    (void)loss.forward(logits, /*target_class=*/1);

    Tensor grad = loss.backward();

    EXPECT_NEAR(grad.data()[0], 0.23122390f, 1e-5f);
    EXPECT_NEAR(grad.data()[1], -0.37146828f, 1e-5f);
    EXPECT_NEAR(grad.data()[2], 0.14024438f, 1e-5f);
}

// Independent correctness check: central-difference numerical gradient of forward()
// w.r.t. each logit must match backward()'s analytical gradient -- the same rigor this
// project already holds every explainer to (finite-difference/closed-form cross-checks),
// applied to a loss for the first time. Catches sign errors and off-by-one softmax bugs
// a single hand-derived case could miss.
TEST_F(CrossEntropyLossTest, BackwardMatchesFiniteDifferenceGradient) {
    Tensor logits(Shape({4}), &backend, {0.3f, -1.2f, 2.1f, 0.7f});
    constexpr int64_t target_class = 2;
    constexpr float eps = 1e-3f;

    (void)loss.forward(logits, target_class);
    Tensor analytical_grad = loss.backward();

    for (int64_t i = 0; i < logits.numel(); ++i) {
        Tensor plus(logits);
        plus.data()[i] += eps;
        float loss_plus = loss.forward(plus, target_class);

        Tensor minus(logits);
        minus.data()[i] -= eps;
        float loss_minus = loss.forward(minus, target_class);

        float numerical_grad = (loss_plus - loss_minus) / (2.0f * eps);
        EXPECT_NEAR(analytical_grad.data()[i], numerical_grad, 1e-3f) << "mismatch at logit index " << i;
    }
}

using CrossEntropyLossDeathTest = CrossEntropyLossTest;

TEST_F(CrossEntropyLossDeathTest, ForwardAbortsOnOutOfRangeTargetClass) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor logits(Shape({3}), &backend, {1.0f, 2.0f, 0.5f});
    EXPECT_DEATH({ (void)loss.forward(logits, 3); }, "PULSATRIX_ASSERT failed");
}

}  // namespace
}  // namespace pulsatrix
