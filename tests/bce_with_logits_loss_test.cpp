#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/bce_with_logits_loss.hpp"
#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

class BCEWithLogitsLossTest : public ::testing::Test {
protected:
    CPUBackend backend;
    BCEWithLogitsLoss loss{&backend};
};

// A zero logit is sigmoid(0) = 0.5, i.e. maximum uncertainty, whose BCE is -log(0.5) = ln 2
// for either label. The one value the combined form must get exactly right regardless of y.
TEST_F(BCEWithLogitsLossTest, ForwardIsLog2AtZeroLogitForEitherLabel) {
    Tensor logits(Shape({1, 2}), &backend, {0.0f, 0.0f});
    Tensor target(Shape({1, 2}), &backend, {1.0f, 0.0f});

    EXPECT_NEAR(loss.forward(logits, target), std::log(2.0f), 1e-6f);
}

TEST_F(BCEWithLogitsLossTest, ForwardComputesHandVerifiedValue) {
    // x = [2, -1], y = [1, 0]. Per-element max(x,0) - x*y + log(1 + exp(-|x|)):
    //   i=0: 2 - 2 + log(1 + exp(-2)) = log(1.13533528) = 0.12692801
    //   i=1: 0 - 0 + log(1 + exp(-1)) = log(1.36787944) = 0.31326169
    // mean = (0.12692801 + 0.31326169) / 2 = 0.22009485
    Tensor logits(Shape({1, 2}), &backend, {2.0f, -1.0f});
    Tensor target(Shape({1, 2}), &backend, {1.0f, 0.0f});

    EXPECT_NEAR(loss.forward(logits, target), 0.22009485f, 1e-6f);
}

// The whole reason this class fuses sigmoid into the loss: the naive
// -y*log(sigmoid(x)) - (1-y)*log(1-sigmoid(x)) form saturates sigmoid() to exactly 0.0f/1.0f
// in float at |x| ~ 90 and then returns inf from log(0). The stable form must stay finite
// there -- and must stay numerically *accurate*, since for a confidently-wrong prediction the
// loss tends to |x| itself (max(x,0) - x*y dominates, log1p(exp(-|x|)) underflows to 0).
TEST_F(BCEWithLogitsLossTest, ForwardIsFiniteAndAccurateForExtremeLogits) {
    Tensor logits(Shape({1, 4}), &backend, {1000.0f, -1000.0f, 89.0f, -89.0f});
    // Deliberately the *wrong* label in each case, the branch that would be log(0) naively.
    Tensor target(Shape({1, 4}), &backend, {0.0f, 1.0f, 0.0f, 1.0f});

    const float value = loss.forward(logits, target);

    ASSERT_TRUE(std::isfinite(value));
    // Each term collapses to |x|; mean = (1000 + 1000 + 89 + 89) / 4 = 544.5.
    EXPECT_NEAR(value, 544.5f, 1e-2f);
}

// A confidently-correct prediction is the mirrored saturation case: the loss must go to ~0
// rather than to a NaN from 0 * inf.
TEST_F(BCEWithLogitsLossTest, ForwardIsNearZeroForConfidentlyCorrectLogits) {
    Tensor logits(Shape({1, 2}), &backend, {1000.0f, -1000.0f});
    Tensor target(Shape({1, 2}), &backend, {1.0f, 0.0f});

    const float value = loss.forward(logits, target);

    EXPECT_TRUE(std::isfinite(value));
    EXPECT_NEAR(value, 0.0f, 1e-6f);
}

// Pins the mean reduction over *all* N*k elements (not a per-example sum, and not a batch
// sum): duplicating an example across the batch must leave the loss unchanged.
TEST_F(BCEWithLogitsLossTest, ForwardAveragesOverAllElements) {
    Tensor logits_one(Shape({1, 2}), &backend, {0.7f, -1.3f});
    Tensor target_one(Shape({1, 2}), &backend, {1.0f, 0.0f});
    Tensor logits_two(Shape({2, 2}), &backend, {0.7f, -1.3f, 0.7f, -1.3f});
    Tensor target_two(Shape({2, 2}), &backend, {1.0f, 0.0f, 1.0f, 0.0f});

    BCEWithLogitsLoss loss_one(&backend);
    BCEWithLogitsLoss loss_two(&backend);

    EXPECT_NEAR(loss_one.forward(logits_one, target_one), loss_two.forward(logits_two, target_two), 1e-6f);
    // ...and that shared value is genuinely non-zero, so the comparison above is not
    // satisfiable by a loss that always returns 0.
    EXPECT_GT(loss_one.forward(logits_one, target_one), 1e-2f);
}

TEST_F(BCEWithLogitsLossTest, ForwardIsNonNegative) {
    Tensor logits(Shape({2, 3}), &backend, {0.8f, -0.6f, 1.2f, 0.5f, -0.9f, 1.1f});
    Tensor target(Shape({2, 3}), &backend, {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f});

    EXPECT_GT(loss.forward(logits, target), 0.0f);
}

TEST_F(BCEWithLogitsLossTest, ForwardThrowsOnMismatchedShapes) {
    Tensor logits(Shape({2, 3}), &backend);
    Tensor target(Shape({2, 4}), &backend);

    EXPECT_THROW({ (void)loss.forward(logits, target); }, std::invalid_argument);
}

TEST_F(BCEWithLogitsLossTest, BackwardThrowsBeforeForward) {
    EXPECT_THROW({ (void)loss.backward(); }, std::logic_error);
}

TEST_F(BCEWithLogitsLossTest, BackwardComputesHandVerifiedGradient) {
    // x = [0, 2], y = [1, 1], numel = 2. grad = (sigmoid(x) - y) / 2.
    //   i=0: (0.5 - 1) / 2 = -0.25
    //   i=1: (0.88079708 - 1) / 2 = -0.05960146
    Tensor logits(Shape({1, 2}), &backend, {0.0f, 2.0f});
    Tensor target(Shape({1, 2}), &backend, {1.0f, 1.0f});
    (void)loss.forward(logits, target);

    Tensor grad = loss.backward();

    ASSERT_EQ(grad.numel(), 2);
    EXPECT_NEAR(grad.data()[0], -0.25f, 1e-6f);
    EXPECT_NEAR(grad.data()[1], -0.05960146f, 1e-6f);
}

// sigmoid(x) - y vanishes exactly when the prediction matches the label, which for y in {0,1}
// only happens in the saturated limit -- so the gradient must *underflow to zero*, not to a
// NaN, for a confidently-correct extreme logit.
TEST_F(BCEWithLogitsLossTest, BackwardIsFiniteForExtremeLogits) {
    Tensor logits(Shape({1, 4}), &backend, {1000.0f, -1000.0f, 1000.0f, -1000.0f});
    Tensor target(Shape({1, 4}), &backend, {1.0f, 0.0f, 0.0f, 1.0f});
    (void)loss.forward(logits, target);

    Tensor grad = loss.backward();

    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(grad.data()[i])) << "index " << i;
    }
    // Correctly classified: gradient vanishes. Wrongly classified: saturates at +-1/numel.
    EXPECT_NEAR(grad.data()[0], 0.0f, 1e-9f);
    EXPECT_NEAR(grad.data()[1], 0.0f, 1e-9f);
    EXPECT_NEAR(grad.data()[2], 0.25f, 1e-6f);
    EXPECT_NEAR(grad.data()[3], -0.25f, 1e-6f);
}

// ---------------------------------------------------------------------------------------
// Central finite-difference gradient check.
//
// The loss is itself the scalar objective, so no artificial grad seed is needed -- the
// numeric derivative is taken directly on forward()'s return value. The fixture is
// deliberately non-uniform (N != k, logits spanning both signs over ~0.4-2.5 so sigmoid()
// ranges ~0.08-0.92, targets mixing 0s and 1s in both saturating directions) rather than
// small-and-uniform: RWKVModule's mission found the hard way that a uniform small-scale
// fixture can make a check pass *vacuously* against the tolerance floor. Tolerance is
// relative (1e-3 + 2e-2*|numeric|) for the same established reason. Non-vacuity is
// additionally asserted inline (every numeric gradient is well clear of the floor) and was
// verified by mutation probe (see the mission's close-out evidence).
// ---------------------------------------------------------------------------------------
namespace {

constexpr int64_t kN = 2;
constexpr int64_t kK = 3;

const std::vector<float>& fd_logit_values() {
    static const std::vector<float> v{0.80f, -2.50f, 1.20f, -0.40f, 2.10f, -1.30f};
    return v;
}
const std::vector<float>& fd_target_values() {
    static const std::vector<float> v{1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    return v;
}

float scalar_loss(CPUBackend& backend, const std::vector<float>& logit_values) {
    BCEWithLogitsLoss l(&backend);
    Tensor logits(Shape({kN, kK}), &backend, logit_values);
    Tensor target(Shape({kN, kK}), &backend, fd_target_values());
    return l.forward(logits, target);
}

float relative_tolerance(float numeric) {
    return 1e-3f + 2e-2f * std::fabs(numeric);
}

}  // namespace

TEST_F(BCEWithLogitsLossTest, BackwardMatchesFiniteDifference) {
    const float h = 1e-3f;
    Tensor logits(Shape({kN, kK}), &backend, fd_logit_values());
    Tensor target(Shape({kN, kK}), &backend, fd_target_values());
    (void)loss.forward(logits, target);
    Tensor grad = loss.backward();

    ASSERT_EQ(grad.numel(), kN * kK);
    for (size_t idx = 0; idx < fd_logit_values().size(); ++idx) {
        std::vector<float> plus = fd_logit_values();
        std::vector<float> minus = fd_logit_values();
        plus[idx] += h;
        minus[idx] -= h;
        const float numeric = (scalar_loss(backend, plus) - scalar_loss(backend, minus)) / (2.0f * h);
        EXPECT_NEAR(grad.data()[static_cast<int64_t>(idx)], numeric, relative_tolerance(numeric))
            << "logit index " << idx;
        // Non-vacuity: every numeric gradient is well clear of the 1e-3 tolerance floor, so
        // the check above cannot be satisfied by a backward() that returns all zeros.
        EXPECT_GT(std::fabs(numeric), 1e-2f) << "logit index " << idx;
    }
}

using BCEWithLogitsLossDeathTest = BCEWithLogitsLossTest;

// forward() dereferences Tensor::data() directly in a raw host loop -- undefined behavior on
// a CUDA-backed Tensor, so it is PULSATRIX_ASSERT-guarded (mission_host_loop_guards.md). No real
// GPU needed: see LinearModuleDeathTest for the mislabeled-Tensor testing pattern this reuses.
//
// backward() is NOT independently guarded: it has no parameters, it only ever reads
// last_logits_/last_target_, and those are only ever populated by forward() -- which already
// rejects a non-Cpu tensor before caching it. There is no reachable call sequence that gets a
// non-Cpu tensor into backward()'s cached state, so a second guard there would be untestable
// dead code, not a real safety net. Identical to MSELoss::backward()'s precedent.
TEST_F(BCEWithLogitsLossDeathTest, ForwardAbortsOnNonCpuLogits) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor logits(Shape({1, 2}), &backend, {1.0f, 2.0f}, DeviceType::Cuda);
    Tensor target(Shape({1, 2}), &backend, {1.0f, 0.0f});
    EXPECT_DEATH({ (void)loss.forward(logits, target); }, "PULSATRIX_ASSERT failed");
}

TEST_F(BCEWithLogitsLossDeathTest, ForwardAbortsOnNonCpuTarget) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor logits(Shape({1, 2}), &backend, {1.0f, 2.0f});
    Tensor target(Shape({1, 2}), &backend, {1.0f, 0.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)loss.forward(logits, target); }, "PULSATRIX_ASSERT failed");
}

}  // namespace
}  // namespace pulsatrix
