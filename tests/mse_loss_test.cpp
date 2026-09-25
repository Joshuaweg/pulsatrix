#include <gtest/gtest.h>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/mse_loss.hpp"

namespace pulsatrix {
namespace {

class MSELossTest : public ::testing::Test {
protected:
    CPUBackend backend;
    MSELoss loss{&backend};
};

TEST_F(MSELossTest, ForwardComputesHandVerifiedValue) {
    // pred=[1,2,3], target=[1,0,3]. diff=[0,2,0]. squared=[0,4,0]. mean = 4/3.
    Tensor pred(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor target(Shape({3}), &backend, {1.0f, 0.0f, 3.0f});

    float value = loss.forward(pred, target);

    EXPECT_NEAR(value, 4.0f / 3.0f, 1e-5f);
}

TEST_F(MSELossTest, ForwardIsZeroWhenPredictionMatchesTarget) {
    Tensor pred(Shape({2}), &backend, {5.0f, -3.0f});
    Tensor target(Shape({2}), &backend, {5.0f, -3.0f});

    EXPECT_FLOAT_EQ(loss.forward(pred, target), 0.0f);
}

TEST_F(MSELossTest, BackwardComputesHandVerifiedGradient) {
    // grad = (2/n) * (pred - target) = (2/3) * [0, 2, 0] = [0, 4/3, 0]
    Tensor pred(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor target(Shape({3}), &backend, {1.0f, 0.0f, 3.0f});
    (void)loss.forward(pred, target);

    Tensor grad = loss.backward();

    EXPECT_NEAR(grad.data()[0], 0.0f, 1e-5f);
    EXPECT_NEAR(grad.data()[1], 4.0f / 3.0f, 1e-5f);
    EXPECT_NEAR(grad.data()[2], 0.0f, 1e-5f);
}

using MSELossDeathTest = MSELossTest;

// forward() dereferences Tensor::data() directly in a raw host loop -- undefined behavior
// on a CUDA-backed Tensor. Phase 1.5 Mission 2 (mission_host_loop_guards.md) guards it with
// PULSATRIX_ASSERT. No real GPU needed: see LinearModuleDeathTest for the mislabeled-Tensor
// testing pattern this reuses.
//
// backward() is NOT independently guarded: it has no parameters, it only ever reads
// last_prediction_/last_target_, and those are only ever populated by forward() -- which
// already rejects a non-Cpu tensor before caching it. There is no reachable call sequence
// that gets a non-Cpu tensor into backward()'s cached state, so a second guard there would
// be untestable dead code, not a real safety net.
TEST_F(MSELossDeathTest, ForwardAbortsOnNonCpuPrediction) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor pred(Shape({3}), &backend, {1.0f, 2.0f, 3.0f}, DeviceType::Cuda);
    Tensor target(Shape({3}), &backend, {1.0f, 0.0f, 3.0f});
    EXPECT_DEATH({ (void)loss.forward(pred, target); }, "PULSATRIX_ASSERT failed");
}

TEST_F(MSELossDeathTest, ForwardAbortsOnNonCpuTarget) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor pred(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor target(Shape({3}), &backend, {1.0f, 0.0f, 3.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)loss.forward(pred, target); }, "PULSATRIX_ASSERT failed");
}

}  // namespace
}  // namespace pulsatrix
