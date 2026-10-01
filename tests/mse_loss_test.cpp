#include <gtest/gtest.h>

#include <stdexcept>

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

// GPU-native-kernels Mission 1: forward is device-generic, so a mixed-device pair is now a
// reachable caller error rather than a host-loop guard.
TEST_F(MSELossTest, ForwardThrowsOnMixedDevices) {
    Tensor prediction(Shape({2}), &backend, {1.0f, 2.0f});
    Tensor target(Shape({2}), &backend, {1.0f, 2.0f}, DeviceType::Cuda);
    EXPECT_THROW((void)loss.forward(prediction, target), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
