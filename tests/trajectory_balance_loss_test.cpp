/** @file trajectory_balance_loss_test.cpp
 *  @brief TrajectoryBalanceLoss forward/gradient-accessor correctness.
 *
 * Hand-derived reference: sum_log_pf=1.0, sum_log_pb=0.5, log_reward=2.0, log_z=0.3.
 * Delta = log_z + sum_log_pf - log_reward - sum_log_pb = 0.3 + 1.0 - 2.0 - 0.5 = -1.2.
 * Loss = Delta^2 = 1.44. grad_log_z = 2*Delta = -2.4. grad_weight_for_log_pf = -2*Delta = 2.4.
 */
#include <gtest/gtest.h>

#include <stdexcept>

#include "pulsatrix/trajectory_balance_loss.hpp"

namespace pulsatrix {
namespace {

TEST(TrajectoryBalanceLossTest, ForwardMatchesHandDerivedValue) {
    TrajectoryBalanceLoss loss;
    const float value = loss.forward(1.0f, 0.5f, 2.0f, 0.3f);
    EXPECT_NEAR(value, 1.44f, 1e-5f);
}

TEST(TrajectoryBalanceLossTest, DeltaMatchesHandDerivedValue) {
    TrajectoryBalanceLoss loss;
    (void)loss.forward(1.0f, 0.5f, 2.0f, 0.3f);
    EXPECT_NEAR(loss.delta(), -1.2f, 1e-5f);
}

TEST(TrajectoryBalanceLossTest, GradLogZMatchesHandDerivedValue) {
    TrajectoryBalanceLoss loss;
    (void)loss.forward(1.0f, 0.5f, 2.0f, 0.3f);
    EXPECT_NEAR(loss.grad_log_z(), -2.4f, 1e-5f);
}

TEST(TrajectoryBalanceLossTest, GradWeightForLogPfMatchesHandDerivedValue) {
    TrajectoryBalanceLoss loss;
    (void)loss.forward(1.0f, 0.5f, 2.0f, 0.3f);
    EXPECT_NEAR(loss.grad_weight_for_log_pf(), 2.4f, 1e-5f);
}

TEST(TrajectoryBalanceLossTest, ZeroDeltaGivesZeroLossAndZeroGradients) {
    // log_z + sum_log_pf == log_reward + sum_log_pb -> Delta == 0, the global optimum.
    TrajectoryBalanceLoss loss;
    const float value = loss.forward(/*sum_log_pf=*/0.5f, /*sum_log_pb=*/0.5f, /*log_reward=*/1.0f, /*log_z=*/1.0f);
    EXPECT_NEAR(value, 0.0f, 1e-6f);
    EXPECT_NEAR(loss.delta(), 0.0f, 1e-6f);
    EXPECT_NEAR(loss.grad_log_z(), 0.0f, 1e-6f);
    EXPECT_NEAR(loss.grad_weight_for_log_pf(), 0.0f, 1e-6f);
}

TEST(TrajectoryBalanceLossTest, AccessorsThrowBeforeForward) {
    TrajectoryBalanceLoss loss;
    EXPECT_THROW({ (void)loss.delta(); }, std::logic_error);
    EXPECT_THROW({ (void)loss.grad_log_z(); }, std::logic_error);
    EXPECT_THROW({ (void)loss.grad_weight_for_log_pf(); }, std::logic_error);
}

}  // namespace
}  // namespace pulsatrix
