/** @file detailed_balance_loss_test.cpp
 *  @brief DetailedBalanceLoss forward/gradient-accessor correctness, including the exit-case
 *         reduction (log_pb=0.0).
 *
 * Hand-derived reference (a real, non-exit transition): log_flow_s=0.8, log_pf=-0.4,
 * log_flow_s_next=1.5, log_pb=-0.3.
 * Delta = 0.8 + (-0.4) - 1.5 - (-0.3) = 0.8 - 0.4 - 1.5 + 0.3 = -0.8.
 * Loss = 0.64. grad_log_flow_s = -1.6. grad_log_flow_s_next = 1.6. grad_weight_for_log_pf = 1.6.
 */
#include <gtest/gtest.h>

#include <stdexcept>

#include "pulsatrix/detailed_balance_loss.hpp"

namespace pulsatrix {
namespace {

TEST(DetailedBalanceLossTest, ForwardMatchesHandDerivedValue) {
    DetailedBalanceLoss loss;
    const float value = loss.forward(0.8f, -0.4f, 1.5f, -0.3f);
    EXPECT_NEAR(value, 0.64f, 1e-5f);
}

TEST(DetailedBalanceLossTest, DeltaMatchesHandDerivedValue) {
    DetailedBalanceLoss loss;
    (void)loss.forward(0.8f, -0.4f, 1.5f, -0.3f);
    EXPECT_NEAR(loss.delta(), -0.8f, 1e-5f);
}

TEST(DetailedBalanceLossTest, GradLogFlowSMatchesHandDerivedValue) {
    DetailedBalanceLoss loss;
    (void)loss.forward(0.8f, -0.4f, 1.5f, -0.3f);
    EXPECT_NEAR(loss.grad_log_flow_s(), -1.6f, 1e-5f);
}

TEST(DetailedBalanceLossTest, GradLogFlowSNextMatchesHandDerivedValue) {
    DetailedBalanceLoss loss;
    (void)loss.forward(0.8f, -0.4f, 1.5f, -0.3f);
    EXPECT_NEAR(loss.grad_log_flow_s_next(), 1.6f, 1e-5f);
}

TEST(DetailedBalanceLossTest, GradWeightForLogPfMatchesHandDerivedValue) {
    DetailedBalanceLoss loss;
    (void)loss.forward(0.8f, -0.4f, 1.5f, -0.3f);
    EXPECT_NEAR(loss.grad_weight_for_log_pf(), 1.6f, 1e-5f);
}

TEST(DetailedBalanceLossTest, ZeroDeltaGivesZeroLossAndZeroGradients) {
    // log_flow_s + log_pf == log_flow_s_next + log_pb -> Delta == 0.
    DetailedBalanceLoss loss;
    const float value = loss.forward(/*log_flow_s=*/0.5f, /*log_pf=*/-0.2f, /*log_flow_s_next=*/1.0f,
                                      /*log_pb=*/-0.7f);
    EXPECT_NEAR(value, 0.0f, 1e-6f);
    EXPECT_NEAR(loss.delta(), 0.0f, 1e-6f);
    EXPECT_NEAR(loss.grad_log_flow_s(), 0.0f, 1e-6f);
    EXPECT_NEAR(loss.grad_log_flow_s_next(), 0.0f, 1e-6f);
    EXPECT_NEAR(loss.grad_weight_for_log_pf(), 0.0f, 1e-6f);
}

// Exit-case reduction: log_pb=0.0 (log(1), the trivial sink backward policy), log_flow_s_next
// is the fixed log-reward, not a learned flow. Delta = log_flow_s + log_pf - log_reward.
TEST(DetailedBalanceLossTest, ExitCaseReductionMatchesHandDerivedValue) {
    DetailedBalanceLoss loss;
    // log_flow_s=1.2, log_pf(stop)=-0.5, log_reward=0.955 (log(2.6)), log_pb=0.0.
    // Delta = 1.2 - 0.5 - 0.955 = -0.255.
    const float value = loss.forward(1.2f, -0.5f, 0.955f, 0.0f);
    EXPECT_NEAR(loss.delta(), -0.255f, 1e-5f);
    EXPECT_NEAR(value, 0.255f * 0.255f, 1e-5f);
}

TEST(DetailedBalanceLossTest, AccessorsThrowBeforeForward) {
    DetailedBalanceLoss loss;
    EXPECT_THROW({ (void)loss.delta(); }, std::logic_error);
    EXPECT_THROW({ (void)loss.grad_log_flow_s(); }, std::logic_error);
    EXPECT_THROW({ (void)loss.grad_log_flow_s_next(); }, std::logic_error);
    EXPECT_THROW({ (void)loss.grad_weight_for_log_pf(); }, std::logic_error);
}

}  // namespace
}  // namespace pulsatrix
