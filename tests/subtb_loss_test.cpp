/** @file subtb_loss_test.cpp
 *  @brief SubTBLoss forward/gradient-accessor correctness, including reduction tests proving
 *         it matches TrajectoryBalanceLoss and DetailedBalanceLoss exactly at ratio=1.0.
 *
 * Hand-derived reference: log_flow_i=0.8, sum_log_pf=-0.4, log_flow_j=1.5, sum_log_pb=-0.3,
 * pair_weight_ratio=0.5.
 * Delta = 0.8 - 0.4 - 1.5 + 0.3 = -0.8. loss = 0.5 * 0.64 = 0.32.
 * grad_log_flow_i = 0.5*2*(-0.8) = -0.8. grad_log_flow_j = 0.5*(-2)*(-0.8) = 0.8.
 * grad_weight_for_log_pf_range = 0.5*(-2)*(-0.8) = 0.8.
 */
#include <gtest/gtest.h>

#include <stdexcept>

#include "pulsatrix/detailed_balance_loss.hpp"
#include "pulsatrix/subtb_loss.hpp"
#include "pulsatrix/trajectory_balance_loss.hpp"

namespace pulsatrix {
namespace {

TEST(SubTBLossTest, ForwardMatchesHandDerivedValue) {
    SubTBLoss loss;
    const float value = loss.forward(0.8f, -0.4f, 1.5f, -0.3f, 0.5f);
    EXPECT_NEAR(value, 0.32f, 1e-5f);
}

TEST(SubTBLossTest, DeltaMatchesHandDerivedValue) {
    SubTBLoss loss;
    (void)loss.forward(0.8f, -0.4f, 1.5f, -0.3f, 0.5f);
    EXPECT_NEAR(loss.delta(), -0.8f, 1e-5f);
}

TEST(SubTBLossTest, GradLogFlowIMatchesHandDerivedValue) {
    SubTBLoss loss;
    (void)loss.forward(0.8f, -0.4f, 1.5f, -0.3f, 0.5f);
    EXPECT_NEAR(loss.grad_log_flow_i(), -0.8f, 1e-5f);
}

TEST(SubTBLossTest, GradLogFlowJMatchesHandDerivedValue) {
    SubTBLoss loss;
    (void)loss.forward(0.8f, -0.4f, 1.5f, -0.3f, 0.5f);
    EXPECT_NEAR(loss.grad_log_flow_j(), 0.8f, 1e-5f);
}

TEST(SubTBLossTest, GradWeightForLogPfRangeMatchesHandDerivedValue) {
    SubTBLoss loss;
    (void)loss.forward(0.8f, -0.4f, 1.5f, -0.3f, 0.5f);
    EXPECT_NEAR(loss.grad_weight_for_log_pf_range(), 0.8f, 1e-5f);
}

TEST(SubTBLossTest, ZeroDeltaGivesZeroLossAndZeroGradients) {
    SubTBLoss loss;
    const float value = loss.forward(/*log_flow_i=*/0.5f, /*sum_log_pf=*/-0.2f, /*log_flow_j=*/1.0f,
                                      /*sum_log_pb=*/-0.7f, /*pair_weight_ratio=*/0.3f);
    EXPECT_NEAR(value, 0.0f, 1e-6f);
    EXPECT_NEAR(loss.delta(), 0.0f, 1e-6f);
    EXPECT_NEAR(loss.grad_log_flow_i(), 0.0f, 1e-6f);
    EXPECT_NEAR(loss.grad_log_flow_j(), 0.0f, 1e-6f);
    EXPECT_NEAR(loss.grad_weight_for_log_pf_range(), 0.0f, 1e-6f);
}

TEST(SubTBLossTest, AccessorsThrowBeforeForward) {
    SubTBLoss loss;
    EXPECT_THROW({ (void)loss.delta(); }, std::logic_error);
    EXPECT_THROW({ (void)loss.grad_log_flow_i(); }, std::logic_error);
    EXPECT_THROW({ (void)loss.grad_log_flow_j(); }, std::logic_error);
    EXPECT_THROW({ (void)loss.grad_weight_for_log_pf_range(); }, std::logic_error);
}

// ---------------------------------------------------------------------------------------
// Reduction tests: at pair_weight_ratio=1.0, SubTBLoss must match TB/DB's own formulas
// exactly for matching inputs -- both are literal special cases of the same math.
// ---------------------------------------------------------------------------------------

TEST(SubTBLossTest, ReducesToTrajectoryBalanceLossAtRatioOne) {
    // Same inputs as trajectory_balance_loss_test.cpp's own hand-derived case.
    TrajectoryBalanceLoss tb;
    const float tb_value = tb.forward(1.0f, 0.5f, 2.0f, 0.3f);

    SubTBLoss subtb;
    const float subtb_value = subtb.forward(/*log_flow_i=*/0.3f, /*sum_log_pf=*/1.0f, /*log_flow_j=*/2.0f,
                                             /*sum_log_pb=*/0.5f, /*pair_weight_ratio=*/1.0f);

    EXPECT_NEAR(subtb_value, tb_value, 1e-6f);
    EXPECT_NEAR(subtb.delta(), tb.delta(), 1e-6f);
    EXPECT_NEAR(subtb.grad_log_flow_i(), tb.grad_log_z(), 1e-6f);
    EXPECT_NEAR(subtb.grad_log_flow_j(), -tb.grad_log_z(), 1e-6f);
    EXPECT_NEAR(subtb.grad_weight_for_log_pf_range(), tb.grad_weight_for_log_pf(), 1e-6f);
}

TEST(SubTBLossTest, ReducesToDetailedBalanceLossAtRatioOne) {
    // Same inputs as detailed_balance_loss_test.cpp's own hand-derived case.
    DetailedBalanceLoss db;
    const float db_value = db.forward(0.8f, -0.4f, 1.5f, -0.3f);

    SubTBLoss subtb;
    const float subtb_value = subtb.forward(/*log_flow_i=*/0.8f, /*sum_log_pf=*/-0.4f, /*log_flow_j=*/1.5f,
                                             /*sum_log_pb=*/-0.3f, /*pair_weight_ratio=*/1.0f);

    EXPECT_NEAR(subtb_value, db_value, 1e-6f);
    EXPECT_NEAR(subtb.delta(), db.delta(), 1e-6f);
    EXPECT_NEAR(subtb.grad_log_flow_i(), db.grad_log_flow_s(), 1e-6f);
    EXPECT_NEAR(subtb.grad_log_flow_j(), db.grad_log_flow_s_next(), 1e-6f);
    EXPECT_NEAR(subtb.grad_weight_for_log_pf_range(), db.grad_weight_for_log_pf(), 1e-6f);
}

}  // namespace
}  // namespace pulsatrix
