/** @file gflownet_forward_policy_test.cpp
 *  @brief GFlowNetForwardPolicy construction validation, masking, and sampling determinism.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/gflownet_forward_policy.hpp"
#include "pulsatrix/linear_module.hpp"

namespace pulsatrix {
namespace {

class GFlowNetForwardPolicyTest : public ::testing::Test {
protected:
    CPUBackend backend;

    Tensor obs(float x0, float x1) { return Tensor(Shape({1, 2}), &backend, {x0, x1}); }
};

TEST_F(GFlowNetForwardPolicyTest, ConstructionThrowsOnNullNetwork) {
    EXPECT_THROW({ GFlowNetForwardPolicy policy(nullptr, 3, &backend); }, std::invalid_argument);
}

TEST_F(GFlowNetForwardPolicyTest, ConstructionThrowsOnZeroActionDim) {
    LinearModule net(2, 3, &backend);
    EXPECT_THROW({ GFlowNetForwardPolicy policy(&net, 0, &backend); }, std::invalid_argument);
}

TEST_F(GFlowNetForwardPolicyTest, SampleThrowsOnWrongMaskSize) {
    LinearModule net(2, 3, &backend);
    GFlowNetForwardPolicy policy(&net, 3, &backend);
    std::vector<bool> wrong_size_mask = {true, true};
    EXPECT_THROW({ (void)policy.sample(obs(0.0f, 0.0f), wrong_size_mask); }, std::invalid_argument);
}

TEST_F(GFlowNetForwardPolicyTest, SampleThrowsOnAllInvalidMask) {
    LinearModule net(2, 3, &backend);
    GFlowNetForwardPolicy policy(&net, 3, &backend);
    std::vector<bool> all_invalid = {false, false, false};
    EXPECT_THROW({ (void)policy.sample(obs(0.0f, 0.0f), all_invalid); }, std::invalid_argument);
}

TEST_F(GFlowNetForwardPolicyTest, SampleThrowsOnWrongNetworkOutputShape) {
    LinearModule net(2, 4, &backend);  // action_dim mismatch: policy expects 3
    GFlowNetForwardPolicy policy(&net, 3, &backend);
    std::vector<bool> mask = {true, true, true};
    EXPECT_THROW({ (void)policy.sample(obs(0.0f, 0.0f), mask); }, std::invalid_argument);
}

TEST_F(GFlowNetForwardPolicyTest, OnlySamplesValidActions) {
    // Uniform (zero) logits -- without masking, all 3 actions would be equally likely. With
    // only action 1 valid, every one of many draws must return action 1.
    LinearModule net(2, 3, &backend);  // zero-initialized weight/bias -> logits are (0, 0, 0)
    GFlowNetForwardPolicy policy(&net, 3, &backend, /*seed=*/7);
    std::vector<bool> only_one_valid = {false, true, false};
    for (int i = 0; i < 20; ++i) {
        GFlowNetSampledAction result = policy.sample(obs(0.0f, 0.0f), only_one_valid);
        EXPECT_FLOAT_EQ(result.action.data()[0], 1.0f);
        EXPECT_NEAR(result.log_prob, 0.0f, 1e-4f);  // log(1.0) -- the only valid action has probability 1
    }
}

TEST_F(GFlowNetForwardPolicyTest, UniformLogitsGiveUniformLogProbOverValidActions) {
    // Zero logits over 2 valid actions (of 3 total, one masked out) -> each valid action has
    // probability exactly 0.5, log_prob == log(0.5) == -ln(2).
    LinearModule net(2, 3, &backend);
    GFlowNetForwardPolicy policy(&net, 3, &backend, /*seed=*/11);
    std::vector<bool> two_valid = {true, true, false};
    const float expected_log_prob = std::log(0.5f);
    bool saw_action_0 = false;
    bool saw_action_1 = false;
    for (int i = 0; i < 50; ++i) {
        GFlowNetSampledAction result = policy.sample(obs(0.0f, 0.0f), two_valid);
        const float a = result.action.data()[0];
        EXPECT_TRUE(a == 0.0f || a == 1.0f);
        EXPECT_NEAR(result.log_prob, expected_log_prob, 1e-4f);
        if (a == 0.0f) saw_action_0 = true;
        if (a == 1.0f) saw_action_1 = true;
    }
    EXPECT_TRUE(saw_action_0);
    EXPECT_TRUE(saw_action_1);
}

TEST_F(GFlowNetForwardPolicyTest, SameSeedProducesSameSequence) {
    LinearModule net_a(2, 3, &backend);
    LinearModule net_b(2, 3, &backend);
    GFlowNetForwardPolicy policy_a(&net_a, 3, &backend, /*seed=*/99);
    GFlowNetForwardPolicy policy_b(&net_b, 3, &backend, /*seed=*/99);
    std::vector<bool> all_valid = {true, true, true};
    for (int i = 0; i < 10; ++i) {
        GFlowNetSampledAction a = policy_a.sample(obs(0.0f, 0.0f), all_valid);
        GFlowNetSampledAction b = policy_b.sample(obs(0.0f, 0.0f), all_valid);
        EXPECT_FLOAT_EQ(a.action.data()[0], b.action.data()[0]);
        EXPECT_FLOAT_EQ(a.log_prob, b.log_prob);
    }
}

}  // namespace
}  // namespace pulsatrix
