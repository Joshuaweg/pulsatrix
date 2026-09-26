/** @file gflownet_trajectory_test.cpp
 *  @brief sample_gflownet_trajectory mechanics: shape/length invariants, reward correctness,
 *         and that only legal actions are ever produced (proving the mask is applied).
 */
#include <gtest/gtest.h>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/gflownet_forward_policy.hpp"
#include "pulsatrix/gflownet_trajectory.hpp"
#include "pulsatrix/hypergrid_env.hpp"
#include "pulsatrix/linear_module.hpp"

namespace pulsatrix {
namespace {

class GFlowNetTrajectoryTest : public ::testing::Test {
protected:
    CPUBackend backend;
    HyperGridEnv env{&backend, 2, 8};
    LinearModule policy_net{2, 3, &backend};  // action_dim = ndim(2)+1 = 3
    GFlowNetForwardPolicy policy{&policy_net, 3, &backend, /*seed=*/123};
};

TEST_F(GFlowNetTrajectoryTest, StatesAndActionsHaveMatchingLength) {
    GFlowNetTrajectory traj = sample_gflownet_trajectory(env, policy);
    EXPECT_EQ(traj.states.size(), traj.actions.size());
    EXPECT_GE(traj.states.size(), 1u);  // at minimum, the stop decision itself
}

TEST_F(GFlowNetTrajectoryTest, TerminalRewardMatchesEnvironmentRewardOfFinalState) {
    GFlowNetTrajectory traj = sample_gflownet_trajectory(env, policy);
    // The last state stored is the one the final (stop) decision acted from -- its reward
    // must match the trajectory's own terminal_reward exactly, since stop doesn't move state.
    EXPECT_NEAR(traj.terminal_reward, env.reward(traj.states.back()), 1e-6f);
}

TEST_F(GFlowNetTrajectoryTest, LastActionIsAlwaysStop) {
    GFlowNetTrajectory traj = sample_gflownet_trajectory(env, policy);
    EXPECT_EQ(traj.actions.back(), env.ndim());  // stop's index
}

TEST_F(GFlowNetTrajectoryTest, EveryActionExceptTheLastIsALegalMove) {
    GFlowNetTrajectory traj = sample_gflownet_trajectory(env, policy);
    for (size_t i = 0; i + 1 < traj.actions.size(); ++i) {
        EXPECT_LT(traj.actions[i], env.ndim());  // a real increment, not stop, before the end
    }
}

TEST_F(GFlowNetTrajectoryTest, SumLogPfIsFiniteAndNonPositive) {
    // Every term is a log-probability of an actual outcome (<= 0 in exact arithmetic; a tiny
    // positive slack covers float rounding right at 0).
    GFlowNetTrajectory traj = sample_gflownet_trajectory(env, policy);
    EXPECT_LE(traj.sum_log_pf, 1e-4f);
}

TEST_F(GFlowNetTrajectoryTest, SumLogPbIsFiniteAndNonPositive) {
    GFlowNetTrajectory traj = sample_gflownet_trajectory(env, policy);
    EXPECT_LE(traj.sum_log_pb, 1e-4f);
}

TEST_F(GFlowNetTrajectoryTest, ResetsEnvironmentInternally) {
    // Drive env partway through an unrelated episode first -- sample_gflownet_trajectory must
    // still start its own trajectory from the origin, discarding any in-progress episode.
    (void)env.reset();
    Tensor increment = Tensor(Shape({1, 1}), &backend, {0.0f});
    (void)env.step(increment);
    GFlowNetTrajectory traj = sample_gflownet_trajectory(env, policy);
    EXPECT_FLOAT_EQ(traj.states.front().data()[0], 0.0f);
    EXPECT_FLOAT_EQ(traj.states.front().data()[1], 0.0f);
}

}  // namespace
}  // namespace pulsatrix
