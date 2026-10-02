#include "pulsatrix/gflownet_trajectory.hpp"

#include <cmath>
#include <vector>

namespace pulsatrix {

GFlowNetTrajectory sample_gflownet_trajectory(HyperGridEnv& env, GFlowNetForwardPolicy& forward_policy) {
    GFlowNetTrajectory trajectory;

    Tensor observation = env.reset();
    while (true) {
        const std::vector<bool> mask = env.valid_actions_mask(observation);
        const GFlowNetSampledAction sampled = forward_policy.sample(observation, mask);
        // Host boundary (GPU-native-kernels Mission 7): the sampled action may live on the
        // policy backend's device; one device->host copy of it.
        const int64_t action_index = static_cast<int64_t>(std::round(sampled.action.to_host_vector()[0]));

        trajectory.states.push_back(observation);
        trajectory.actions.push_back(action_index);
        trajectory.sum_log_pf += sampled.log_prob;

        const StepResult result = env.step(sampled.action);

        // stop (action_index == ndim()) does not move the state -- no P_B term for it. Every
        // other action is a real move, contributing a backward-transition probability
        // regardless of whether this step happens to also be the one that hits the max_steps
        // cap.
        if (action_index < env.ndim()) {
            trajectory.sum_log_pb += env.backward_log_prob(result.observation, action_index);
        }

        observation = result.observation;
        if (result.done) {
            trajectory.terminal_reward = result.reward;
            break;
        }
    }

    return trajectory;
}

}  // namespace pulsatrix
