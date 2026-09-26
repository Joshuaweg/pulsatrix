/** @file agent.hpp
 *  @brief Abstract RL agent interface -- the inference-time policy contract, act() only.
 *  @ingroup rl
 */
#pragma once

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief Base class for anything that maps an observation to an action.
 *
 * @note Exactly one method. There is deliberately no learn()/update() on this interface:
 *       training is algorithm-specific (DQN's replay-buffer TD update, REINFORCE/A2C/PPO's
 *       policy-gradient step, SAC's twin-critic update all have genuinely different
 *       signatures), so a common update() would either be a lowest-common-denominator lie
 *       or an unused pure-virtual every subclass stubs. Agent is the *policy* contract only;
 *       each later phase's algorithm owns its own training surface.
 * @note Not a Module subclass, for the same reason Environment isn't: an Agent may not be
 *       differentiable at all (a tabular or epsilon-greedy policy isn't), and typically
 *       *owns* Modules rather than being one.
 */
class Agent {
public:
    virtual ~Agent() = default;

    /**
     * @brief Chooses an action for the given observation.
     * @param observation Observation from an Environment, shape (1, observation_dim()).
     * @return The chosen action, in whatever encoding the target Environment expects --
     *         shape (1, 1) holding the action index as a float, for a discrete environment.
     * @note Non-const: a stochastic policy advances its own internal generator state here,
     *       and an epsilon-greedy policy decays epsilon, so act() is genuinely a mutating
     *       operation for most real implementations.
     */
    [[nodiscard]] virtual Tensor act(const Tensor& observation) = 0;
};

}  // namespace pulsatrix
