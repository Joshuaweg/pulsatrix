/** @file dqn_agent.hpp
 *  @brief Epsilon-greedy DQN policy over an arbitrary Q-network Module.
 *  @ingroup rl
 */
#pragma once

#include <cstdint>

#include "pulsatrix/agent.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief The epsilon-greedy behaviour policy of Mnih et al. 2015: with probability epsilon
 *        act uniformly at random, otherwise take `argmax_a Q(observation, a)`.
 *
 * @note Takes a `Module*` rather than defining a Q-network type of its own. A Q-network is
 *       just "some module mapping (1, observation_dim) to (1, action_dim)" -- a
 *       SequentialModule of LinearModule/ReluModule builds one with no new class, and a
 *       DQNAgent-specific subclass would only hard-code an architecture choice that belongs
 *       to the training loop. Not owned; must outlive this agent -- this codebase's universal
 *       non-owned-pointer convention.
 * @note No optimizer, no replay buffer, no target network, no loss. This is purely the
 *       policy half of DQN, exactly as Agent's interface defines it. The update half (sample
 *       a batch, compute targets via ComputeDQNTarget/ComputeDoubleDQNTarget, DQNLoss,
 *       backward, optimizer.step) lives in the training loop, the same split the GAN
 *       building blocks already established.
 * @note Exploration randomness comes from an internal deterministic LCG seeded at
 *       construction -- never `\<random\>`, whose engine outputs beyond mt19937 are
 *       implementation-defined. Two agents with the same seed and the same observations act
 *       identically, which is the only thing that makes an epsilon-greedy policy testable.
 *       Same generator and constants as CartPoleEnv and ReplayBuffer.
 */
class DQNAgent : public Agent {
public:
    /**
     * @brief Constructs an epsilon-greedy policy over a Q-network.
     * @param q_network Network mapping a (1, observation_dim) observation to (1, action_dim)
     *        Q-values. Not owned; must outlive this agent.
     * @param action_dim Number of discrete actions. Must be >= 1.
     * @param epsilon Initial exploration probability, in [0, 1].
     * @param backend Backend to allocate the returned action Tensor through. Not owned; must
     *        outlive this agent.
     * @param seed Seed for the internal deterministic LCG.
     * @throws std::invalid_argument if q_network is null, action_dim <= 0, or epsilon is
     *         outside [0, 1] -- external boundaries.
     */
    DQNAgent(Module* q_network, int64_t action_dim, float epsilon, DeviceBackend* backend, uint32_t seed = 42);

    /**
     * @brief Chooses an action epsilon-greedily, advancing the internal LCG.
     * @param observation Observation, shape (1, observation_dim) -- whatever q_network accepts.
     * @return The chosen action index encoded as a (1, 1) float Tensor, identical to the
     *         encoding Environment::step() expects for a discrete environment.
     * @throws std::invalid_argument if q_network's output is not (1, action_dim) -- external
     *         boundary: the network and action_dim are two independent constructor arguments
     *         a caller can genuinely mismatch, and an unchecked argmax would then read past
     *         the output row.
     * @note Exactly one LCG draw is consumed for the explore/exploit coin flip, plus a second
     *       draw for the action index if and only if that flip explores. The greedy branch
     *       therefore leaves the stream one step further along than it found it, which keeps
     *       an epsilon=0 agent's stream position well-defined rather than frozen.
     * @note Raw host loop over the network output's Tensor::data() (argmax has no
     *       DeviceBackend primitive) -- PULSATRIX_REQUIRE_HOST(observation)
     *       guards against silent UB on a CUDA-backed Tensor; see mission_host_loop_guards.md.
     */
    [[nodiscard]] Tensor act(const Tensor& observation) override;

    /**
     * @brief The purely greedy (evaluation-time) policy: argmax with no exploration at all.
     * @param observation Observation, shape (1, observation_dim).
     * @return The greedy action index as a (1, 1) float Tensor.
     * @throws std::invalid_argument if q_network's output is not (1, action_dim).
     * @note Consumes no randomness and advances no state, so calling it never perturbs a
     *       subsequent act() sequence -- an evaluation rollout interleaved with training
     *       cannot change the training run's trajectory. That is what makes this a separate
     *       method rather than "call act() with epsilon temporarily set to 0".
     * @note Not const despite its name's suggestion of purity: Module::forward() is non-const
     *       (every module caches forward state for its own backward()), so running the
     *       network necessarily mutates it. Marking this const would require either a
     *       const_cast or a mutable member -- both worse lies than the missing keyword.
     */
    [[nodiscard]] Tensor act_greedy(const Tensor& observation);

    /**
     * @brief Sets the exploration probability -- the hook an epsilon-decay schedule drives.
     * @param epsilon New exploration probability, in [0, 1].
     * @throws std::invalid_argument if epsilon is outside [0, 1] -- external boundary, same
     *         classification as the constructor's identical check.
     * @note The decay *schedule* itself deliberately lives in the training loop, not here: a
     *       linear/exponential/step schedule is a training hyperparameter, not a property of
     *       the policy, and baking one in would force every caller into that one choice.
     */
    void set_epsilon(float epsilon);

    /** @brief The current exploration probability. */
    [[nodiscard]] float epsilon() const { return epsilon_; }

    /** @brief Number of discrete actions, as passed to the constructor. */
    [[nodiscard]] int64_t action_dim() const { return action_dim_; }

private:
    /** @brief Advances the LCG one step and returns a uniform draw in [0, 1) -- half-open, so
     *         `u < epsilon` is false for every draw at epsilon == 0 and true for every draw
     *         at epsilon == 1. Both boundaries are exact, not merely near-certain. */
    [[nodiscard]] float next_unit();

    /** @brief Advances the LCG one step and returns a uniform index in [0, bound). */
    [[nodiscard]] int64_t next_index(int64_t bound);

    /** @brief Runs the Q-network and returns the row-wise argmax as a (1, 1) float Tensor. */
    [[nodiscard]] Tensor greedy_action(const Tensor& observation);

    Module* q_network_;
    int64_t action_dim_;
    float epsilon_;
    DeviceBackend* backend_;
    uint32_t lcg_state_;
};

}  // namespace pulsatrix
