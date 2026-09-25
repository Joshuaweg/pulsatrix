/** @file categorical_policy_agent.hpp
 *  @brief Stochastic categorical (discrete-action) policy over a logit-producing Module.
 */
#pragma once

#include <cstdint>

#include "pulsatrix/agent.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief Samples an action from `softmax(policy_network(observation))` -- the discrete-action
 *        stochastic policy every policy-gradient method in this phase (REINFORCE, A2C, PPO)
 *        is built on.
 *
 * @note The policy network outputs raw *logits*, not probabilities. The numerically stable
 *       softmax (subtract the row max before exponentiating) is computed here, fused into the
 *       agent, for exactly the reason CrossEntropyLoss fuses softmax with NLL rather than
 *       composing a SoftmaxModule with a separate log: the log-probability this agent must
 *       report is `logit - max - log(sum exp(logit - max))`, which never materializes a
 *       probability that could underflow to zero and then be logged.
 * @note Takes a `Module*` rather than defining a policy-network type of its own, and does not
 *       own it -- identical convention to DQNAgent's `q_network`. Must outlive this agent.
 * @note `log_prob()` is an extra method *beyond* the Agent interface, not an extension of it.
 *       `Agent::act()` returns only an action Tensor, but policy-gradient methods need the
 *       log-probability the policy assigned to the action it just sampled (RolloutBuffer
 *       stores it). Resolved the same way DQNAgent already added `act_greedy()`/`set_epsilon()`
 *       beyond the interface: a non-interface method reading cached forward-pass state, the
 *       same convention every Module already uses for its own `last_*_` cache.
 * @note Sampling randomness comes from an internal deterministic LCG seeded at construction --
 *       never `\<random\>`, whose engine outputs beyond mt19937 are implementation-defined.
 *       Byte-for-byte the generator and constants CartPoleEnv, ReplayBuffer and DQNAgent use,
 *       which is the only thing that makes a stochastic policy hand-traceable in a test.
 */
class CategoricalPolicyAgent : public Agent {
public:
    /**
     * @brief Constructs a categorical policy over a logit-producing network.
     * @param policy_network Network mapping a (1, observation_dim) observation to
     *        (1, action_dim) raw logits. Not owned; must outlive this agent.
     * @param action_dim Number of discrete actions. Must be >= 1.
     * @param backend Backend to allocate the returned action Tensor through. Not owned; must
     *        outlive this agent.
     * @param seed Seed for the internal deterministic LCG.
     * @throws std::invalid_argument if policy_network is null or action_dim <= 0 -- external
     *         boundaries, same classification as DQNAgent's identical checks.
     */
    CategoricalPolicyAgent(Module* policy_network, int64_t action_dim, DeviceBackend* backend, uint32_t seed = 42);

    /**
     * @brief Samples an action from the policy's categorical distribution, advancing the LCG.
     * @param observation Observation, shape (1, observation_dim) -- whatever policy_network
     *        accepts.
     * @return The sampled action index encoded as a (1, 1) float Tensor, identical to the
     *         encoding Environment::step() expects for a discrete environment.
     * @throws std::invalid_argument if policy_network's output is not (1, action_dim) --
     *         external boundary: the network and action_dim are two independent constructor
     *         arguments a caller can genuinely mismatch, and an unchecked scan would then read
     *         past the output row.
     * @note Exactly one LCG draw is consumed per call, unconditionally: `u` uniform in [0, 1),
     *       then `action` is the smallest index whose cumulative probability reaches `u`
     *       (inverse-CDF sampling -- a single cumulative-sum scan, not a rejection loop, so the
     *       number of LCG steps is content-independent and the stream stays easy to reason
     *       about in a determinism test).
     * @note Caches the sampled action's log-probability for log_prob(), taken straight from the
     *       stable log-softmax rather than re-derived as `log(p[action])` -- one rounding path,
     *       not two possibly inconsistent ones.
     * @note Raw host loop over the network output's Tensor::data() (softmax and the inverse-CDF
     *       scan have no DeviceBackend primitive) -- PULSATRIX_ASSERT(observation.device() ==
     *       DeviceType::Cpu) guards against silent UB on a CUDA-backed Tensor; see
     *       mission_host_loop_guards.md.
     */
    [[nodiscard]] Tensor act(const Tensor& observation) override;

    /**
     * @brief The deterministic (evaluation-time) policy: `argmax_a logits[a]`, no sampling.
     * @param observation Observation, shape (1, observation_dim).
     * @return The greedy action index as a (1, 1) float Tensor.
     * @throws std::invalid_argument if policy_network's output is not (1, action_dim).
     * @note argmax over the logits is argmax over the softmax probabilities -- softmax is
     *       strictly monotone -- so no exponentiation is needed here at all.
     * @note Consumes no randomness, advances no state, and deliberately does *not* update
     *       log_prob(): an evaluation rollout interleaved with training can therefore neither
     *       perturb the training run's action sequence nor corrupt the log-probability the
     *       training loop is about to store for the last genuinely sampled action. That is what
     *       makes this a separate method rather than "call act() and ignore the randomness",
     *       mirroring DQNAgent::act_greedy exactly.
     * @note Not const despite its name's suggestion of purity: Module::forward() is non-const
     *       (every module caches forward state for its own backward()), so running the network
     *       necessarily mutates it -- identical reasoning to DQNAgent::act_greedy.
     */
    [[nodiscard]] Tensor act_greedy(const Tensor& observation);

    /**
     * @brief The log-probability the policy assigned to the most recent act() sample.
     * @return `log_softmax[action]` for the action act() last returned.
     * @throws std::logic_error if act() has never been called -- there is no meaningful
     *         "probability of no action", and returning 0.0f (a probability of 1) would be a
     *         silently plausible lie that a training loop would happily backpropagate.
     * @note Unaffected by act_greedy(), which samples nothing.
     */
    [[nodiscard]] float log_prob() const;

    /** @brief Number of discrete actions, as passed to the constructor. */
    [[nodiscard]] int64_t action_dim() const { return action_dim_; }

private:
    /** @brief Advances the LCG one step and returns a uniform draw in [0, 1) -- half-open, so
     *         the inverse-CDF scan can never be handed a `u` above the final cumulative
     *         probability of 1. Byte-for-byte DQNAgent::next_unit. */
    [[nodiscard]] float next_unit();

    /** @brief Runs the policy network and validates its output is exactly (1, action_dim). */
    [[nodiscard]] Tensor policy_logits(const Tensor& observation);

    Module* policy_network_;
    int64_t action_dim_;
    DeviceBackend* backend_;
    uint32_t lcg_state_;
    float last_log_prob_ = 0.0f;
    bool has_acted_ = false;
};

}  // namespace pulsatrix
