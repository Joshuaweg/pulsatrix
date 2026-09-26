/** @file gflownet_forward_policy.hpp
 *  @brief Masked stochastic categorical policy over a logit-producing Module -- P_F for
 *         GFlowNet training objectives.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief One sample from GFlowNetForwardPolicy::sample: the chosen action and the
 *         log-probability the masked distribution assigned to it. */
struct GFlowNetSampledAction {
    Tensor action;
    float log_prob;
};

/**
 * @brief Samples from `softmax(mask(policy_network(observation)))` -- a categorical policy
 *        restricted to a caller-supplied set of valid actions at the current state.
 *
 * @note Byte-for-byte `CategoricalPolicyAgent`'s LCG (Numerical Recipes constants) and
 *       numerically-stable log-softmax + inverse-CDF sampling, deliberately duplicated rather
 *       than reusing that class directly: `CategoricalPolicyAgent::act()` has no notion of a
 *       per-call action mask (every existing consumer -- CartPole, etc. -- has no invalid
 *       actions), and `HyperGridEnv::step()` throws on an off-grid increment, so a forward
 *       policy that ever sampled one would crash training. See
 *       mission_shared_gflownet_machinery.md's Recon for why this is a new, justified class
 *       rather than a modification to a closed, tested class from a different campaign.
 * @note Masking is the standard additive-mask technique: an invalid action's logit is driven
 *       to `std::numeric_limits<float>::lowest()` before softmax, giving it ~0 probability
 *       without risking a NaN from actual `-infinity` arithmetic.
 * @note Returns `{action, log_prob}` directly from `sample()` rather than `CategoricalPolicyAgent`'s
 *       stateful `act()` + separate `log_prob()` accessor -- a cleaner API this new class is
 *       free to choose, since nothing else depends on matching that older class's shape.
 */
class GFlowNetForwardPolicy {
public:
    /**
     * @brief Constructs a masked categorical policy over a logit-producing network.
     * @param policy_network Network mapping a (1, observation_dim) observation to
     *        (1, action_dim) raw logits. Not owned; must outlive this object.
     * @param action_dim Number of actions. Must be >= 1.
     * @param backend Backend to allocate the returned action Tensor through. Not owned; must
     *        outlive this object.
     * @param seed Seed for the internal deterministic LCG.
     * @throws std::invalid_argument if policy_network is null or action_dim <= 0.
     */
    GFlowNetForwardPolicy(Module* policy_network, int64_t action_dim, DeviceBackend* backend, uint32_t seed = 42);

    /**
     * @brief Samples an action from the policy's distribution, restricted to valid_actions.
     * @param observation Observation, shape (1, observation_dim) -- whatever policy_network
     *        accepts.
     * @param valid_actions Boolean mask, size action_dim(). At least one entry must be true.
     * @return The sampled action index (shape (1, 1) float Tensor) and the log-probability
     *         the masked distribution assigned to it.
     * @throws std::invalid_argument if policy_network's output is not (1, action_dim), if
     *         valid_actions.size() != action_dim(), or if every entry of valid_actions is
     *         false (no valid action to sample) -- all external boundary.
     * @note Exactly one LCG draw is consumed per call, unconditionally -- same determinism
     *       convention as CategoricalPolicyAgent::act().
     */
    [[nodiscard]] GFlowNetSampledAction sample(const Tensor& observation, const std::vector<bool>& valid_actions);

    /**
     * @brief The masked categorical distribution's probabilities, without sampling.
     * @param observation Observation, shape (1, observation_dim).
     * @param valid_actions Same mask contract as sample().
     * @return `softmax(mask(policy_network(observation)))`, size action_dim().
     * @throws Same conditions as sample(), except no LCG draw is consumed and nothing is
     *         sampled -- a training loop's two-pass backward step (mission_trajectory_balance_loss.md's
     *         Design section) needs to recompute the *exact* distribution sample() used, to
     *         derive the softmax/log gradient identity, without perturbing the LCG stream a
     *         second act() would.
     */
    [[nodiscard]] std::vector<float> masked_probs(const Tensor& observation, const std::vector<bool>& valid_actions);

    /** @brief Number of actions, as passed to the constructor. */
    [[nodiscard]] int64_t action_dim() const { return action_dim_; }

private:
    /** @brief Advances the LCG one step and returns a uniform draw in [0, 1). */
    [[nodiscard]] float next_unit();

    Module* policy_network_;
    int64_t action_dim_;
    DeviceBackend* backend_;
    uint32_t lcg_state_;
};

}  // namespace pulsatrix
