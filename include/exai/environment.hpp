/** @file environment.hpp
 *  @brief Abstract RL environment interface (gymnasium-shaped reset/step) + StepResult.
 */
#pragma once

#include <cstdint>

#include "exai/tensor.hpp"

namespace exai {

/**
 * @brief What one Environment::step() produces: the next observation, this step's reward,
 *        and whether the episode ended.
 * @note Plain data, no behavior -- same precedent as ReparamGrad (reparameterize.hpp): a
 *       small named return-tuple beats an out-parameter or a std::tuple whose members are
 *       positional and therefore unreadable at every call site.
 * @note Deliberately no `truncated` flag (gymnasium's five-tuple splits time-limit
 *       truncation from genuine termination). This campaign's Phase 1 folds both into
 *       `done`, matching the classic Gym four-tuple; splitting them is a bootstrapping
 *       concern for value-function targets that no algorithm in this codebase exists to
 *       care about yet, and adding it later is additive.
 */
struct StepResult {
    Tensor observation;
    float reward;
    bool done;
};

/**
 * @brief Base class for every RL environment (CartPoleEnv, and whatever later phases add).
 *
 * @note Deliberately NOT a Module subclass. A Module is a differentiable, parameterized
 *       layer with an LRP rule and a gradient; an environment has no parameters, no
 *       gradient, and no relevance to propagate. Forcing the two together would put a
 *       pure-virtual propagate_relevance() on a type for which the concept is undefined --
 *       exactly the "no default rule" failure mode module.hpp's charter note rules out.
 *       Environment/Agent are therefore a new, independent interface family.
 * @note Gymnasium-API-shaped on purpose (campaign Risk Register mitigation): reset/step,
 *       reward, done. There is no Python dependency here, but a future pybind11 binding
 *       should be able to wrap this in a `gymnasium.Env` with minimal adaptation rather
 *       than a translation layer.
 * @note Observations are always shape (1, observation_dim()) -- this codebase's established
 *       always-batched convention (SinusoidalTimestepEmbedding's own (1, D) decision), so
 *       an observation feeds straight into any Module::forward() without a reshape.
 * @note Actions are always a Tensor, including for discrete action spaces, where the action
 *       is a shape-(1,1) Tensor holding the action index as a float. is_discrete() tells a
 *       caller which interpretation applies. This keeps the "everything is a Tensor"
 *       uniformity the rest of the codebase already has.
 */
class Environment {
public:
    virtual ~Environment() = default;

    /**
     * @brief Starts a new episode from an implementation-chosen initial state.
     * @return The initial observation, shape (1, observation_dim()).
     * @note Any randomness is expected to come from an internal *deterministic* generator
     *       seeded at construction, not from a global/nondeterministic source -- the same
     *       "randomness is a reproducible, caller-controlled input" convention as
     *       Reparameterize's caller-supplied epsilon and NoiseSchedule's epsilon/z.
     */
    [[nodiscard]] virtual Tensor reset() = 0;

    /**
     * @brief Starts a new episode from an exact caller-supplied initial state.
     * @param initial_state State to start from, shape (1, observation_dim()).
     * @return The initial observation (the state just set), shape (1, observation_dim()).
     * @throws std::invalid_argument if initial_state's shape is wrong -- external boundary.
     * @note Pure-virtual rather than a base-class default that validates and delegates to a
     *       protected set_state hook: the mission leaves the choice to the implementer, and
     *       a pure-virtual pair keeps Environment a true pure interface with no state and no
     *       protected extension point, which is the simpler contract to subclass against.
     *       Each environment implements both overloads directly.
     */
    [[nodiscard]] virtual Tensor reset(const Tensor& initial_state) = 0;

    /**
     * @brief Advances the environment one timestep under the given action.
     * @param action The action to take, shape (1, 1) for a discrete environment.
     * @return The resulting observation, reward and done flag.
     * @throws std::invalid_argument if the action is malformed or the environment has not
     *         been reset yet (implementation-defined which preconditions apply).
     */
    [[nodiscard]] virtual StepResult step(const Tensor& action) = 0;

    /** @brief Number of components in an observation vector. */
    [[nodiscard]] virtual int64_t observation_dim() const = 0;

    /**
     * @brief The action space's size: the number of distinct actions when is_discrete(),
     *        otherwise the dimension of a continuous action vector.
     */
    [[nodiscard]] virtual int64_t action_dim() const = 0;

    /** @brief Whether action_dim() counts discrete choices (true) or vector components. */
    [[nodiscard]] virtual bool is_discrete() const = 0;
};

}  // namespace exai
