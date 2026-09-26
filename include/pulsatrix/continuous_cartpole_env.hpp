/** @file continuous_cartpole_env.hpp
 *  @brief Continuous-action variant of CartPoleEnv -- identical physics, force is a fraction.
 *  @ingroup rl
 */
#pragma once

#include <cstdint>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/environment.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief The cart-pole balancing task with a *continuous* action: the action is a single
 *        force fraction in [-1, 1] rather than a discrete left/right choice.
 *
 * The physics, constants, integration order, reward and termination thresholds are exactly
 * CartPoleEnv's (Barto, Sutton & Anderson 1983 / OpenAI Gym's CartPoleEnv equations, cited
 * as an independently-verifiable reference, not reused as a dependency). The *only*
 * difference is how `action` becomes `force`:
 *
 *   - CartPoleEnv:           force = (index == 1) ? +force_mag : -force_mag
 *   - ContinuousCartPoleEnv: force = clamp(action, -1, 1) * force_mag
 *
 * So `action = +1.0` is exactly CartPoleEnv's discrete `action = 1`, `action = -1.0` is
 * exactly its discrete `action = 0`, and every value between is a proportionally weaker
 * push. That equivalence at the extremes is a test in this class's suite, tying its physics
 * to the already-verified discrete environment rather than only to a re-derivation.
 *
 * State is (x, x_dot, theta, theta_dot); the observation is that 4-vector, shape (1, 4).
 * Every step yields reward 1.0, the terminating step included (Gym's convention). The
 * episode ends when |x| > 2.4, |theta| > 0.20943951 rad (~12 degrees), or max_steps steps
 * have been taken.
 *
 * @note Exists because Phase 4 (SAC) needs a continuous-control test case and CartPoleEnv is
 *       discrete-only. Deliberately a *sibling* class rather than a mode flag or a template
 *       parameter on CartPoleEnv: the discrete environment is already verified and depended
 *       on by four training-integration tests, and an action-interpretation branch inside its
 *       step() would make every one of those tests pay for a case it never takes. The
 *       duplicated physics block is ~10 lines and is pinned to the original by a direct
 *       trajectory-equality test, which is a stronger guarantee than shared code without one.
 * @note The physics constants are fixed, not constructor-configurable -- the same deliberate
 *       scope cut CartPoleEnv documents. A knob with exactly one used value is speculative.
 * @note Integration is explicit (forward) Euler in Gym's exact order: x and theta are updated
 *       from the *pre-update* velocities. Semi-implicit Euler is a one-line difference that
 *       produces measurably different trajectories; do not "fix" this.
 * @note Physics is computed and stored in double; only the observation Tensor is float.
 *       Repeated float-precision Euler steps drift enough over a 200-step episode to make a
 *       hand-derived reference trajectory unreproducible.
 */
class ContinuousCartPoleEnv : public Environment {
public:
    /** @brief Cart position past which the episode terminates. */
    static constexpr double kXThreshold = 2.4;
    /** @brief Pole angle (radians) past which the episode terminates (~12 degrees). */
    static constexpr double kThetaThreshold = 0.20943951;
    /**
     * @brief How far outside [-1, 1] an action may sit before step() rejects it.
     *
     * A tanh-squashed policy (what Phase 4's SAC actor is) mathematically stays inside
     * [-1, 1] up to float rounding, so a tolerance this small accommodates that round-trip
     * without silently accepting a genuinely out-of-range action. It is the same 1e-4
     * action-tolerance convention CartPoleEnv established, applied to a continuous bound
     * instead of an integer check.
     * @note Compared in double, not float: `1.0f + 1e-4f` rounds to the same float as
     *       `1.0001f` (floats near 1.0 are spaced ~1.2e-7 apart, and the two values differ by
     *       ~1.6e-8), so a float-domain comparison would accept 1.0001f -- a value that is
     *       unambiguously out of range. Widening to double makes the boundary mean what it
     *       says.
     */
    static constexpr double kActionRangeTolerance = 1e-4;

    /**
     * @brief Constructs a fresh, not-yet-reset continuous cart-pole environment.
     * @param backend Backend to allocate observation tensors through. Not owned; must
     *        outlive this object.
     * @param max_steps Episode length limit. Must be >= 1. Defaults to 200, matching
     *        CartPoleEnv.
     * @param seed Seed for the internal deterministic LCG used by the no-argument reset().
     * @throws std::invalid_argument if max_steps < 1 -- external boundary.
     */
    explicit ContinuousCartPoleEnv(DeviceBackend* backend, int64_t max_steps = 200, uint32_t seed = 42);

    /**
     * @brief Starts a new episode from a small random state: each of the 4 state variables
     *        drawn i.i.d. uniform in [-0.05, 0.05] from the internal LCG.
     * @return The initial observation, shape (1, 4).
     * @note Same Numerical-Recipes LCG, same seeding and advance-across-resets behavior as
     *       CartPoleEnv, so a given seed produces an identical reset sequence in both
     *       classes -- deterministic and therefore testable, unlike `\<random\>`'s
     *       implementation-defined engines.
     */
    [[nodiscard]] Tensor reset() override;

    /**
     * @brief Starts a new episode from an exact caller-supplied state, bypassing the LCG.
     * @param initial_state (x, x_dot, theta, theta_dot), shape (1, 4).
     * @return The initial observation (a copy of initial_state's values), shape (1, 4).
     * @throws std::invalid_argument if initial_state's shape is not (1, 4) -- external
     *         boundary.
     * @note Does not check the state against the termination thresholds: pinning an
     *       already-terminal state and confirming the very next step() reports done is a
     *       legitimate (and tested) use.
     */
    [[nodiscard]] Tensor reset(const Tensor& initial_state) override;

    /**
     * @brief Advances the physics one timestep (tau = 0.02 s) under the given continuous
     *        action.
     * @param action Shape (1, 1), holding a force fraction in
     *        [-1 - kActionRangeTolerance, 1 + kActionRangeTolerance]. The value is clamped to
     *        exactly [-1, 1] before `force = clamped * force_mag`, so a tiny float-rounding
     *        excess cannot produce a force beyond +/-force_mag.
     * @return The resulting observation (1, 4), reward 1.0, and whether the episode ended.
     * @throws std::invalid_argument if neither reset() overload has been called yet, if the
     *         action's shape is not (1, 1), or if the action is outside the tolerated range.
     *         All external boundary: an action can originate from an untrusted policy output
     *         or, eventually, Python bindings.
     * @note PULSATRIX_ASSERT(action.device() == DeviceType::Cpu) -- this is a raw host-loop
     *       physics update dereferencing Tensor::data() directly, not yet backend-generic, so
     *       a CUDA-backed action Tensor would be silent UB. Same convention as CartPoleEnv
     *       and every prior module; do not remove without actually routing through
     *       DeviceBackend.
     * @note NaN actions are rejected: every comparison against a NaN is false, so the
     *       range check is written as "reject unless inside the band" rather than "reject if
     *       outside it," which would let NaN through and poison the state permanently.
     * @note Stepping past a done=true result is allowed and keeps integrating; the episode
     *       loop is the caller's responsibility, exactly as in CartPoleEnv.
     */
    [[nodiscard]] StepResult step(const Tensor& action) override;

    /** @brief 4 -- (x, x_dot, theta, theta_dot). */
    [[nodiscard]] int64_t observation_dim() const override { return 4; }

    /** @brief 1 -- a single continuous force fraction. */
    [[nodiscard]] int64_t action_dim() const override { return 1; }

    /** @brief False -- this environment's action space is continuous. */
    [[nodiscard]] bool is_discrete() const override { return false; }

    /** @brief Steps taken since the last reset(). */
    [[nodiscard]] int64_t step_count() const { return step_count_; }

    /** @brief Episode length limit this environment was constructed with. */
    [[nodiscard]] int64_t max_steps() const { return max_steps_; }

private:
    /** @brief Packs the current (x, x_dot, theta, theta_dot) into a (1, 4) Tensor. */
    [[nodiscard]] Tensor observation() const;

    /** @brief Advances the LCG one step and returns a uniform draw in [-0.05, 0.05]. */
    [[nodiscard]] double next_uniform();

    DeviceBackend* backend_;
    int64_t max_steps_;
    uint32_t lcg_state_;

    double x_ = 0.0;
    double x_dot_ = 0.0;
    double theta_ = 0.0;
    double theta_dot_ = 0.0;

    int64_t step_count_ = 0;
    bool has_reset_ = false;
};

}  // namespace pulsatrix
