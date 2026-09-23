/** @file cartpole_env.hpp
 *  @brief CartPole-v1 environment -- classic cart-pole balancing physics, discrete actions.
 */
#pragma once

#include <cstdint>

#include "exai/device_backend.hpp"
#include "exai/environment.hpp"
#include "exai/tensor.hpp"

namespace exai {

/**
 * @brief The classic cart-pole balancing task (Barto, Sutton & Anderson 1983), the same
 *        equations OpenAI Gym's own CartPoleEnv implements -- cited as the canonical,
 *        independently-verifiable reference for this class's correctness tests, not reused
 *        as a code or runtime dependency.
 *
 * State is (x, x_dot, theta, theta_dot): cart position, cart velocity, pole angle from
 * vertical (radians), pole angular velocity. The observation is exactly that 4-vector,
 * shape (1, 4). Two discrete actions: 0 pushes the cart left, 1 pushes it right.
 *
 * Every step yields reward 1.0, including the terminating step (Gym's own convention -- the
 * agent is rewarded for having survived *through* this step). The episode ends when
 * |x| > 2.4, |theta| > 0.20943951 rad (~12 degrees), or max_steps steps have been taken.
 *
 * @note The physics constants (gravity, masses, pole length, force magnitude, integration
 *       timestep, termination thresholds) are fixed, not constructor-configurable -- a
 *       deliberate scope cut. Parameterizing them is a future mission's job if a real need
 *       appears; a knob with exactly one used value is speculative generality.
 * @note Integration is explicit (forward) Euler in Gym's exact order: x and theta are
 *       updated from the *pre-update* velocities. Semi-implicit Euler is a one-line
 *       difference that produces measurably different trajectories; do not "fix" this.
 * @note Physics is computed in double and stored in double; only the observation Tensor is
 *       float. Repeated float-precision Euler steps drift enough over a 200-step episode to
 *       make a hand-derived reference trajectory unreproducible, which would undermine this
 *       class's whole reason for existing.
 */
class CartPoleEnv : public Environment {
public:
    /** @brief Cart position past which the episode terminates. */
    static constexpr double kXThreshold = 2.4;
    /** @brief Pole angle (radians) past which the episode terminates (~12 degrees). */
    static constexpr double kThetaThreshold = 0.20943951;

    /**
     * @brief Constructs a fresh, not-yet-reset CartPole environment.
     * @param backend Backend to allocate observation tensors through. Not owned; must
     *        outlive this object.
     * @param max_steps Episode length limit. Must be >= 1. Defaults to 200 -- deliberately
     *        below Gym's canonical 500, since this is a correctness-test environment rather
     *        than a benchmark.
     * @param seed Seed for the internal deterministic LCG used by the no-argument reset().
     * @throws std::invalid_argument if max_steps < 1 -- external boundary.
     * @note The DeviceBackend* parameter is not in the mission's stated signature, but a
     *       Tensor cannot be constructed without one, and every other allocating type in
     *       this codebase takes the backend by injection rather than reaching for a global.
     *       Resolved that way; max_steps/seed keep their stated defaults and order.
     */
    explicit CartPoleEnv(DeviceBackend* backend, int64_t max_steps = 200, uint32_t seed = 42);

    /**
     * @brief Starts a new episode from a small random state: each of the 4 state variables
     *        drawn i.i.d. uniform in [-0.05, 0.05] from the internal LCG.
     * @return The initial observation, shape (1, 4).
     * @note The LCG (the same Numerical-Recipes constants this codebase's tests already use
     *       for reproducible pseudo-randomness) is seeded at construction and advances
     *       across resets, so two environments built with the same seed produce identical
     *       reset sequences and different seeds produce different ones -- deterministic and
     *       therefore testable, unlike the standard library's implementation-defined
     *       engines (`\<random\>`).
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
     * @brief Advances the physics one timestep (tau = 0.02 s) under the given action.
     * @param action Shape (1, 1), holding the action index as a float: 0 (push left) or
     *        1 (push right), within 1e-4 of an integer.
     * @return The resulting observation (1, 4), reward 1.0, and whether the episode ended.
     * @throws std::invalid_argument if neither reset() overload has been called yet, if the
     *         action's shape is not (1, 1), or if the decoded index is not within 1e-4 of
     *         an integer in [0, action_dim()). All external boundary: an action can
     *         originate from an untrusted policy output or, eventually, Python bindings.
     * @note EXAI_ASSERT(action.device() == DeviceType::Cpu) -- this is a raw host-loop
     *       physics update dereferencing Tensor::data() directly, not yet backend-generic,
     *       so a CUDA-backed action Tensor would be silent UB. Same convention as every
     *       prior module; do not remove without actually routing through DeviceBackend.
     * @note Stepping past a done=true result is allowed and keeps integrating; enforcing
     *       "reset after done" would be a second precondition with no caller to serve, and
     *       Gym itself only warns. The episode loop is the caller's responsibility.
     */
    [[nodiscard]] StepResult step(const Tensor& action) override;

    /** @brief 4 -- (x, x_dot, theta, theta_dot). */
    [[nodiscard]] int64_t observation_dim() const override { return 4; }

    /** @brief 2 -- push left (0) or push right (1). */
    [[nodiscard]] int64_t action_dim() const override { return 2; }

    /** @brief True -- CartPole's action space is discrete. */
    [[nodiscard]] bool is_discrete() const override { return true; }

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

}  // namespace exai
