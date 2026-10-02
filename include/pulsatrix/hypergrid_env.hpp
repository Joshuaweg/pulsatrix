/** @file hypergrid_env.hpp
 *  @brief HyperGrid -- the standard minimal GFlowNet correctness-check environment.
 *  @ingroup mech_interp
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/environment.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief An n-dimensional grid: state is an integer coordinate in `[0, H-1]^ndim`, actions
 *        increment one coordinate or stop the episode, reward is concentrated near the
 *        grid's corners.
 *
 * @note Deliberately not a reproduction of any single paper's exact reward constants -- see
 *       mission_hypergrid_env.md's Design section for the reward shape this class implements
 *       and why it was chosen (a real, non-degenerate near-corner band distinct from the
 *       exact-corner set, hand-verifiable at named cells for the default side_length).
 * @note Deterministic reset() to the origin -- unlike CartPoleEnv's randomized reset, this
 *       environment's whole purpose is a fixed, hand-traceable start state.
 * @note Reward is 0 at every non-terminal step and R(x) only at the terminal state (the
 *       `stop` action or the max_steps cap) -- the standard GFlowNet convention that R is a
 *       property of the *terminal object* x, not of any intermediate transition.
 */
class HyperGridEnv : public Environment {
public:
    /**
     * @brief Constructs a fresh, not-yet-reset HyperGrid environment.
     * @param backend Backend to allocate observation tensors through. Not owned; must
     *        outlive this object.
     * @param ndim Number of grid dimensions. Must be >= 1.
     * @param side_length Number of cells per dimension (H). Must be >= 4 -- below that the
     *        near-corner band (Design section) degenerates to exactly the corner set, which
     *        would make R1 and R2 indistinguishable rather than a real distinct band.
     * @param r0 Base reward, everywhere. Must be >= 0.
     * @param r1 Additional reward in the near-corner band. Must be >= 0.
     * @param r2 Additional reward exactly at a corner. Must be >= 0.
     * @param max_steps Episode length safety cap -- forces termination (as if `stop` were
     *        taken) if reached without one. Must be >= 1.
     * @throws std::invalid_argument on any violated precondition above -- external boundary,
     *         same classification as CartPoleEnv's constructor.
     */
    explicit HyperGridEnv(DeviceBackend* backend, int64_t ndim = 2, int64_t side_length = 8, float r0 = 0.1f,
                           float r1 = 0.5f, float r2 = 2.0f, int64_t max_steps = 64);

    /**
     * @brief Starts a new episode at the origin `(0, ..., 0)`.
     * @return The initial observation, shape (1, ndim()).
     * @note No randomness -- HyperGrid's whole point is a fixed, hand-traceable start state.
     */
    [[nodiscard]] Tensor reset() override;

    /**
     * @brief Starts a new episode from an exact caller-supplied state.
     * @param initial_state Grid coordinate, shape (1, ndim()), each entry an integer-valued
     *        float in `[0, side_length()-1]`.
     * @return The initial observation (a copy of initial_state's values), shape (1, ndim()).
     * @throws std::invalid_argument if initial_state's shape is wrong, an entry is not within
     *         a small tolerance of an integer, or an entry is out of `[0, side_length()-1]` --
     *         all external boundary.
     */
    [[nodiscard]] Tensor reset(const Tensor& initial_state) override;

    /**
     * @brief Advances the episode one step: increments a coordinate, or stops.
     * @param action Shape (1, 1), holding the action index as a float: `[0, ndim())` increment
     *        that coordinate, `ndim()` stops the episode at the current state.
     * @return The resulting observation, reward (0 unless this step is terminal), and whether
     *         the episode ended (via `stop`, an implicit max_steps cap, or neither).
     * @throws std::invalid_argument if reset() has never been called, the action's shape is
     *         wrong, the decoded index is out of range, or an increment action would move a
     *         coordinate past `side_length()-1` (illegal off-grid move) -- all external
     *         boundary: any of these can originate from an untrained or malformed policy.
     * @throws std::logic_error if the most recent step already reported `done=true` -- a
     *         GFlowNet trajectory has no meaning past its terminal state; unlike CartPoleEnv,
     *         this is enforced here rather than left to the caller.
     * @note Host boundary (GPU-native-kernels Mission 7): the grid walk is integer host logic.
     *       `action` (and every `state` argument of reset()/reward()/backward_log_prob()/
     *       valid_actions_mask()) may live on any device -- one device->host copy per call --
     *       and observations are returned through this environment's own backend.
     */
    [[nodiscard]] StepResult step(const Tensor& action) override;

    /**
     * @brief The reward of an arbitrary valid grid state, independent of this environment's
     *        current episode state.
     * @param state Grid coordinate, shape (1, ndim()), each entry an integer-valued float in
     *        `[0, side_length()-1]`.
     * @return `R(state)` per the Design section's formula.
     * @throws std::invalid_argument if state's shape or entries are invalid -- external
     *         boundary, same validation as reset(const Tensor&).
     * @note A pure function -- no precondition on reset() having been called, no mutation.
     *       Later missions (TrajectoryBalanceLoss and friends) need R(x) at a sampled terminal
     *       state directly, without re-driving the environment through step().
     */
    [[nodiscard]] float reward(const Tensor& state) const;

    /**
     * @brief The closed-form uniform backward-policy log-probability P_B(a|state).
     * @param state Grid coordinate, shape (1, ndim()), each entry an integer-valued float in
     *        `[0, side_length()-1]`.
     * @param action Dimension index `[0, ndim())` -- decrementing this coordinate must reach
     *        a valid parent state.
     * @return `log(1 / count_nonzero(state))` -- HyperGrid's action structure (exactly one
     *         way to reach any non-origin state, by incrementing one coordinate) makes the
     *         *correct* backward distribution uniform over `state`'s nonzero coordinates; see
     *         mission_shared_gflownet_machinery.md's Recon for why this is closed-form rather
     *         than a second learned policy.
     * @throws std::invalid_argument if state is invalid (same validation as reward()), if
     *         action is out of `[0, ndim())`, or if `state[action] == 0` (decrementing it
     *         would leave the grid -- not a valid parent transition) -- all external boundary.
     * @note A pure function, like reward() -- no precondition on reset() having been called.
     */
    [[nodiscard]] float backward_log_prob(const Tensor& state, int64_t action) const;

    /**
     * @brief Which actions are legal to take from an arbitrary valid state.
     * @param state Grid coordinate, shape (1, ndim()), each entry an integer-valued float in
     *        `[0, side_length()-1]`.
     * @return A mask of size action_dim(): entry `i < ndim()` is true iff
     *         `state[i] < side_length()-1` (incrementing that coordinate stays on the grid);
     *         entry `ndim()` (`stop`) is always true.
     * @throws std::invalid_argument if state is invalid -- same validation as reward().
     * @note A pure function, like reward()/backward_log_prob() -- no precondition on reset()
     *       having been called. `GFlowNetForwardPolicy::sample()` needs this mask to avoid
     *       ever sampling an illegal increment that would make step() throw.
     */
    [[nodiscard]] std::vector<bool> valid_actions_mask(const Tensor& state) const;

    /** @brief Number of grid dimensions, as passed to the constructor. */
    [[nodiscard]] int64_t observation_dim() const override { return ndim_; }

    /** @brief `ndim() + 1` -- one increment action per dimension, plus `stop`. */
    [[nodiscard]] int64_t action_dim() const override { return ndim_ + 1; }

    /** @brief True -- HyperGrid's action space is discrete. */
    [[nodiscard]] bool is_discrete() const override { return true; }

    /** @brief Number of grid dimensions. Alias for observation_dim(), named for readability
     *         at HyperGrid-specific call sites. */
    [[nodiscard]] int64_t ndim() const { return ndim_; }

    /** @brief Number of cells per dimension (H), as passed to the constructor. */
    [[nodiscard]] int64_t side_length() const { return side_length_; }

    /** @brief Episode length safety cap this environment was constructed with. */
    [[nodiscard]] int64_t max_steps() const { return max_steps_; }

    /** @brief Steps taken since the last reset(). */
    [[nodiscard]] int64_t step_count() const { return step_count_; }

private:
    /** @brief Packs the current coordinate into a (1, ndim()) Tensor. */
    [[nodiscard]] Tensor observation() const;

    /** @brief Validates a state Tensor's shape and integer-in-range entries, decoding it into
     *         `out` (resized to ndim()). Shared by reset(const Tensor&) and reward(). */
    void decode_state(const Tensor& state, std::vector<int64_t>& out) const;

    DeviceBackend* backend_;
    int64_t ndim_;
    int64_t side_length_;
    float r0_;
    float r1_;
    float r2_;
    int64_t max_steps_;

    std::vector<int64_t> coord_;
    int64_t step_count_ = 0;
    bool has_reset_ = false;
    bool done_ = false;
};

}  // namespace pulsatrix
