/** @file rollout_buffer.hpp
 *  @brief On-policy trajectory storage: fill-once fixed-length rollout + discounted returns.
 */
#pragma once

#include <cstdint>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief One whole stored rollout, reduced to what a policy-gradient update consumes:
 *        the visited observations, the actions taken, the discounted return-to-go of each
 *        step, and the log-probability the acting policy assigned to each action.
 * @note Plain data, no behavior -- the same small named return-struct precedent as
 *       ReparamGrad (reparameterize.hpp), StepResult (environment.hpp) and ReplayBatch
 *       (replay_buffer.hpp): four positional members in a std::tuple would be unreadable at
 *       every call site.
 * @note All four tensors share the same leading dimension, RolloutBuffer::size() at the time
 *       compute_returns() was called, and row t of every tensor belongs to the same stored
 *       step. observations is (size, observation_dim), actions is (size, action_dim),
 *       returns and log_probs are (size, 1).
 * @note No `dones` member, unlike ReplayBatch. The done flags exist only to segment the
 *       return computation at episode boundaries; once `returns` has been computed they
 *       carry no further information a REINFORCE/A2C/PPO update uses, and handing them back
 *       would invite a caller to re-derive a reduction this buffer has already performed.
 */
struct RolloutBatch {
    Tensor observations;
    Tensor actions;
    Tensor returns;
    Tensor log_probs;
};

/**
 * @brief Fixed-length, fill-once on-policy trajectory buffer of
 *        (observation, action, reward, log_prob, done) steps, with discounted
 *        return-to-go computation -- the storage REINFORCE/A2C/PPO collect a rollout into.
 *
 * The usage protocol is a cycle: add() exactly max_length() steps (or fewer, then stop
 * collecting), compute_returns(gamma), do one gradient update, clear(), repeat.
 *
 * @note Deliberately NOT circular, unlike ReplayBuffer. An on-policy update consumes the
 *       whole rollout and then discards it -- reusing stale data for a second update is the
 *       on-policy/off-policy distinction itself -- so there is no rolling window to
 *       overwrite into. add() past max_length() therefore throws std::logic_error rather
 *       than wrapping around: the training loop forgot its compute-and-clear step, and
 *       silently dropping the oldest step of a trajectory would corrupt the return
 *       computation rather than merely losing data.
 * @note std::logic_error, not the std::invalid_argument every other throw here uses, and the
 *       distinction is deliberate: the arguments to an over-capacity add() are perfectly
 *       well-formed. What is wrong is the *sequence of calls*, a usage-protocol violation,
 *       not a malformed argument. The two are separately catchable on purpose.
 * @note Stores log_prob as a scalar per step, not the action distribution's parameters.
 *       That scalar is exactly what REINFORCE's policy-gradient estimator and PPO's
 *       probability ratio consume directly, and it is distribution-agnostic: a categorical,
 *       Gaussian or any other family's log-probability is still one float by the time it
 *       reaches this buffer, so the buffer never has to know which family produced it.
 * @note No `value` slot. Value-function estimates are needed by A2C/PPO's advantage
 *       computation but not by plain REINFORCE; baking one into a Phase 1 generic buffer
 *       would be speculative, and extending or wrapping this buffer when a later phase
 *       actually needs one is additive. Same "core mechanism, not the full block"
 *       discipline RWKVModule applied to a layer, applied here to a data structure.
 * @note Not a Module subclass, and not an Environment/Agent either. No parameters, no
 *       gradient, no forward/backward and nothing for an LRP rule to explain -- forcing it
 *       into Module would put a pure-virtual propagate_relevance() on a type for which the
 *       concept is undefined, exactly the failure mode module.hpp's charter note rules out.
 *       Same disposition as ReplayBuffer, MSELoss and Reparameterize: a plain utility class.
 * @note Storage is five pre-allocated (max_length, *)-shaped Tensors written into at the
 *       current length, not a std::vector of per-step tensors -- one allocation per field
 *       for the buffer's whole lifetime. clear() resets the length only; it never
 *       deallocates, so a training loop's repeated rollouts allocate nothing.
 */
class RolloutBuffer {
public:
    /**
     * @brief Constructs an empty buffer with all storage pre-allocated and zero-filled.
     * @param max_length Number of steps one rollout holds. Must be >= 1.
     * @param observation_dim Width of an observation row. Must be >= 1.
     * @param action_dim Width of an action row (Environment::action_dim()). Must be >= 1.
     * @param backend Backend to allocate through. Not owned; must outlive this buffer.
     * @throws std::invalid_argument if max_length, observation_dim or action_dim is <= 0 --
     *         external boundary, the same classification as ReplayBuffer's capacity check.
     */
    RolloutBuffer(int64_t max_length, int64_t observation_dim, int64_t action_dim, DeviceBackend* backend);

    /**
     * @brief Appends one step to the rollout.
     * @param observation State the action was chosen from, shape (1, observation_dim()).
     * @param action Action taken, shape (1, action_dim()).
     * @param reward Scalar reward received.
     * @param log_prob Log-probability the acting policy assigned to `action`.
     * @param done Whether the episode ended on this step (segments the return computation).
     * @throws std::invalid_argument if either tensor has the wrong shape -- external
     *         boundary: a mismatched-shape Tensor can arrive from any caller, and silently
     *         writing it would corrupt neighbouring rows of the storage block.
     * @throws std::logic_error if size() already equals max_length(). See the class note:
     *         a full rollout is a protocol violation, not a malformed argument, and the two
     *         exception types are distinct so a caller can tell them apart.
     * @note PULSATRIX_ASSERT(... .device() == DeviceType::Cpu) on both tensor arguments -- this is
     *       a raw host-loop row copy dereferencing Tensor::data() directly, not yet
     *       backend-generic, so a CUDA-backed Tensor would be silent UB. Same convention as
     *       every prior host-loop site (mission_host_loop_guards.md); do not remove without
     *       actually routing the copy through DeviceBackend.
     */
    void add(const Tensor& observation, const Tensor& action, float reward, float log_prob, bool done);

    /**
     * @brief Reduces the stored rollout to per-step discounted return-to-go,
     *        `G_t = sum_{k=t}^{T-1} gamma^(k-t) * r_k`, alongside the stored
     *        observations/actions/log_probs.
     * @param gamma Discount factor, must be in (0, 1]. gamma == 1 is legal (undiscounted).
     * @return The whole rollout as a RolloutBatch, leading dimension size().
     * @throws std::invalid_argument if gamma <= 0 or gamma > 1 -- external boundary; those
     *         are not discount factors, and a negative or >1 gamma would produce
     *         sign-alternating or divergent returns rather than an error the caller notices.
     * @note Computed in a single reverse pass, `G_t = r_t + gamma * G_{t+1}`, with the
     *       running accumulator reset to 0 at every step whose `done` flag is set, *before*
     *       that step's own reward is added. A rollout may span several episodes when
     *       episodes are shorter than max_length(), and without that reset episode k+1's
     *       return would silently leak backwards into episode k's final steps -- a bootstrap
     *       across a terminal state, which is exactly the value the `done` flag exists to
     *       forbid. The returns are therefore per-episode, not rollout-cumulative.
     * @note const: this is a pure reduction over the stored steps. It does not consume or
     *       clear the buffer, so calling it twice yields the same batch; clear() is the
     *       caller's explicit, separate step once the update has been applied.
     */
    [[nodiscard]] RolloutBatch compute_returns(float gamma) const;

    /**
     * @brief The raw per-step rewards exactly as add()-ed, shape (size(), 1), in stored order.
     * @return A fresh (size(), 1) Tensor; the buffer's own storage is not exposed.
     * @note Added for PPO (Phase 3 Mission 4), and deliberately *not* speculative API surface.
     *       The class note on RolloutBatch reasoned that once `returns` has been computed the
     *       raw rewards/dones "carry no further information a REINFORCE/A2C/PPO update uses" --
     *       true of REINFORCE and A2C, and false of PPO: ComputeGAE() performs its own,
     *       differently-weighted reduction of the raw rewards/dones and never calls
     *       compute_returns(). So this is not "re-deriving a reduction this buffer already
     *       performed"; it is the input to a *different* reduction the buffer does not perform.
     * @note Purely a read of already-stored state -- no new logic, and add()/compute_returns()/
     *       clear()/RolloutBatch are all untouched by its addition.
     */
    [[nodiscard]] Tensor rewards() const;

    /**
     * @brief The raw per-step termination flags as 0.0f/1.0f floats, shape (size(), 1), in
     *        stored order -- the same encoding ReplayBatch, ComputeDQNTarget and ComputeGAE use.
     * @return A fresh (size(), 1) Tensor; the buffer's own storage is not exposed.
     * @note Same rationale as rewards(): ComputeGAE() cuts both its bootstrap and its trace
     *       recursion on these flags, so a PPO training loop needs them per step rather than
     *       only as the episode segmentation compute_returns() already applied internally.
     */
    [[nodiscard]] Tensor dones() const;

    /**
     * @brief Empties the rollout, making the buffer reusable for the next one.
     * @note Resets the length to 0 only. The storage Tensors stay allocated at max_length()
     *       capacity and their stale contents are simply unreachable, since every read path
     *       is bounded by size(); zero-filling them would be work no observer can detect.
     */
    void clear();

    /** @brief Steps currently stored -- rises to max_length(), never past it. */
    [[nodiscard]] int64_t size() const { return size_; }

    /** @brief Steps one rollout holds, as passed to the constructor. */
    [[nodiscard]] int64_t max_length() const { return max_length_; }

    /** @brief Width of an observation row, as passed to the constructor. */
    [[nodiscard]] int64_t observation_dim() const { return observation_dim_; }

    /** @brief Width of an action row, as passed to the constructor. */
    [[nodiscard]] int64_t action_dim() const { return action_dim_; }

private:
    int64_t max_length_;
    int64_t observation_dim_;
    int64_t action_dim_;
    DeviceBackend* backend_;

    int64_t size_ = 0;

    Tensor observations_;
    Tensor actions_;
    Tensor rewards_;
    Tensor log_probs_;
    Tensor dones_;
};

}  // namespace pulsatrix
