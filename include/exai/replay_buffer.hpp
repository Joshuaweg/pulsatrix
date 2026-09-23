/** @file replay_buffer.hpp
 *  @brief Off-policy experience replay: fixed-capacity circular transition store + sampling.
 */
#pragma once

#include <cstdint>

#include "exai/device_backend.hpp"
#include "exai/tensor.hpp"

namespace exai {

/**
 * @brief One uniformly-sampled minibatch of transitions, one Tensor per transition field.
 * @note Plain data, no behavior -- the same small named return-struct precedent as
 *       ReparamGrad (reparameterize.hpp) and StepResult (environment.hpp): five positional
 *       members in a std::tuple would be unreadable at every call site, and five out-
 *       parameters would be worse.
 * @note All five tensors share the same leading dimension, the batch_size passed to
 *       ReplayBuffer::sample(), and row b of every tensor belongs to the same transition.
 *       observations/next_observations are (batch_size, observation_dim), actions is
 *       (batch_size, action_dim), rewards and dones are (batch_size, 1).
 * @note `dones` holds 0.0f/1.0f floats rather than bools -- this codebase's established
 *       "everything is a Tensor" convention, the same one that makes Environment encode a
 *       discrete action index as a float. A DQN target computation wants (1 - done) as a
 *       float multiplier anyway, so no conversion is imposed on the caller.
 */
struct ReplayBatch {
    Tensor observations;
    Tensor actions;
    Tensor rewards;
    Tensor next_observations;
    Tensor dones;
};

/**
 * @brief Fixed-capacity circular replay buffer of (observation, action, reward,
 *        next_observation, done) transitions, with uniform-random-with-replacement batch
 *        sampling -- the off-policy experience store of Mnih et al. 2015 (DQN), reused
 *        unchanged by SAC and every other off-policy learner in this campaign.
 *
 * Once `capacity` transitions have been stored, the oldest is overwritten by the next
 * add(): memory is bounded regardless of how long training runs.
 *
 * @note Not a Module subclass, and not an Environment/Agent either. It has no parameters,
 *       no gradient, no forward/backward and nothing for an LRP rule to explain -- forcing
 *       it into Module would put a pure-virtual propagate_relevance() on a type for which
 *       the concept is undefined, exactly the failure mode module.hpp's charter note rules
 *       out. Same disposition as MSELoss and Reparameterize: a plain utility class.
 * @note Environment-agnostic by construction. It is built from bare observation_dim /
 *       action_dim integers, not from an Environment&, so it neither depends on nor
 *       outlives any particular environment; CartPoleEnv is simply one source of the
 *       tensors a caller happens to add().
 * @note Storage is five pre-allocated (capacity, *)-shaped Tensors written into at a
 *       circular index, not a std::vector of per-transition tensors. That is one allocation
 *       per field for the buffer's whole lifetime instead of five per stored transition,
 *       and it makes sample() a set of row-gathers rather than a loop of tensor copies.
 * @note Discrete and continuous actions are stored identically, as action_dim floats -- a
 *       discrete action is its single-element encoded index, exactly as Environment::step()
 *       already accepts it. No buffer-side special-casing.
 */
class ReplayBuffer {
public:
    /**
     * @brief Constructs an empty buffer with all storage pre-allocated and zero-filled.
     * @param capacity Maximum number of transitions retained. Must be >= 1.
     * @param observation_dim Width of an observation row. Must be >= 1.
     * @param action_dim Width of an action row (Environment::action_dim()). Must be >= 1.
     * @param backend Backend to allocate through. Not owned; must outlive this buffer.
     * @param seed Seed for the internal deterministic LCG used by sample().
     * @throws std::invalid_argument if capacity, observation_dim or action_dim is <= 0 --
     *         external boundary, the same classification as CartPoleEnv's max_steps check.
     */
    ReplayBuffer(int64_t capacity, int64_t observation_dim, int64_t action_dim, DeviceBackend* backend,
                 uint32_t seed = 42);

    /**
     * @brief Stores one transition at the current circular write index, overwriting the
     *        oldest transition once the buffer is full.
     * @param observation State before the action, shape (1, observation_dim()).
     * @param action Action taken, shape (1, action_dim()).
     * @param reward Scalar reward received.
     * @param next_observation State after the action, shape (1, observation_dim()).
     * @param done Whether the episode ended on this transition.
     * @throws std::invalid_argument if any of the three tensors has the wrong shape --
     *         external boundary: a mismatched-shape Tensor can arrive from any caller, and
     *         silently writing it would corrupt neighbouring rows of the storage block.
     * @note EXAI_ASSERT(... .device() == DeviceType::Cpu) on all three tensor arguments --
     *       this is a raw host-loop row copy dereferencing Tensor::data() directly, not yet
     *       backend-generic, so a CUDA-backed Tensor would be silent UB. Same convention as
     *       every prior host-loop site (mission_host_loop_guards.md); do not remove without
     *       actually routing the copy through DeviceBackend.
     */
    void add(const Tensor& observation, const Tensor& action, float reward, const Tensor& next_observation,
             bool done);

    /**
     * @brief Draws batch_size transitions uniformly at random, *with* replacement.
     * @param batch_size Number of transitions to draw. Must be in [1, size()].
     * @return The sampled transitions, one Tensor per field (see ReplayBatch).
     * @throws std::invalid_argument if batch_size <= 0, or if batch_size > size() (which
     *         includes every call on an empty buffer). External boundary: sampling more than
     *         has actually been stored is a real caller error worth a real message, not
     *         something to satisfy with garbage rows from never-written slots.
     * @note With replacement, so a batch may legitimately repeat a transition -- the
     *       standard DQN formulation, and what keeps batch_size == size() valid rather than
     *       a degenerate full-buffer permutation.
     * @note Indices come from the internal LCG seeded at construction (the same
     *       Numerical-Recipes constants CartPoleEnv::reset() and this codebase's tests
     *       already use), never from `\<random\>`, whose engines are implementation-defined:
     *       two buffers with the same seed and the same contents sample identically, which
     *       is what makes sampling testable at all.
     */
    [[nodiscard]] ReplayBatch sample(int64_t batch_size);

    /** @brief Transitions currently stored -- rises to capacity(), then stays there. */
    [[nodiscard]] int64_t size() const { return size_; }

    /** @brief Maximum transitions retained before the oldest starts being overwritten. */
    [[nodiscard]] int64_t capacity() const { return capacity_; }

    /** @brief Width of an observation row, as passed to the constructor. */
    [[nodiscard]] int64_t observation_dim() const { return observation_dim_; }

    /** @brief Width of an action row, as passed to the constructor. */
    [[nodiscard]] int64_t action_dim() const { return action_dim_; }

private:
    /** @brief Advances the LCG one step and returns a uniform index in [0, bound). */
    [[nodiscard]] int64_t next_index(int64_t bound);

    int64_t capacity_;
    int64_t observation_dim_;
    int64_t action_dim_;
    DeviceBackend* backend_;
    uint32_t lcg_state_;

    int64_t size_ = 0;
    int64_t write_index_ = 0;

    Tensor observations_;
    Tensor actions_;
    Tensor rewards_;
    Tensor next_observations_;
    Tensor dones_;
};

}  // namespace exai
