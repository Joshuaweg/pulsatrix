/** @file dqn_loss.hpp
 *  @brief DQN's masked-MSE Bellman loss -- squared error on the taken action only.
 *  @ingroup rl
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief loss = mean_b( (q_values[b, a_b] - targets[b,0])^2 ), where `a_b` is the action
 *        actually taken on transition `b` -- the semi-gradient TD update of Mnih et al. 2015.
 *
 * This is MSELoss masked by the action: of the `action_dim` Q-values the online network
 * produced for a state, exactly one -- the Q-value of the action the behaviour policy
 * actually took -- has a Bellman target to be regressed against. The other columns are not
 * "predicted wrong", they are simply unobserved on this transition, so they receive exactly
 * zero gradient. That zero pattern in backward() is the defining property of the DQN update,
 * not an optimization.
 *
 * @note The Bellman target is *not* recomputed here. The caller (a training loop) computes it
 *       once via ComputeDQNTarget()/ComputeDoubleDQNTarget() -- under no-gradient semantics,
 *       which is what makes the update semi-gradient -- and passes it in as an ordinary
 *       Tensor. That keeps this class target-agnostic in exactly the way MSELoss is
 *       prediction-agnostic, and is what lets vanilla DQN and Double DQN share one loss.
 * @note Not a Module subclass, for MSELoss/BCEWithLogitsLoss's reason: losses are the seed
 *       point relevance/gradient propagation starts *from*, not something a
 *       `propagate_relevance` rule is defined for -- LRP explains a model's prediction, not
 *       the loss function used to train it.
 * @note backward() takes no incoming gradient: like MSELoss, this loss is a graph root.
 */
class DQNLoss {
public:
    /**
     * @brief Constructs a DQN loss.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this loss.
     */
    explicit DQNLoss(DeviceBackend* backend);

    /**
     * @brief Computes the masked MSE and caches everything backward() needs.
     * @param q_values The online network's full output, shape (N, action_dim), action_dim >= 1.
     * @param actions Float-encoded action indices, shape (N, 1) -- the same discrete-action
     *        encoding Environment::step() accepts, so a batch straight out of a ReplayBuffer
     *        needs no conversion.
     * @param targets Precomputed Bellman targets, shape (N, 1).
     * @return The scalar mean squared TD error.
     * @throws std::invalid_argument if any tensor has the wrong rank, if the three batch
     *         dimensions disagree, if an encoded action is not within 1e-4 of a whole number,
     *         or if a decoded index falls outside [0, action_dim) -- all external boundaries.
     *         The integer tolerance is byte-for-byte the one CartPoleEnv::step() already
     *         established for decoding a discrete action: loose enough to tolerate a policy's
     *         float round-trip, tight enough to catch a genuinely fractional "action" (an
     *         un-argmaxed probability, say).
     * @note Device-generic (GPU-native-kernels Mission 7): the N encoded actions are copied to
     *       the host once for validation, then the squared TD errors run through
     *       DeviceBackend::rl_rows(DqnLoss) and are summed with column_sums in row order.
     */
    [[nodiscard]] float forward(const Tensor& q_values, const Tensor& actions, const Tensor& targets);

    /**
     * @brief Gradient w.r.t. `q_values`: 2*(q_values[b,a_b] - targets[b,0])/N in the taken
     *        action's column, exactly 0.0f in every other column.
     * @return Gradient tensor, shape (N, action_dim) -- the shape of the q_values passed to
     *         forward(), so it feeds straight into the Q-network's own backward().
     * @throws std::logic_error if forward() has never been called -- uses the cached state.
     * @note The `2/N` factor is the same mean-square scaling MSELoss::backward() already
     *       folds in; N is the batch size, not the element count, because the mean is taken
     *       over transitions (one squared error each), not over all N*action_dim Q-values.
     * @note Device-generic: one DeviceBackend::rl_rows(DqnGrad) pass writes the taken action's
     *       element of each row into the zero-initialized gradient.
     */
    [[nodiscard]] Tensor backward() const;

private:
    DeviceBackend* backend_;
    Tensor last_q_values_;
    Tensor last_targets_;
    /** @brief Decoded, already-validated action index per batch row, shape (N, 1), as
     *         whole-number floats on backend_'s device -- decoded once in forward() rather than
     *         re-derived (and re-validated) in backward(). */
    Tensor last_action_indices_;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
