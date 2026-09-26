/** @file trajectory_balance_loss.hpp
 *  @brief GFlowNet Trajectory Balance loss (Malkin et al. 2022, arXiv:2201.13259).
 */
#pragma once

namespace pulsatrix {

/**
 * @brief `Δ(τ) = log Zθ + Σ log P_F(s_{t+1}|s_t) − log R(x) − Σ log P_B(s_t|s_{t+1})`,
 *        `loss = Δ(τ)²`.
 *
 * @note Deliberately does not follow `MSELoss`/`PolicyGradientLoss`'s single-tensor
 *       `backward()` shape: `Δ` feeds two heterogeneous consumers (a bare scalar `log Z`, and
 *       a *sequence* of per-step policy-network logits from different forward-policy calls at
 *       different states) that no single gradient tensor covers. Instead exposes `delta()`
 *       (the one real error signal) plus two named derived accessors. See
 *       `mission_trajectory_balance_loss.md`'s Design section for the full derivation.
 * @note Not a Module subclass, for `MSELoss`'s own reason: a loss is the seed point relevance
 *       propagation starts from, not something `propagate_relevance` is defined for.
 */
class TrajectoryBalanceLoss {
public:
    TrajectoryBalanceLoss() = default;

    /**
     * @brief Computes `Δ(τ)` and the loss `Δ(τ)²`, caching `Δ` for the accessors below.
     * @param sum_log_pf `Σ_t log P_F(s_{t+1}|s_t)` over the whole trajectory.
     * @param sum_log_pb `Σ_t log P_B(s_t|s_{t+1})` over the whole trajectory.
     * @param log_reward `log R(x)` at the trajectory's terminal state.
     * @param log_z The current `log Zθ` estimate (`LearnableScalar::value()`).
     * @return The scalar loss, `Δ(τ)²`.
     */
    [[nodiscard]] float forward(float sum_log_pf, float sum_log_pb, float log_reward, float log_z);

    /** @brief `Δ(τ)` from the most recent forward() call. */
    [[nodiscard]] float delta() const;

    /** @brief `d(loss)/d(log Z) = 2Δ` -- feed directly into `LearnableScalar::accumulate_grad`. */
    [[nodiscard]] float grad_log_z() const;

    /**
     * @brief The weight `w = -2Δ` such that the per-trajectory-step policy-network gradient
     *        is `w · (p[k] − 1{k==a_t})` -- `PolicyGradientLoss`'s exact softmax/log gradient
     *        identity shape, with this trajectory's `Δ` playing the role of the per-step
     *        return.
     */
    [[nodiscard]] float grad_weight_for_log_pf() const;

private:
    float last_delta_ = 0.0f;
    bool has_forwarded_ = false;

    /** @brief Throws std::logic_error if forward() has never been called. Shared by every
     *         accessor below. */
    void require_forwarded() const;
};

}  // namespace pulsatrix
