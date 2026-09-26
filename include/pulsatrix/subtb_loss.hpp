/** @file subtb_loss.hpp
 *  @brief GFlowNet SubTB(lambda) loss (Madan et al., "Learning GFlowNets from partial
 *         episodes for improved convergence and stability", arXiv:2209.12782).
 */
#pragma once

namespace pulsatrix {

/**
 * @brief One sub-trajectory pair's contribution to the SubTB(λ) loss:
 *        `Δ(i,j) = log F(s_i) + Σ log P_F − log F(s_j) − Σ log P_B` (summed over the edges
 *        spanned by `[i,j)`), weighted by `pair_weight_ratio = λ^{j-i} / Σ_{i'<j'} λ^{j'-i'}`
 *        (the pre-normalized share of the total weighted-average loss this specific pair
 *        contributes).
 *
 * @note Structurally identical math to `TrajectoryBalanceLoss`/`DetailedBalanceLoss` (`a+b-c-d`,
 *       squared), generalized to an arbitrary sub-trajectory span with an explicit weighting
 *       factor -- `TrajectoryBalanceLoss` is the `(i=0, j=n+1, ratio=1)` special case;
 *       `DetailedBalanceLoss` is any adjacent-pair `(i, i+1, ratio=1)` special case. See
 *       `mission_subtb_loss.md`'s Design section.
 * @note `pair_weight_ratio` is computed by the caller (from `λ`, `i`, `j`, and the total
 *       pair-weight sum `W` for the trajectory), not by this class -- `W` depends on every
 *       pair in the trajectory, information this single-pair class has no reason to own.
 * @note Not a Module subclass, for `MSELoss`'s own reason.
 */
class SubTBLoss {
public:
    SubTBLoss() = default;

    /**
     * @brief Computes this pair's `Δ(i,j)` and its weighted contribution to the total SubTB
     *        loss, `pair_weight_ratio · Δ(i,j)²`.
     * @param log_flow_i `log F_θ(s_i)`.
     * @param sum_log_pf `Σ_{t=i}^{j-1} log P_F(s_{t+1}|s_t)` over the spanned edges.
     * @param log_flow_j `log F_θ(s_j)` for a non-terminal `s_j`, or `log R(x)` if `j` is the
     *        trajectory's terminal sink position.
     * @param sum_log_pb `Σ_{t=i}^{j-1} log P_B(s_t|s_{t+1})` over the spanned edges (the exit
     *        edge contributes `0.0` to this sum, per `DetailedBalanceLoss`'s own convention).
     * @param pair_weight_ratio `λ^{j-i} / W`, this pair's pre-normalized share of the total
     *        weighted-average loss. Must be in `[0, 1]` for the total loss across all pairs of
     *        a trajectory to sum to a true weighted average -- not validated here (a
     *        caller-internal invariant, not an external boundary).
     * @return This pair's weighted contribution to the total loss.
     */
    [[nodiscard]] float forward(float log_flow_i, float sum_log_pf, float log_flow_j, float sum_log_pb,
                                 float pair_weight_ratio);

    /** @brief `Δ(i,j)` (unweighted) from the most recent forward() call. */
    [[nodiscard]] float delta() const;

    /** @brief `pair_weight_ratio · 2Δ(i,j)`. */
    [[nodiscard]] float grad_log_flow_i() const;

    /** @brief `pair_weight_ratio · (-2Δ(i,j))`. Only meaningful when `s_j` is non-terminal (a
     *         real `F_θ` network call) -- callers must not apply this when `j` is the
     *         trajectory's terminal sink position (fixed `log R(x)`, no network to backprop
     *         into). */
    [[nodiscard]] float grad_log_flow_j() const;

    /**
     * @brief The weight `w = pair_weight_ratio · (-2Δ(i,j))`, applied *uniformly* to every
     *        edge `t` in `[i,j)`'s own `P_F` softmax/log gradient identity
     *        (`w · (p[k] − 1{k==a_t})`) -- one `Δ(i,j)` affects every decision point the
     *        sub-trajectory spans equally, via the chain rule through `sum_log_pf`.
     */
    [[nodiscard]] float grad_weight_for_log_pf_range() const;

private:
    float last_delta_ = 0.0f;
    float last_pair_weight_ratio_ = 0.0f;
    bool has_forwarded_ = false;

    void require_forwarded() const;
};

}  // namespace pulsatrix
