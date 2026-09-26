/** @file detailed_balance_loss.hpp
 *  @brief GFlowNet Detailed Balance loss (Bengio et al., "GFlowNet Foundations", arXiv:2111.09266).
 *  @ingroup mech_interp
 */
#pragma once

namespace pulsatrix {

/**
 * @brief `Δ(s,s') = log F(s) + log P_F(s'|s) − log F(s') − log P_B(s|s')`, `loss = Δ(s,s')²` --
 *        the per-*transition* credit-assignment alternative to `TrajectoryBalanceLoss`'s
 *        per-trajectory constraint.
 *
 * @note The terminal/exit transition is not a special case of this class: passing
 *       `log_f_s_next = log R(x)` (the reward, not a learned flow) and `log_pb = 0.0`
 *       (`log(1)`, the trivial backward policy from a unique virtual sink state) reduces the
 *       general formula to `Δ_exit = log F(s_n) + log P_F(stop|s_n) − log R(x)` exactly. See
 *       `mission_detailed_balance_loss.md`'s Design section.
 * @note Structurally identical math to `TrajectoryBalanceLoss` (`a + b − c − d`, squared) at a
 *       different granularity -- implemented as its own class rather than sharing a base,
 *       matching this codebase's own precedent (`MSELoss`/`CrossEntropyLoss`/
 *       `BCEWithLogitsLoss`/`KLDivergenceLoss` are all independent despite similar shapes).
 * @note Not a Module subclass, for `MSELoss`'s own reason.
 */
class DetailedBalanceLoss {
public:
    DetailedBalanceLoss() = default;

    /**
     * @brief Computes `Δ(s,s')` and the loss `Δ(s,s')²`, caching `Δ` for the accessors below.
     * @param log_flow_s `log F_θ(s)`.
     * @param log_pf `log P_F(s'|s)` (or `log P_F(stop|s)` for the exit transition).
     * @param log_flow_s_next `log F_θ(s')` for a non-terminal `s'`, or `log R(x)` for the
     *        exit transition (`s' = x`, the trajectory's terminal state).
     * @param log_pb `log P_B(s|s')` for a real transition, or `0.0` for the exit transition.
     * @return The scalar loss, `Δ(s,s')²`.
     */
    [[nodiscard]] float forward(float log_flow_s, float log_pf, float log_flow_s_next, float log_pb);

    /** @brief `Δ(s,s')` from the most recent forward() call. */
    [[nodiscard]] float delta() const;

    /** @brief `d(loss)/d(log F(s)) = 2Δ`. */
    [[nodiscard]] float grad_log_flow_s() const;

    /** @brief `d(loss)/d(log F(s')) = -2Δ`. Only meaningful when s' is non-terminal (a real
     *         `F_θ` network call) -- callers must not apply this to the exit transition's
     *         fixed `log R(x)` endpoint, which has no network to backprop into. */
    [[nodiscard]] float grad_log_flow_s_next() const;

    /**
     * @brief The weight `w = -2Δ` such that this transition's policy-network gradient is
     *        `w · (p[k] − 1{k==a})` -- `TrajectoryBalanceLoss`/`PolicyGradientLoss`'s exact
     *        softmax/log gradient identity shape.
     */
    [[nodiscard]] float grad_weight_for_log_pf() const;

private:
    float last_delta_ = 0.0f;
    bool has_forwarded_ = false;

    void require_forwarded() const;
};

}  // namespace pulsatrix
