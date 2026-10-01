/** @file softmax_module.hpp
 *  @brief Rank-agnostic softmax over the last axis, with AttnLRP's Eq. 13 DTD relevance rule.
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Softmax over the tensor's last dimension, applied independently to every "row"
 *        (every fixed combination of all leading dimensions).
 * @note Rank-agnostic by design -- Phase 3's MultiHeadAttentionModule softmaxes attention
 *       scores of shape (N, num_heads, L, L) over the final L axis, so this module must not
 *       assume rank 1 the way CrossEntropyLoss's private inline softmax does.
 * @note No learnable parameters (parameters() returns empty, matching ReluModule).
 */
class SoftmaxModule : public Module {
public:
    /**
     * @brief Constructs a softmax module.
     * @param backend Backend to allocate through. Not owned; must outlive this module.
     * @note Takes only a backend -- the softmax axis is always the last one, so there is no
     *       further configuration (matches ReluModule's constructor shape).
     */
    explicit SoftmaxModule(DeviceBackend* backend);

    /**
     * @brief Computes the gradient w.r.t. this module's input.
     * @param grad_output Gradient w.r.t. this module's output. Must match the shape of the
     *        most recent forward() call's output.
     * @return Gradient w.r.t. this module's input, per row:
     *         grad_in[i] = s[i] * (grad_out[i] - sum_j(s[j] * grad_out[j])), the standard
     *         softmax Jacobian-vector product with s the cached forward output.
     * @note Must be called after forward() -- uses the output cached from that call.
     * @note Not yet backend-generic -- dereferences Tensor::data() directly in a raw host
     *       loop. PULSATRIX_REQUIRE_HOST(grad_output) guards against silent
     *       UB on a CUDA-backed Tensor; see mission_host_loop_guards.md. Do not remove this
     *       guard without actually retrofitting the method to route through DeviceBackend.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Activation per charter's closed OpType set (same category as ReluModule). */
    [[nodiscard]] OpType op_type() const override { return OpType::Activation; }

    /**
     * @brief AttnLRP's softmax relevance rule (Achtibat et al. 2024, Eq. 13 -- Deep Taylor
     *        Decomposition), applied per row:
     *        R_in[i] = x[i] * (R_out[i] - s[i] * sum_j(R_out[j])),
     *        with x the cached forward *input* and s the cached forward *output*.
     * @param relevance_out Relevance at this module's output. Must match forward()'s shape.
     * @param config Unused -- Eq. 13 takes no epsilon/gamma parameter. Deliberately NOT
     *        epsilon-stabilized (see the conservation note below).
     * @return Relevance at this module's input.
     * @note **This rule does not conserve relevance** -- sum(R_in) != sum(R_out) in general,
     *       unlike every other propagate_relevance in this codebase. Eq. 13 is a first-order
     *       Taylor/DTD approximation around a nonzero reference point and the paper's own
     *       text describes the residual "hidden bias term". Do NOT add a stabilizer or
     *       rescale to force conservation: that would silently diverge from the cited
     *       formula. This module is intentionally excluded from
     *       tests/lrp_conservation_test.cpp's AllModuleTypeCases(); its conservation gap is
     *       measured and reported in tests/softmax_module_test.cpp instead.
     * @see cpp_engineering.aDNA's what/context/cpp_tdd/context_tdd_lrp_rule_pattern_taxonomy.md,
     *      "Known-Non-Conserving-by-Design Note" -- this is the cited exception to rule
     *      shape 2 (bilinear split), not a fifth rule shape.
     * @note Must be called after forward() -- uses both the input and the output cached from
     *       that call (the input is needed by the x[i] factor, which is why this module
     *       caches both).
     * @note Raw host loop; PULSATRIX_REQUIRE_HOST(relevance_out) guarded.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

protected:
    /**
     * @brief Numerically stable softmax over the last axis, per row (subtract the row max
     *        before exponentiating -- same convention as CrossEntropyLoss::forward).
     * @param input Input tensor. Must be rank >= 1 and Cpu-resident (raw host loop).
     * @return Softmax probabilities, same shape as input; each row sums to 1.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    DeviceBackend* backend_;
    Tensor last_input_;   ///< Cached forward input x -- required by Eq. 13's x[i] factor.
    Tensor last_output_;  ///< Cached forward output s -- required by backward() and Eq. 13.
};

}  // namespace pulsatrix
