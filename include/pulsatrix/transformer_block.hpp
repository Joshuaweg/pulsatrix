/** @file transformer_block.hpp
 *  @brief Pre-LN transformer block -- Phase 3's literal exit-gate deliverable, third and
 *         last composition mission (RMSNorm x2, MultiHeadAttentionModule, SwiGLUModule,
 *         plus the resolved residual-split LRP rule).
 *  @ingroup dl_modules
 */
#pragma once

#include <optional>
#include "pulsatrix/module.hpp"
#include "pulsatrix/multihead_attention_module.hpp"
#include "pulsatrix/rms_norm_module.hpp"
#include "pulsatrix/swiglu_module.hpp"

namespace pulsatrix {

/**
 * @brief `y1 = x + MHA(RMSNorm(x))`, `y2 = y1 + SwiGLU(RMSNorm(y1))`. Shape
 *        `(N, L, d_model) -> (N, L, d_model)`.
 *
 * **Composition over inheritance**, third exercise of this principle after
 * `MultiHeadAttentionModule` and `SwiGLUModule`: `norm1_`/`norm2_`/`mha_`/`swiglu_` are real
 * sub-objects driven through their own `forward()`/`backward()`/`propagate_relevance()`. The
 * only hand-written operation is the residual add, using this codebase's own two-term
 * weighted-sum epsilon/z-rule (the same shape `LSTMModule`'s `c_t = f_t*c_{t-1}+i_t*g_t` and
 * `GRUModule`'s `h_t = (1-z_t)*h_{t-1}+z_t*n_t` already use, weight fixed at 1 instead of a
 * gate value) -- AttnLRP does not itself address residual connections; this is
 * mission_transformer_block.md's own resolution, not a new rule invented ad hoc.
 *
 * @note **`RMSNormModule`, not `LayerNormModule`**, for stylistic consistency with this
 *       block's other LLaMA-family choices (`SwiGLU`, `RoPE`, optional QK-Norm) -- not a
 *       technical requirement; both are AttnLRP Eq. 19 identity-pass-through and would work
 *       identically here.
 * @note `op_type()` reuses `OpType::Elementwise` -- the residual add is a plain elementwise
 *       binary op with fixed weight 1, the same category `RoPEModule` and `SwiGLUModule`'s
 *       gate multiply already reuse. Not `Composite` (this module owns real math, the
 *       residual split, disqualifying it by `MultiHeadAttentionModule`'s own precedent) and
 *       not a new category (unlike attention's genuinely novel cross-position mixing, a
 *       residual add is not architecturally novel).
 * @note **Conservation is dominated by `MultiHeadAttentionModule`'s own known large gap**
 *       (measured ~49.6% of its own output in mission_multihead_attention.md), propagated
 *       through unchanged by the two residual splits (which conserve near-exactly, same
 *       argument as `SwiGLUModule`'s diagonal split) and by `SwiGLUModule`'s own near-exact
 *       contribution. Measured and decomposed stage-by-stage in
 *       tests/transformer_block_test.cpp, not assumed.
 * @see cpp_engineering.aDNA's what/context/cpp_tdd/context_tdd_lrp_rule_pattern_taxonomy.md,
 *      "Known-Non-Conserving-by-Design Note" -- this block's gap is inherited from
 *      MultiHeadAttentionModule, not a new instance of the exception.
 */
class TransformerBlock : public Module {
public:
    /**
     * @brief Constructs a transformer block with zero-initialized sub-module parameters.
     * @param d_model Model/embedding width.
     * @param num_heads Number of attention heads; forwarded to `MultiHeadAttentionModule`.
     * @param d_ff SwiGLU hidden width; forwarded to `SwiGLUModule`.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @param use_rope Forwarded to `MultiHeadAttentionModule`.
     * @param use_qk_norm Forwarded to `MultiHeadAttentionModule`.
     * @throws std::invalid_argument propagated from `MultiHeadAttentionModule`'s or
     *         `SwiGLUModule`'s own constructors (`d_model <= 0`, `num_heads <= 0`,
     *         `d_model % num_heads != 0`, `d_ff <= 0`, odd `head_dim` with `use_rope`) --
     *         no redundant re-validation here.
     */
    TransformerBlock(int64_t d_model, int64_t num_heads, int64_t d_ff, DeviceBackend* backend, bool use_rope = true,
                      bool use_qk_norm = false);

    /**
     * @brief Gradient w.r.t. this module's input; sub-module parameter gradients accumulate
     *        inside norm1_/mha_/norm2_/swiglu_ (reachable through parameters()).
     * @param grad_output Gradient w.r.t. this module's output, shape `(N, L, d_model)`
     *        matching the cached forward shape.
     * @return Gradient w.r.t. this module's input, same shape.
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if grad_output's shape differs from the cached forward shape.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 2).
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Elementwise per this module's own op_type() note above. */
    [[nodiscard]] OpType op_type() const override { return OpType::Elementwise; }

    /**
     * @brief LRP relevance propagation: two residual epsilon/z-rule splits composed with
     *        norm1_'s/mha_'s/norm2_'s/swiglu_'s own propagate_relevance().
     * @param relevance_out Relevance at this module's output, matching the cached forward shape.
     * @param config Supplies the epsilon stabilizer for the residual splits and every
     *        composed sub-module rule.
     * @return Relevance at this module's input, same shape as relevance_out.
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if relevance_out's shape differs from the cached forward shape.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 3).
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    /** @brief norm1_'s, mha_'s, norm2_'s, and swiglu_'s parameters, flattened. */
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;

    /** @brief Cascades to every sub-module, the same way SequentialModule/MultiHeadAttentionModule do. */
    void set_training(bool training) override;

    [[nodiscard]] int64_t d_model() const { return mha_.d_model(); }

    /** @name Sub-module access -- weight initialization from tests/loaders, and inspection. */
    ///@{
    [[nodiscard]] RMSNormModule& norm1() { return norm1_; }
    [[nodiscard]] MultiHeadAttentionModule& mha() { return mha_; }
    [[nodiscard]] RMSNormModule& norm2() { return norm2_; }
    [[nodiscard]] SwiGLUModule& swiglu() { return swiglu_; }
    ///@}


    /** @brief Where this layer computes, so forward() rejects an input on another device (FND-8). */
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }

protected:
    /**
     * @brief Runs: norm1 -> attention -> residual add -> norm2 -> SwiGLU -> residual add.
     * @param input `(N, L, d_model)`, any device.
     * @return `(N, L, d_model)`.
     * @throws std::invalid_argument if input's rank < 2 or final dimension != d_model.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t d_model_;
    DeviceBackend* backend_;

    RMSNormModule norm1_;
    MultiHeadAttentionModule mha_;
    RMSNormModule norm2_;
    SwiGLUModule swiglu_;

    // Forward caches -- the four operands the two residual splits need.
    Shape last_input_shape_ = Shape({0});
    Tensor last_x_;         ///< Cached input, (N, L, d_model).
    Tensor last_attn_out_;  ///< mha_'s output, (N, L, d_model).
    Tensor last_y1_;        ///< x + attn_out, (N, L, d_model).
    Tensor last_ffn_out_;   ///< swiglu_'s output, (N, L, d_model).
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
