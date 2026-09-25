/** @file multihead_attention_module.hpp
 *  @brief Multi-head scaled dot-product attention -- this codebase's first Module composed
 *         out of other real Modules, plus AttnLRP's Eq. 15 bilinear relevance rule.
 */
#pragma once

#include <memory>
#include <vector>

#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/rms_norm_module.hpp"
#include "pulsatrix/rope_module.hpp"
#include "pulsatrix/softmax_module.hpp"

namespace pulsatrix {

/**
 * @brief `softmax(Q @ K^T / sqrt(head_dim)) @ V`, multi-head, with optional RoPE and
 *        optional QK-Norm. Shape `(N, L, d_model) -> (N, L, d_model)`.
 *
 * **Composition over inheritance.** Every sub-operation that already exists as a shipped,
 * tested Module in this codebase is held here as a real member object and driven through its
 * own `forward()`/`backward()`/`propagate_relevance()`:
 *   - `LinearModule q_proj_/k_proj_/v_proj_/out_proj_` (all `d_model -> d_model`),
 *   - `RoPEModule q_rope_/k_rope_` (only when `use_rope`),
 *   - `RMSNormModule q_norm_/k_norm_` (only when `use_qk_norm`, `head_dim`-sized gamma),
 *   - `SoftmaxModule softmax_`.
 * None of their math is reimplemented here. What *is* implemented here is only what has no
 * existing module: the two batched matmuls (`Q@K^T`, `Attn@V`), the `1/sqrt(head_dim)`
 * scale, and the head split/merge permutation -- with their own forward, backward and LRP
 * rule.
 *
 * @note **Two RoPE instances and two QK-Norm instances, not one shared each.** Both of those
 *       module types cache their own forward input/output for use by `backward()` and
 *       `propagate_relevance()`. A single shared instance applied to Q and then to K would
 *       leave only K's activations in the cache, so the subsequent Q backward/relevance pass
 *       would silently redistribute through K's numbers. Separate instances are the only
 *       correct choice given those modules' caching contract (and, for QK-Norm, it also
 *       matches the usual published formulation, which gives Q and K independent gammas).
 * @note **QK-Norm gamma is initialized to 1.0**, not to `RMSNormModule`'s own zero default.
 *       A zero gamma would make QK-Norm annihilate Q and K entirely, so `use_qk_norm=true`
 *       on a freshly constructed module would produce uniform attention regardless of the
 *       input -- a silently degenerate configuration rather than a neutral default. Every
 *       other parameter here keeps this codebase's zero-init convention.
 * @note **`use_rope=true` requires an even `head_dim`** -- `RoPEModule` rotates adjacent
 *       feature pairs and rejects an odd dimension. Checked here, before the sub-module is
 *       constructed, so the error message names the real cause.
 */
class MultiHeadAttentionModule : public Module {
public:
    /**
     * @brief Constructs a multi-head attention block with zero-initialized projections.
     * @param d_model Model/embedding width. Input and output are both `(N, L, d_model)`.
     * @param num_heads Number of attention heads; `head_dim = d_model / num_heads`.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @param use_rope Apply RoPE to Q and K after the projections (and after QK-Norm).
     * @param use_qk_norm Apply RMSNorm over each head's `head_dim` features of Q and K.
     * @throws std::invalid_argument if `d_model <= 0`, `num_heads <= 0`,
     *         `d_model % num_heads != 0`, or `use_rope` with an odd `head_dim` -- external
     *         boundary (constructor arguments can originate from Phase 5's Python bindings
     *         with no upstream validation), same convention as every other module.
     */
    MultiHeadAttentionModule(int64_t d_model, int64_t num_heads, DeviceBackend* backend, bool use_rope = true,
                             bool use_qk_norm = false);

    /**
     * @brief Gradient w.r.t. this module's input; sub-module parameter gradients accumulate
     *        inside those sub-modules (reachable through parameters()).
     * @param grad_output Gradient w.r.t. this module's output, shape `(N, L, d_model)`
     *        matching the cached forward shape.
     * @return Gradient w.r.t. this module's input, shape `(N, L, d_model)` -- the sum of the
     *         three paths (through Q, through K, through V) back into the shared input.
     * @note The only hand-written gradient math here is the two batched matmuls
     *       (`dA = dC @ B^T`, `dB = A^T @ dC`, per `(n, h)` slice) and the head
     *       split/merge inverses (pure data movement). Everything else is a real
     *       sub-module `backward()` call. Verified against central finite differences over
     *       the input *and* every sub-module parameter in
     *       tests/multihead_attention_module_test.cpp.
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if grad_output's shape differs from the cached forward shape.
     * @note Raw host loops; PULSATRIX_ASSERT(grad_output.device() == DeviceType::Cpu) guarded --
     *       see mission_host_loop_guards.md.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Attention per the charter's closed OpType set -- see OpType::Attention's own note. */
    [[nodiscard]] OpType op_type() const override { return OpType::Attention; }

    /**
     * @brief LRP relevance propagation, composed from the sub-modules' own rules plus
     *        AttnLRP's Eq. 15 bilinear ("uniform") rule for the two batched matmuls.
     *
     * For `O = A @ B` per `(n, h)` slice, Eq. 15 (Achtibat et al. 2024) splits each output's
     * relevance evenly between the two *operands* -- hence the factor 2 in the denominator,
     * which this codebase's additive epsilon rule (LinearModule/RNNModule/RoPEModule) does
     * **not** have:
     *   `R_A[i,j] += sum_k (A[i,j]*B[j,k] / (2*O[i,k] + eps*sign(O[i,k]))) * R_O[i,k]`
     *   `R_B[j,k] += sum_i (A[i,j]*B[j,k] / (2*O[i,k] + eps*sign(O[i,k]))) * R_O[i,k]`
     * Applied once to `context = Attn @ V` and once to `scores_raw = Q @ K^T` (there `B` is
     * `K^T`, so the resulting `R_B` is transposed back into K's layout).
     *
     * @param relevance_out Relevance at this module's output, `(N, L, d_model)`.
     * @param config Supplies the epsilon stabilizer for Eq. 15 and for the projections'
     *        epsilon rule.
     * @return Relevance at this module's input, `(N, L, d_model)` -- summed over the Q/K/V paths.
     * @note **This composed rule does not conserve relevance**, and is deliberately excluded
     *       from tests/lrp_conservation_test.cpp's AllModuleTypeCases() for the same reason
     *       SoftmaxModule is: the softmax step in the middle of the pipeline is AttnLRP
     *       Eq. 13, a first-order DTD approximation with a known residual "hidden bias term"
     *       (see softmax_module.hpp). Eq. 15 is itself only conserving in the sense that the
     *       two operands' shares sum back to `R_O` -- and only when the pipeline feeding it
     *       conserves. The composed block's actual measured gap is reported by
     *       tests/multihead_attention_module_test.cpp's dedicated measurement test rather
     *       than asserted away or forced into the shared tolerance.
     * @note The `1/sqrt(head_dim)` scale is a positive constant factor, under which the
     *       epsilon rule is exactly the identity (`x*c/(c*x) == 1`), so relevance at the
     *       scaled scores equals relevance at the raw product and Eq. 15 is applied against
     *       the cached *raw* `Q @ K^T` -- no separate scale step, and no scale-dependent
     *       relevance.
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if relevance_out's shape differs from the cached forward shape.
     * @note Raw host loops; PULSATRIX_ASSERT(relevance_out.device() == DeviceType::Cpu) guarded.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    /** @brief Every sub-module's parameters, flattened -- Q/K/V/O weights and biases, plus
     *         the two QK-Norm gammas when enabled. RoPE and softmax contribute none. */
    [[nodiscard]] std::vector<ParamRef> parameters() override;

    /** @brief Cascades to every sub-module, the same way SequentialModule does. */
    void set_training(bool training) override;

    [[nodiscard]] int64_t d_model() const { return d_model_; }
    [[nodiscard]] int64_t num_heads() const { return num_heads_; }
    [[nodiscard]] int64_t head_dim() const { return head_dim_; }
    [[nodiscard]] bool uses_rope() const { return use_rope_; }
    [[nodiscard]] bool uses_qk_norm() const { return use_qk_norm_; }

    /** @name Sub-module access -- weight initialization from tests/loaders, and inspection. */
    ///@{
    [[nodiscard]] LinearModule& q_proj() { return q_proj_; }
    [[nodiscard]] LinearModule& k_proj() { return k_proj_; }
    [[nodiscard]] LinearModule& v_proj() { return v_proj_; }
    [[nodiscard]] LinearModule& out_proj() { return out_proj_; }
    /** @brief Q's QK-Norm sub-module, or nullptr when use_qk_norm is false. */
    [[nodiscard]] RMSNormModule* q_norm() { return q_norm_.get(); }
    /** @brief K's QK-Norm sub-module, or nullptr when use_qk_norm is false. */
    [[nodiscard]] RMSNormModule* k_norm() { return k_norm_.get(); }
    ///@}

    /** @brief Cached attention weights of the last forward, `(N, num_heads, L, L)` -- the
     *         softmax output. Exposed because "what did each head attend to" is the single
     *         most-asked explainability question about this module. */
    [[nodiscard]] const Tensor& last_attention_weights() const { return last_attn_; }

protected:
    /**
     * @brief Runs the 9-step pipeline: project -> split heads -> (QK-Norm) -> (RoPE) ->
     *        scores -> softmax -> context -> merge heads -> output projection.
     * @param input `(N, L, d_model)`, Cpu-resident.
     * @return `(N, L, d_model)`.
     * @throws std::invalid_argument if input is not rank-3 with a final dimension of d_model.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t d_model_;
    int64_t num_heads_;
    int64_t head_dim_;
    bool use_rope_;
    bool use_qk_norm_;
    DeviceBackend* backend_;

    LinearModule q_proj_;
    LinearModule k_proj_;
    LinearModule v_proj_;
    LinearModule out_proj_;
    SoftmaxModule softmax_;
    // Held by pointer rather than std::optional purely so the "disabled" state costs nothing
    // and needs no move/copy of a Module subclass (Module declares a virtual destructor,
    // which suppresses implicit move construction -- std::optional's in-place paths would
    // work but the pointer is unambiguous).
    std::unique_ptr<RoPEModule> q_rope_;
    std::unique_ptr<RoPEModule> k_rope_;
    std::unique_ptr<RMSNormModule> q_norm_;
    std::unique_ptr<RMSNormModule> k_norm_;

    // Forward caches. Q/K/V are the post-QK-Norm, post-RoPE, head-split values -- exactly
    // the operands the two matmuls' backward and Eq. 15 rules need.
    int64_t last_N_ = 0;
    int64_t last_L_ = 0;
    Tensor last_q_;           ///< (N, num_heads, L, head_dim)
    Tensor last_k_;           ///< (N, num_heads, L, head_dim)
    Tensor last_v_;           ///< (N, num_heads, L, head_dim)
    Tensor last_scores_raw_;  ///< (N, num_heads, L, L) -- Q@K^T *before* the 1/sqrt scale
    Tensor last_attn_;        ///< (N, num_heads, L, L) -- softmax output
    Tensor last_context_;     ///< (N, num_heads, L, head_dim) -- Attn@V
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
