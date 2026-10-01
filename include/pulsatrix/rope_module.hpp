/** @file rope_module.hpp
 *  @brief Rotary Position Embedding -- fixed per-position pair rotation, epsilon-rule LRP.
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Rotary Position Embedding (RoPE, Su et al. 2021): a fixed, non-learnable,
 *        position-dependent rotation of each adjacent feature pair of a Q/K-shaped tensor.
 *
 * For a feature vector of even dimension `head_dim`, split into pairs `(x[2i], x[2i+1])`
 * for `i = 0 .. head_dim/2 - 1`. At sequence position `pos` the rotation angle is
 * `theta_i = pos * base^(-2i/head_dim)` (`base = 10000` by convention), and
 *   `y[2i]   = x[2i]*cos(theta_i) - x[2i+1]*sin(theta_i)`
 *   `y[2i+1] = x[2i]*sin(theta_i) + x[2i+1]*cos(theta_i)`.
 *
 * @note **Shape convention -- rank-agnostic over leading dims, last two axes are
 *       `(L, head_dim)`**: input shape `(..., L, head_dim)` is treated as
 *       `num_matrices = numel / (L * head_dim)` independent `(L, head_dim)` slices. The
 *       position index comes from the `L` axis (restarting at 0 in every slice) and the
 *       feature-pair index from the `head_dim` axis. This covers `(N, L, head_dim)`
 *       directly as well as a future MultiHeadAttentionModule's
 *       `(N, num_heads, L, head_dim)` with no change -- the same rank-agnostic reasoning
 *       SoftmaxModule applies to its last axis, one axis deeper.
 * @note No learnable parameters (`parameters()` returns empty, matching ReluModule /
 *       SoftmaxModule). Nothing is precomputed across calls either -- `cos`/`sin` are
 *       evaluated directly from `pos`/`i` inside the forward loop; this module has no
 *       state worth caching an angle table for.
 */
class RoPEModule : public Module {
public:
    /**
     * @brief Constructs a RoPE module.
     * @param head_dim Size of the final (feature) axis. Must be positive and even -- the
     *        rotation acts on adjacent pairs, so an odd dimension has no valid pairing.
     * @param backend Backend to allocate through. Not owned; must outlive this module.
     * @param base Frequency base of the geometric angle schedule; 10000.0 is the standard
     *        RoPE convention.
     * @throws std::invalid_argument if head_dim <= 0 or head_dim is odd -- external
     *         boundary (constructor arguments can originate from Phase 5's Python
     *         bindings with no upstream validation), same convention as every other
     *         module's constructor argument checks.
     */
    RoPEModule(int64_t head_dim, DeviceBackend* backend, float base = 10000.0f);

    /**
     * @brief Computes the gradient w.r.t. this module's input.
     * @param grad_output Gradient w.r.t. this module's output. Must match the shape of the
     *        most recent forward() call's input.
     * @return Gradient w.r.t. this module's input: the *inverse* rotation applied to
     *         grad_output (the rotation matrix is orthogonal, so `R^{-1} = R^T`, which is
     *         the forward formula with the `sin` terms' signs swapped):
     *         `grad_x[2i]   =  grad_y[2i]*cos(theta_i) + grad_y[2i+1]*sin(theta_i)`
     *         `grad_x[2i+1] = -grad_y[2i]*sin(theta_i) + grad_y[2i+1]*cos(theta_i)`.
     *         Verified against central finite differences in rope_module_test.cpp, not
     *         merely asserted from the orthogonality argument.
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if grad_output's shape differs from the cached
     *         forward shape.
     * @note Raw host loop; PULSATRIX_REQUIRE_HOST(grad_output) guarded --
     *       see mission_host_loop_guards.md. Do not remove without routing through
     *       DeviceBackend.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /**
     * @brief Elementwise per the charter's closed OpType set.
     * @note Justification for reusing `Elementwise` rather than adding an enum value (the
     *       enum grows only for genuinely new *operation categories*): RoPE is a fixed,
     *       non-learnable, input-shape-preserving transform whose output element depends
     *       only on its own feature pair at its own position -- a structured
     *       elementwise-*pair* map, strictly local in every axis. It is not `Linear` (no
     *       learnable weight matrix, no contraction across features), not `Activation`
     *       (not a pointwise nonlinearity), not `Normalization`/`Reduction` (no statistic
     *       is computed over any axis), and not `Embedding` (no table lookup -- the
     *       position enters as an angle, not an index into learned vectors). `Elementwise`
     *       is the closest real fit: a shape-preserving, weight-free, local map, which is
     *       exactly what a graph query for "which nodes are cheap local transforms" wants
     *       this node to answer.
     */
    [[nodiscard]] OpType op_type() const override { return OpType::Elementwise; }

    /**
     * @brief Standard weighted-connection epsilon-rule LRP for this fixed linear map.
     *
     * Each output component of a pair is a two-term weighted sum of the *same* input
     * pair's two components with fixed cos/sin coefficients, so the epsilon-rule mechanics
     * are identical to RNNModule's two-weighted-source case, just with `head_dim`-local
     * pairing instead of an across-timestep source. Per pair `i` at position `pos`:
     *   - `R_y[2i]` is split across `x[2i]` (weight `cos`) and `x[2i+1]` (weight `-sin`),
     *     proportional to their weighted contributions over `y[2i] + eps*sign(y[2i])`;
     *   - `R_y[2i+1]` is split across `x[2i]` (weight `sin`) and `x[2i+1]` (weight `cos`)
     *     over `y[2i+1] + eps*sign(y[2i+1])`.
     *
     * @param relevance_out Relevance at this module's output. Must match forward()'s shape.
     * @param config Supplies the epsilon stabilizer.
     * @return Relevance at this module's input.
     * @note **Both contributions are summed into each of `R_x[2i]`/`R_x[2i+1]`** -- every
     *       x-component receives relevance from *both* output components of its pair
     *       (structurally analogous to GRUModule's two-path accumulator, simpler because
     *       both paths land within one forward() call). The implementation zero-fills the
     *       result and uses `+=` for all four writes; an overwrite there would silently
     *       drop half the relevance and break conservation.
     * @note Unlike SoftmaxModule's AttnLRP Eq. 13, this rule *does* conserve relevance up
     *       to the epsilon stabilizer (the map is linear and bias-free, so each pair's two
     *       redistributions each sum to their own source relevance times
     *       `y/(y + eps*sign(y))`). Covered by tests/lrp_conservation_test.cpp's
     *       AllModuleTypeCases() entry and by a per-pair conservation test.
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if relevance_out's shape differs from the cached
     *         forward shape.
     * @note Raw host loop; PULSATRIX_REQUIRE_HOST(relevance_out) guarded.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

protected:
    /**
     * @brief Applies the per-position pair rotation.
     * @param input Input tensor of shape `(..., L, head_dim)`. Must be rank >= 2 with a
     *        final dimension equal to head_dim, and Cpu-resident (raw host loop).
     * @return The rotated tensor, same shape as input.
     * @throws std::invalid_argument if input.rank() < 2 or its last dimension != head_dim.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t head_dim_;
    float base_;
    DeviceBackend* backend_;
    Tensor last_input_;   ///< Cached forward input x -- the epsilon rule's numerators.
    Tensor last_output_;  ///< Cached forward output y -- the epsilon rule's denominators.
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
