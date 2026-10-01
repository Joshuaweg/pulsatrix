/** @file residual_module.hpp
 *  @brief Generic skip-connection wrapper -- Phase 4's only mission, the ResNet-style
 *         residual block generalized past a fixed Conv-BN-ReLU stack to any Module.
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief `y = x + inner->forward(x)` for an arbitrary already-built Module. The classic
 *        ResNet shortcut connection, owning its own native relevance-split rule -- the
 *        charter's explicitly named failure mode to avoid is Captum/Zennit's "Canonizer
 *        surgery" (an external post-hoc graph rewrite for residual connections).
 *
 * Not a new LRP rule: reuses this codebase's own two-term weighted-sum epsilon/z-rule
 * (LSTMModule's `c_t = f_t*c_{t-1}+i_t*g_t`, GRUModule's analogous carry-split,
 * TransformerBlock's own two residual adds), weight fixed at 1 -- resolved at Phase 3
 * activation, not re-derived here.
 *
 * @note `inner_` is held by non-owning pointer -- `"not owned; must outlive this object"`,
 *       the same convention as `SequentialModule`'s `layers_`. Any already-composed
 *       `Module` can be wrapped: a `SequentialModule` chaining `Conv2DModule`/
 *       `BatchNormModule`/`ReluModule` for a real ResNet basic block, a bare
 *       `Conv2DModule`, a `LinearModule`, even another `ResidualModule`.
 * @note `op_type()` reuses `OpType::Elementwise` -- this module's entire purpose is the
 *       residual add itself (fixed weight 1, computed independently per element), the
 *       same category `TransformerBlock`'s own residual adds already use. `inner_`'s own
 *       `op_type()` is unaffected.
 * @note `inner_->forward(x)` must return the same shape as `x` -- this module cannot
 *       validate that structurally in advance (it does not know `inner_`'s internals); a
 *       mismatch surfaces as `Tensor`'s own shape-mismatch error at the add or at the
 *       relevance split.
 */
class ResidualModule : public Module {
public:
    /**
     * @brief Constructs a residual wrapper around an existing Module.
     * @param inner The wrapped function F. Not owned; must outlive this object.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this
     *        module. Unlike inner, not null-checked -- every DeviceBackend* member
     *        elsewhere in this codebase follows the same "not owned, must outlive, not
     *        validated" convention (LinearModule, RoPEModule, etc. never null-check their
     *        own backend argument either); inner is validated because, unlike backend,
     *        this constructor's member-initializer-list Tensor construction doesn't
     *        dereference it before any body-level check could run.
     * @throws std::invalid_argument if inner is null -- external boundary, same convention
     *         as SequentialModule's null-entry check.
     */
    ResidualModule(Module* inner, DeviceBackend* backend);

    /**
     * @brief Gradient w.r.t. this module's input: both paths receive grad_output unchanged
     *        (real gradient of a plain sum), then inner_'s own backward() adds its
     *        contribution.
     * @param grad_output Gradient w.r.t. this module's output, matching the cached forward shape.
     * @return Gradient w.r.t. this module's input, same shape.
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if grad_output's shape differs from the cached forward shape.
     * @note Device-generic: inner backward plus DeviceBackend::add (GPU-native-kernels
     *       Mission 1); runs on a GPU tensor whenever inner_ does.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Elementwise per this module's own op_type() note above. */
    [[nodiscard]] OpType op_type() const override { return OpType::Elementwise; }

    /**
     * @brief LRP relevance propagation: the residual epsilon/z-rule split between x and
     *        inner_->forward(x), then inner_'s own propagate_relevance() for its share.
     * @param relevance_out Relevance at this module's output, matching the cached forward shape.
     * @param config Supplies the epsilon stabilizer for the residual split and inner_'s own rule.
     * @return Relevance at this module's input, same shape as relevance_out.
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if relevance_out's shape differs from the cached forward shape.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 3).
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    /** @brief inner_'s own parameters() -- this module owns none of its own. */
    [[nodiscard]] std::vector<ParamRef> parameters() override;

    /** @brief Cascades to inner_, the same way SequentialModule/MultiHeadAttentionModule do. */
    void set_training(bool training) override;

    [[nodiscard]] Module& inner() { return *inner_; }

protected:
    /**
     * @brief y = x + inner_->forward(x).
     * @param input Any shape inner_ accepts; inner_->forward(input) must return the same shape.
     * @return Same shape as input.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    Module* inner_;
    DeviceBackend* backend_;

    Tensor last_x_;    ///< Cached input.
    Tensor last_f_x_;  ///< inner_->forward(x)'s output.
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
