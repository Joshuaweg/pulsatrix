/** @file activation_module.hpp
 *  @brief A pointwise activation chosen at construction: ReLU, tanh, sigmoid, SiLU, exact GELU or
 *         GELU's tanh approximation (PLM-1).
 *  @ingroup dl_modules
 */
#pragma once

#include <optional>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief y = f(x), elementwise, for one of the backend's activations. No parameters.
 * @note LRP passes relevance through unchanged, the standard rule for a pointwise nonlinearity
 *       (Montavon et al. 2019) and the one AttnLRP and LXT apply to a transformer MLP's GELU.
 */
class ActivationModule : public Module {
public:
    /**
     * @param op ElementwiseOp::Relu, Tanh, Sigmoid, Silu, Gelu or GeluTanh.
     * @throws std::invalid_argument for ElementwiseOp::Neg or Exp, which aren't activations.
     */
    ActivationModule(ElementwiseOp op, DeviceBackend* backend);

    /** @brief grad_output times f'(x) at the cached input. @throws std::logic_error before any forward(). */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    /** @brief relevance_out, unchanged. */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    [[nodiscard]] bool supports_lrp_rule(LRPRule) const override { return true; }
    [[nodiscard]] OpType op_type() const override { return OpType::Activation; }
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }
    [[nodiscard]] ElementwiseOp op() const { return op_; }

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    ElementwiseOp op_;
    DeviceBackend* backend_;
    Tensor last_input_;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
