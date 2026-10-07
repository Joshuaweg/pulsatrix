/** @file feed_forward_module.hpp
 *  @brief The plain transformer MLP, `Linear -> activation -> Linear`, of BERT, ESM and the
 *         original Transformer (PLM-1). The gated variant is SwiGLUModule.
 *  @ingroup dl_modules
 */
#pragma once

#include <optional>

#include "pulsatrix/activation_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief `fc2(f(fc1(x)))` over the last dimension. Shape `(..., d_model) -> (..., d_model)`.
 * @note Parameters are named `fc1.*` and `fc2.*`. LRP: each Linear's own rule (the config's),
 *       relevance passed through the activation unchanged.
 */
class FeedForwardModule : public Module {
public:
    /**
     * @param activation ElementwiseOp::Gelu for BERT and ESM; any ActivationModule accepts.
     * @param bias Biases on both projections.
     * @throws std::invalid_argument if a width isn't positive, or for an op ActivationModule refuses.
     */
    FeedForwardModule(int64_t d_model, int64_t d_ff, DeviceBackend* backend, ElementwiseOp activation = ElementwiseOp::Gelu,
                      bool bias = true);

    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    /** @brief A chain of other modules. */
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    [[nodiscard]] std::vector<NamedBufferRef> named_buffers() override;
    void set_training(bool training) override;
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }

    [[nodiscard]] LinearModule& fc1() { return fc1_; }
    [[nodiscard]] ActivationModule& activation() { return act_; }
    [[nodiscard]] LinearModule& fc2() { return fc2_; }

protected:
    /** @throws std::invalid_argument unless the input is rank >= 2 with last dimension d_model. */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t d_model_;
    DeviceBackend* backend_;
    LinearModule fc1_;
    ActivationModule act_;
    LinearModule fc2_;
    Shape last_input_shape_ = Shape({0});
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
