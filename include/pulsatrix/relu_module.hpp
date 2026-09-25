/** @file relu_module.hpp
 *  @brief ReLU activation -- the second Module subclass, following LinearModule's pattern.
 */
#pragma once

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief y = max(x, 0), elementwise. No parameters, no parameter gradients.
 */
class ReluModule : public Module {
public:
    /**
     * @brief Constructs a ReLU module.
     * @param backend Backend to compute through. Not owned; must outlive this module.
     * @param device Which device last_input_ is initially tagged as. Defaults to Cpu.
     *        forward_impl()'s output is tagged with the actual input tensor's device on
     *        every call (not this constructor argument), since ReLU has no parameters of
     *        its own to anchor a fixed "module device" the way LinearModule's weight_ does
     *        -- see campaign_exai_dl_library_phase1_5_cuda_backend.md's Mission 3.
     */
    explicit ReluModule(DeviceBackend* backend, DeviceType device = DeviceType::Cpu);

    /**
     * @brief Computes the gradient w.r.t. this module's input.
     * @param grad_output Gradient w.r.t. this module's output. Must match the shape of
     *        the most recent forward() call's output.
     * @return Gradient w.r.t. this module's input: grad_output where input > 0, else 0.
     * @note x == 0 is treated as blocked (project convention -- ReLU's subgradient at 0
     *       is technically any value in [0,1]; this codebase picks 0, matching forward's
     *       own x > 0 threshold for max(x, 0)).
     * @note Must be called after forward() -- uses the input cached from that call.
     * @note Not yet backend-generic -- dereferences Tensor::data() directly in a raw host
     *       loop. PULSATRIX_ASSERT(grad_output.device() == DeviceType::Cpu) guards against
     *       silent UB on a CUDA-backed Tensor; see
     *       campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision. Do not
     *       remove this guard without actually retrofitting the method to route through
     *       DeviceBackend.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Activation per charter's closed OpType set. */
    [[nodiscard]] OpType op_type() const override { return OpType::Activation; }

    /**
     * @brief Pass-through LRP relevance propagation.
     * @note Not a placeholder -- this IS the correct, standard rule for a pointwise
     *       nonlinearity. LRP rules are defined across weighted (Linear/Conv) connections;
     *       activation functions pass relevance through unchanged (Montavon et al. 2019;
     *       cross-checked against xai_context.aDNA's technique_lrp.md, whose own LRP-0/eps
     *       formulation operates on post-nonlinearity activations without giving the
     *       nonlinearity itself a separate redistribution rule).
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    DeviceBackend* backend_;
    Tensor last_input_;
};

}  // namespace pulsatrix
