/** @file relu_module.hpp
 *  @brief ReLU activation -- the second Module subclass, following LinearModule's pattern.
 */
#pragma once

#include "exai/module.hpp"

namespace exai {

/**
 * @brief y = max(x, 0), elementwise. No parameters, no parameter gradients.
 */
class ReluModule : public Module {
public:
    /**
     * @brief Constructs a ReLU module.
     * @param backend Backend to compute through. Not owned; must outlive this module.
     */
    explicit ReluModule(DeviceBackend* backend);

    /**
     * @brief Computes the gradient w.r.t. this module's input.
     * @param grad_output Gradient w.r.t. this module's output. Must match the shape of
     *        the most recent forward() call's output.
     * @return Gradient w.r.t. this module's input: grad_output where input > 0, else 0.
     * @note x == 0 is treated as blocked (project convention -- ReLU's subgradient at 0
     *       is technically any value in [0,1]; this codebase picks 0, matching forward's
     *       own x > 0 threshold for max(x, 0)).
     * @note Must be called after forward() -- uses the input cached from that call.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output);

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

}  // namespace exai
