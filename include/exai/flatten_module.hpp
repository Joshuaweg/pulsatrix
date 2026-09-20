/** @file flatten_module.hpp
 *  @brief Reshape-only Module -- flattens any-rank input to rank-1, for chaining
 *         Conv2DModule's rank-3 output into a LinearModule's rank-1 input (Phase 2
 *         Mission 3, the first classifier-head test network in this codebase).
 */
#pragma once

#include "exai/module.hpp"

namespace exai {

/**
 * @brief y = reshape(x, {x.numel()}). No parameters, no gradient math beyond reshaping.
 * @note op_type() returns OpType::Elementwise -- op_type.hpp's closed set has no
 *       dedicated Reshape tag, and Elementwise is the closest existing fit (identity over
 *       the same values, just re-viewed). A deliberate choice, not an ideal one; flagged
 *       here so a future reader doesn't wonder why a reshape module is tagged Elementwise.
 */
class FlattenModule : public Module {
public:
    /**
     * @brief Constructs a flatten module.
     * @param backend Backend to compute through. Not owned; must outlive this module.
     */
    explicit FlattenModule(DeviceBackend* backend) : backend_(backend), last_input_shape_(Shape({0})) {}

    /**
     * @brief Reshapes the gradient back to the shape forward() last saw.
     * @param grad_output Gradient w.r.t. this module's (flattened) output.
     * @return Gradient w.r.t. this module's (original-shape) input.
     * @note Must be called after forward() -- uses the shape cached from that call.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input(grad_output);
        grad_input.reshape(last_input_shape_);
        return grad_input;
    }

    [[nodiscard]] OpType op_type() const override { return OpType::Elementwise; }

    /**
     * @brief Pass-through LRP relevance propagation, reshaped back to the input's shape.
     * @note No weighted connections exist to redistribute relevance across -- reshaping
     *       is the only thing there is to undo, same rationale as ReluModule's pass-through
     *       rule for a parameterless pointwise op.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) override {
        Tensor relevance_in(relevance_out);
        relevance_in.reshape(last_input_shape_);
        return relevance_in;
    }

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override {
        last_input_shape_ = input.shape();
        Tensor output(input);
        output.reshape(Shape({input.numel()}));
        return output;
    }

private:
    DeviceBackend* backend_;
    Shape last_input_shape_;
};

}  // namespace exai
