#include "exai/dropout_module.hpp"

#include <stdexcept>

#include "exai/assert.hpp"

namespace exai {

DropoutModule::DropoutModule(float p, DeviceBackend* backend, uint64_t seed)
    : p_(p), scale_(p < 1.0f ? 1.0f / (1.0f - p) : 1.0f), backend_(backend), rng_(seed) {
    // External boundary (construction arguments can originate from Phase 5's Python
    // bindings with no upstream validation). p == 1 would make scale diverge.
    if (p < 0.0f || p >= 1.0f) {
        throw std::invalid_argument("DropoutModule: p must be in [0, 1)");
    }
}

Tensor DropoutModule::forward_impl(const Tensor& input) {
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    EXAI_ASSERT(input.device() == DeviceType::Cpu);

    const int64_t n = input.numel();
    last_shape_ = input.shape();
    last_mask_.assign(static_cast<size_t>(n), 1.0f);

    Tensor output(input.shape(), backend_, input.device());

    if (!is_training() || p_ == 0.0f) {
        // Eval mode or p==0: identity. mask stays all-ones (set above), matching what a
        // training-mode forward with p==0 would also produce -- backward()'s mask*scale
        // formula needs no special-casing for this branch.
        for (int64_t i = 0; i < n; ++i) {
            output.data()[i] = input.data()[i];
        }
        has_forwarded_ = true;
        return output;
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < n; ++i) {
        if (dist(rng_) < p_) {
            last_mask_[static_cast<size_t>(i)] = 0.0f;
            output.data()[i] = 0.0f;
        } else {
            output.data()[i] = input.data()[i] * scale_;
        }
    }

    has_forwarded_ = true;
    return output;
}

Tensor DropoutModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("DropoutModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_shape_) {
        throw std::invalid_argument("DropoutModule::backward: grad_output must match the cached forward shape");
    }
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    EXAI_ASSERT(grad_output.device() == DeviceType::Cpu);

    Tensor grad_input(last_shape_, backend_);
    for (int64_t i = 0; i < grad_output.numel(); ++i) {
        // mask is 0 or 1 -- when 0, the scale_ factor is irrelevant (product is still 0).
        grad_input.data()[i] = grad_output.data()[i] * last_mask_[static_cast<size_t>(i)] * scale_;
    }
    return grad_input;
}

Tensor DropoutModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    if (!has_forwarded_) {
        throw std::logic_error("DropoutModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.shape() != last_shape_) {
        throw std::invalid_argument(
            "DropoutModule::propagate_relevance: relevance_out must match the cached forward shape");
    }
    return Tensor(relevance_out);
}

}  // namespace exai
