#include "pulsatrix/aggregator_module.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {
namespace {

/** @brief shape with the leading dimension dropped -- same helper shape as
 *         conjunction_module.cpp's drop_leading_dim, duplicated per this codebase's own
 *         small-helper-per-file convention (see conjunction_module.cpp's identical note). */
Shape drop_leading_dim(const Shape& s) {
    std::vector<int64_t> dims;
    for (int64_t i = 1; i < s.rank(); ++i) {
        dims.push_back(s.dim(static_cast<size_t>(i)));
    }
    return Shape(dims);
}

}  // namespace

AggregatorModule::AggregatorModule(DeviceBackend* backend, float p)
    : backend_(backend),
      p_(p),
      last_input_(Shape({0}), backend),
      last_mean_(Shape({0}), backend),
      last_output_(Shape({0}), backend) {
    if (p == 0.0f) {
        throw std::invalid_argument("AggregatorModule: p must not be zero (pure power-mean undefined at p=0)");
    }
}

Tensor AggregatorModule::forward_impl(const Tensor& input) {

    if (input.rank() < 1) {
        throw std::invalid_argument("AggregatorModule::forward: input must have rank >= 1 (a leading batch axis)");
    }

    const int64_t n = input.shape().dim(0);
    const Shape output_shape = drop_leading_dim(input.shape());
    const int64_t cols = output_shape.numel();

    // Device-generic (GPU-native-kernels Mission 3): one thread per output column.
    Tensor mean_pow(output_shape, backend_, input.device());
    Tensor output(output_shape, backend_, input.device());
    backend_->aggregator_forward(input.data(), mean_pow.data(), output.data(), static_cast<size_t>(n),
                                 static_cast<size_t>(cols), p_);

    last_input_ = input;
    last_mean_ = mean_pow;
    last_output_ = output;
    has_forwarded_ = true;
    return output;
}

Tensor AggregatorModule::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "AggregatorModule::backward");
    if (!has_forwarded_) {
        throw std::logic_error("AggregatorModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_output_.shape()) {
        throw std::invalid_argument("AggregatorModule::backward: grad_output must match the cached forward shape");
    }

    const int64_t n = last_input_.shape().dim(0);
    const int64_t cols = last_output_.numel();
    Tensor grad_input(last_input_.shape(), backend_, last_input_.device());
    backend_->aggregator_backward(last_input_.data(), last_mean_.data(), grad_output.data(), grad_input.data(),
                                  static_cast<size_t>(n), static_cast<size_t>(cols), p_);
    return grad_input;
}

Tensor AggregatorModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    require_device(relevance_out, *compute_device(), "AggregatorModule::propagate_relevance");
    if (!has_forwarded_) {
        throw std::logic_error("AggregatorModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.shape() != last_output_.shape()) {
        throw std::invalid_argument(
            "AggregatorModule::propagate_relevance: relevance_out must match the cached forward shape");
    }

    const int64_t n = last_input_.shape().dim(0);
    const int64_t cols = last_output_.numel();

    Tensor relevance_in(last_input_.shape(), backend_, last_input_.device());
    backend_->aggregator_lrp(last_input_.data(), last_mean_.data(), relevance_out.data(), relevance_in.data(),
                             static_cast<size_t>(n), static_cast<size_t>(cols), p_, config.epsilon);
    return relevance_in;
}

}  // namespace pulsatrix
