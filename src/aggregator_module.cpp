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
    // Raw host loop below -- not yet backend-generic. See mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(input);

    if (input.rank() < 1) {
        throw std::invalid_argument("AggregatorModule::forward: input must have rank >= 1 (a leading batch axis)");
    }

    const int64_t n = input.shape().dim(0);
    const Shape output_shape = drop_leading_dim(input.shape());
    const int64_t cols = output_shape.numel();

    Tensor mean_pow(output_shape, backend_, input.device());
    Tensor output(output_shape, backend_, input.device());
    for (int64_t j = 0; j < cols; ++j) {
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            sum += std::pow(input.data()[i * cols + j], p_);
        }
        const float m = sum / static_cast<float>(n);
        mean_pow.data()[j] = m;
        output.data()[j] = std::pow(m, 1.0f / p_);
    }

    last_input_ = input;
    last_mean_ = mean_pow;
    last_output_ = output;
    has_forwarded_ = true;
    return output;
}

Tensor AggregatorModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("AggregatorModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_output_.shape()) {
        throw std::invalid_argument("AggregatorModule::backward: grad_output must match the cached forward shape");
    }
    // Raw host loop -- see mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(grad_output);

    const int64_t n = last_input_.shape().dim(0);
    const int64_t cols = last_output_.numel();
    const float exponent_m = 1.0f / p_ - 1.0f;

    Tensor grad_input(last_input_.shape(), backend_);
    for (int64_t j = 0; j < cols; ++j) {
        const float m = last_mean_.data()[j];
        const float m_pow = std::pow(m, exponent_m);
        for (int64_t i = 0; i < n; ++i) {
            const float x = last_input_.data()[i * cols + j];
            const float x_pow = std::pow(x, p_ - 1.0f);
            grad_input.data()[i * cols + j] = grad_output.data()[j] * (m_pow * x_pow) / static_cast<float>(n);
        }
    }
    return grad_input;
}

Tensor AggregatorModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) {
        throw std::logic_error("AggregatorModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.shape() != last_output_.shape()) {
        throw std::invalid_argument(
            "AggregatorModule::propagate_relevance: relevance_out must match the cached forward shape");
    }
    // Raw host loop -- see mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(relevance_out);

    const int64_t n = last_input_.shape().dim(0);
    const int64_t cols = last_output_.numel();

    Tensor relevance_in(last_input_.shape(), backend_);
    for (int64_t j = 0; j < cols; ++j) {
        // sum_j(x_j^p) == mean(x^p) * n exactly, by m's own definition -- reuses the cached
        // mean rather than recomputing the sum of powers from scratch.
        const float denom_raw = last_mean_.data()[j] * static_cast<float>(n);
        const float denom = denom_raw + config.epsilon * ((denom_raw >= 0.0f) ? 1.0f : -1.0f);
        for (int64_t i = 0; i < n; ++i) {
            const float x = last_input_.data()[i * cols + j];
            const float contribution = std::pow(x, p_) / denom;
            relevance_in.data()[i * cols + j] = contribution * relevance_out.data()[j];
        }
    }
    return relevance_in;
}

}  // namespace pulsatrix
