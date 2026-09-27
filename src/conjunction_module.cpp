#include "pulsatrix/conjunction_module.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {
namespace {

/** @brief shape with a leading dimension of 1 prepended -- stack_operands()'s building block. */
Shape prepend_one(const Shape& s) {
    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(s.rank() + 1));
    dims.push_back(1);
    for (int64_t i = 0; i < s.rank(); ++i) {
        dims.push_back(s.dim(static_cast<size_t>(i)));
    }
    return Shape(dims);
}

/** @brief shape with the leading dimension dropped -- the inverse of prepend_one, applied
 *         to a stacked (leading dim 2) tensor to recover a single operand's shape. */
Shape drop_leading_dim(const Shape& s) {
    std::vector<int64_t> dims;
    for (int64_t i = 1; i < s.rank(); ++i) {
        dims.push_back(s.dim(static_cast<size_t>(i)));
    }
    return Shape(dims);
}

/**
 * @brief Splits a stacked (leading dim 2) tensor into its two operands.
 * @throws std::invalid_argument if stacked's rank is 0 or its leading dimension isn't 2.
 */
std::pair<Tensor, Tensor> split_operands(const Tensor& stacked, DeviceBackend* backend, const char* caller) {
    if (stacked.rank() < 1 || stacked.shape().dim(0) != 2) {
        throw std::invalid_argument(
            std::string(caller) + ": input must be a Stack of exactly two operand tensors (leading dim == 2)");
    }
    const Shape operand_shape = drop_leading_dim(stacked.shape());
    const int64_t half = operand_shape.numel();

    Tensor a(operand_shape, backend, stacked.device());
    Tensor b(operand_shape, backend, stacked.device());
    for (int64_t i = 0; i < half; ++i) {
        a.data()[i] = stacked.data()[i];
        b.data()[i] = stacked.data()[half + i];
    }
    return {std::move(a), std::move(b)};
}

/** @brief Combines two same-shape operand tensors into a stacked (leading dim 2) tensor. */
Tensor combine_operands(const Tensor& a, const Tensor& b, DeviceBackend* backend) {
    std::vector<int64_t> dims;
    dims.push_back(2);
    for (int64_t i = 0; i < a.rank(); ++i) {
        dims.push_back(a.shape().dim(static_cast<size_t>(i)));
    }
    Tensor result(Shape(dims), backend, a.device());
    const int64_t half = a.numel();
    for (int64_t i = 0; i < half; ++i) {
        result.data()[i] = a.data()[i];
        result.data()[half + i] = b.data()[i];
    }
    return result;
}

}  // namespace

ConjunctionModule::ConjunctionModule(DeviceBackend* backend, TNorm t_norm)
    : backend_(backend), t_norm_(t_norm), last_input_(Shape({0}), backend), last_output_(Shape({0}), backend) {}

Tensor ConjunctionModule::stack_operands(const Tensor& a, const Tensor& b, DeviceBackend* backend) {
    Tensor a1(a);
    a1.reshape(prepend_one(a.shape()));
    Tensor b1(b);
    b1.reshape(prepend_one(b.shape()));
    return Tensor::Stack({a1, b1}, backend);
}

Tensor ConjunctionModule::forward(const Tensor& a, const Tensor& b) {
    Tensor stacked = stack_operands(a, b, backend_);
    return Module::forward(stacked);
}

Tensor ConjunctionModule::forward_impl(const Tensor& input) {
    // Raw host loop below -- not yet backend-generic. See mission_host_loop_guards.md.
    PULSATRIX_ASSERT(input.device() == DeviceType::Cpu);

    auto [a, b] = split_operands(input, backend_, "ConjunctionModule::forward");

    Tensor y(a.shape(), backend_, input.device());
    const int64_t n = a.numel();
    switch (t_norm_) {
        case TNorm::Product:
            for (int64_t i = 0; i < n; ++i) {
                y.data()[i] = a.data()[i] * b.data()[i];
            }
            break;
        case TNorm::Lukasiewicz:
            for (int64_t i = 0; i < n; ++i) {
                const float z = a.data()[i] + b.data()[i] - 1.0f;
                y.data()[i] = (z > 0.0f) ? z : 0.0f;
            }
            break;
        case TNorm::Godel:
            for (int64_t i = 0; i < n; ++i) {
                y.data()[i] = (a.data()[i] <= b.data()[i]) ? a.data()[i] : b.data()[i];
            }
            break;
    }

    last_input_ = input;
    last_output_ = y;
    has_forwarded_ = true;
    return y;
}

Tensor ConjunctionModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("ConjunctionModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_output_.shape()) {
        throw std::invalid_argument("ConjunctionModule::backward: grad_output must match the cached forward shape");
    }
    // Raw host loop -- see mission_host_loop_guards.md.
    PULSATRIX_ASSERT(grad_output.device() == DeviceType::Cpu);

    auto [a, b] = split_operands(last_input_, backend_, "ConjunctionModule::backward");

    Tensor grad_a(a.shape(), backend_);
    Tensor grad_b(b.shape(), backend_);
    const int64_t n = a.numel();
    switch (t_norm_) {
        case TNorm::Product:
            for (int64_t i = 0; i < n; ++i) {
                grad_a.data()[i] = b.data()[i] * grad_output.data()[i];
                grad_b.data()[i] = a.data()[i] * grad_output.data()[i];
            }
            break;
        case TNorm::Lukasiewicz:
            for (int64_t i = 0; i < n; ++i) {
                const float z = a.data()[i] + b.data()[i] - 1.0f;
                const float g = (z > 0.0f) ? grad_output.data()[i] : 0.0f;
                grad_a.data()[i] = g;
                grad_b.data()[i] = g;
            }
            break;
        case TNorm::Godel:
            for (int64_t i = 0; i < n; ++i) {
                const bool a_wins = a.data()[i] <= b.data()[i];  // tie -> a, ReluModule-style convention
                grad_a.data()[i] = a_wins ? grad_output.data()[i] : 0.0f;
                grad_b.data()[i] = a_wins ? 0.0f : grad_output.data()[i];
            }
            break;
    }

    return combine_operands(grad_a, grad_b, backend_);
}

Tensor ConjunctionModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) {
        throw std::logic_error("ConjunctionModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.shape() != last_output_.shape()) {
        throw std::invalid_argument(
            "ConjunctionModule::propagate_relevance: relevance_out must match the cached forward shape");
    }
    // Raw host loop -- see mission_host_loop_guards.md.
    PULSATRIX_ASSERT(relevance_out.device() == DeviceType::Cpu);

    auto [a, b] = split_operands(last_input_, backend_, "ConjunctionModule::propagate_relevance");

    Tensor r_a(a.shape(), backend_);
    Tensor r_b(b.shape(), backend_);
    const int64_t n = a.numel();
    switch (t_norm_) {
        case TNorm::Product:
            // AttnLRP Eq. 15-shaped bilinear split, elementwise (inner dimension 1) case --
            // see the header's propagate_relevance() doc comment for the full derivation.
            for (int64_t i = 0; i < n; ++i) {
                const float y = last_output_.data()[i];
                const float denom = 2.0f * y + config.epsilon * ((y >= 0.0f) ? 1.0f : -1.0f);
                const float contribution = (a.data()[i] * b.data()[i] / denom) * relevance_out.data()[i];
                r_a.data()[i] = contribution;
                r_b.data()[i] = contribution;
            }
            break;
        case TNorm::Lukasiewicz:
            // Active region: LinearModule-style bias-excluded epsilon rule on z = a+b-1.
            // Inactive region: both operands' local derivative is 0 (matches backward()),
            // so both receive 0 -- kept consistent rather than force-conserving a dead branch.
            for (int64_t i = 0; i < n; ++i) {
                const float z = a.data()[i] + b.data()[i] - 1.0f;
                if (z > 0.0f) {
                    const float denom = z + config.epsilon;  // z > 0 here, sign(z) == +1
                    r_a.data()[i] = (a.data()[i] / denom) * relevance_out.data()[i];
                    r_b.data()[i] = (b.data()[i] / denom) * relevance_out.data()[i];
                } else {
                    r_a.data()[i] = 0.0f;
                    r_b.data()[i] = 0.0f;
                }
            }
            break;
        case TNorm::Godel:
            // Exact conservation: the winning (smaller, tie -> a) operand is a pure identity
            // map (y == that operand), so it receives all of R_out; the other receives 0.
            for (int64_t i = 0; i < n; ++i) {
                const bool a_wins = a.data()[i] <= b.data()[i];
                r_a.data()[i] = a_wins ? relevance_out.data()[i] : 0.0f;
                r_b.data()[i] = a_wins ? 0.0f : relevance_out.data()[i];
            }
            break;
    }

    return combine_operands(r_a, r_b, backend_);
}

}  // namespace pulsatrix
