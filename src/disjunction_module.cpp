#include "pulsatrix/disjunction_module.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {
namespace {

Shape prepend_one(const Shape& s) {
    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(s.rank() + 1));
    dims.push_back(1);
    for (int64_t i = 0; i < s.rank(); ++i) {
        dims.push_back(s.dim(static_cast<size_t>(i)));
    }
    return Shape(dims);
}

Shape drop_leading_dim(const Shape& s) {
    std::vector<int64_t> dims;
    for (int64_t i = 1; i < s.rank(); ++i) {
        dims.push_back(s.dim(static_cast<size_t>(i)));
    }
    return Shape(dims);
}

std::pair<Tensor, Tensor> split_operands(const Tensor& stacked, DeviceBackend* backend, const char* caller) {
    if (stacked.rank() < 1 || stacked.shape().dim(0) != 2) {
        throw std::invalid_argument(
            std::string(caller) + ": input must be a Stack of exactly two operand tensors (leading dim == 2)");
    }
    // Dereferences Tensor::data() directly in a raw host loop -- not yet backend-generic
    // (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(stacked);
    const Shape operand_shape = drop_leading_dim(stacked.shape());
    const int64_t half = operand_shape.numel();

    Tensor a(operand_shape, backend, stacked.device());
    Tensor b(operand_shape, backend, stacked.device());
    PULSATRIX_REQUIRE_HOST(a);
    PULSATRIX_REQUIRE_HOST(b);
    for (int64_t i = 0; i < half; ++i) {
        a.data()[i] = stacked.data()[i];
        b.data()[i] = stacked.data()[half + i];
    }
    return {std::move(a), std::move(b)};
}

Tensor combine_operands(const Tensor& a, const Tensor& b, DeviceBackend* backend) {
    std::vector<int64_t> dims;
    dims.push_back(2);
    for (int64_t i = 0; i < a.rank(); ++i) {
        dims.push_back(a.shape().dim(static_cast<size_t>(i)));
    }
    // Dereferences Tensor::data() directly in a raw host loop -- not yet backend-generic
    // (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(a);
    PULSATRIX_REQUIRE_HOST(b);
    Tensor result(Shape(dims), backend, a.device());
    PULSATRIX_REQUIRE_HOST(result);
    const int64_t half = a.numel();
    for (int64_t i = 0; i < half; ++i) {
        result.data()[i] = a.data()[i];
        result.data()[half + i] = b.data()[i];
    }
    return result;
}

}  // namespace

DisjunctionModule::DisjunctionModule(DeviceBackend* backend, TConorm t_conorm)
    : backend_(backend), t_conorm_(t_conorm), last_input_(Shape({0}), backend), last_output_(Shape({0}), backend) {}

Tensor DisjunctionModule::stack_operands(const Tensor& a, const Tensor& b, DeviceBackend* backend) {
    Tensor a1(a);
    a1.reshape(prepend_one(a.shape()));
    Tensor b1(b);
    b1.reshape(prepend_one(b.shape()));
    return Tensor::Stack({a1, b1}, backend);
}

Tensor DisjunctionModule::forward(const Tensor& a, const Tensor& b) {
    Tensor stacked = stack_operands(a, b, backend_);
    return Module::forward(stacked);
}

Tensor DisjunctionModule::forward_impl(const Tensor& input) {
    // Raw host loop below -- not yet backend-generic. See mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(input);

    auto [a, b] = split_operands(input, backend_, "DisjunctionModule::forward");

    Tensor y(a.shape(), backend_, input.device());
    const int64_t n = a.numel();
    switch (t_conorm_) {
        case TConorm::Product:
            for (int64_t i = 0; i < n; ++i) {
                const float av = a.data()[i];
                const float bv = b.data()[i];
                y.data()[i] = av + bv - av * bv;
            }
            break;
        case TConorm::Lukasiewicz:
            for (int64_t i = 0; i < n; ++i) {
                const float z = a.data()[i] + b.data()[i];
                y.data()[i] = (z < 1.0f) ? z : 1.0f;
            }
            break;
        case TConorm::Godel:
            for (int64_t i = 0; i < n; ++i) {
                y.data()[i] = (a.data()[i] >= b.data()[i]) ? a.data()[i] : b.data()[i];
            }
            break;
    }

    last_input_ = input;
    last_output_ = y;
    has_forwarded_ = true;
    return y;
}

Tensor DisjunctionModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("DisjunctionModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_output_.shape()) {
        throw std::invalid_argument("DisjunctionModule::backward: grad_output must match the cached forward shape");
    }
    // Raw host loop -- see mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(grad_output);

    auto [a, b] = split_operands(last_input_, backend_, "DisjunctionModule::backward");

    Tensor grad_a(a.shape(), backend_);
    Tensor grad_b(b.shape(), backend_);
    // Allocated through backend_, so a GPU backend tags them Cuda/Hip -- the host writes below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(grad_a);
    PULSATRIX_REQUIRE_HOST(grad_b);
    const int64_t n = a.numel();
    switch (t_conorm_) {
        case TConorm::Product:
            for (int64_t i = 0; i < n; ++i) {
                grad_a.data()[i] = (1.0f - b.data()[i]) * grad_output.data()[i];
                grad_b.data()[i] = (1.0f - a.data()[i]) * grad_output.data()[i];
            }
            break;
        case TConorm::Lukasiewicz:
            for (int64_t i = 0; i < n; ++i) {
                const float z = a.data()[i] + b.data()[i];
                const float g = (z < 1.0f) ? grad_output.data()[i] : 0.0f;
                grad_a.data()[i] = g;
                grad_b.data()[i] = g;
            }
            break;
        case TConorm::Godel:
            for (int64_t i = 0; i < n; ++i) {
                const bool a_wins = a.data()[i] >= b.data()[i];  // tie -> a
                grad_a.data()[i] = a_wins ? grad_output.data()[i] : 0.0f;
                grad_b.data()[i] = a_wins ? 0.0f : grad_output.data()[i];
            }
            break;
    }

    return combine_operands(grad_a, grad_b, backend_);
}

Tensor DisjunctionModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) {
        throw std::logic_error("DisjunctionModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.shape() != last_output_.shape()) {
        throw std::invalid_argument(
            "DisjunctionModule::propagate_relevance: relevance_out must match the cached forward shape");
    }
    // Raw host loop -- see mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(relevance_out);

    auto [a, b] = split_operands(last_input_, backend_, "DisjunctionModule::propagate_relevance");

    PULSATRIX_REQUIRE_HOST(last_output_);
    Tensor r_a(a.shape(), backend_);
    Tensor r_b(b.shape(), backend_);
    // Allocated through backend_, so a GPU backend tags them Cuda/Hip -- the host writes below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(r_a);
    PULSATRIX_REQUIRE_HOST(r_b);
    const int64_t n = a.numel();
    switch (t_conorm_) {
        case TConorm::Product:
            // Averaged dual-decomposition epsilon rule -- see the header's doc comment for
            // the full derivation (z_a + z_b == y exactly, for every a, b).
            for (int64_t i = 0; i < n; ++i) {
                const float av = a.data()[i];
                const float bv = b.data()[i];
                const float y = last_output_.data()[i];
                const float denom = y + config.epsilon * ((y >= 0.0f) ? 1.0f : -1.0f);
                const float z_a = av * (2.0f - bv) * 0.5f;
                const float z_b = bv * (2.0f - av) * 0.5f;
                r_a.data()[i] = (z_a / denom) * relevance_out.data()[i];
                r_b.data()[i] = (z_b / denom) * relevance_out.data()[i];
            }
            break;
        case TConorm::Lukasiewicz:
            for (int64_t i = 0; i < n; ++i) {
                const float av = a.data()[i];
                const float bv = b.data()[i];
                const float z = av + bv;
                if (z < 1.0f) {
                    const float denom = z + config.epsilon * ((z >= 0.0f) ? 1.0f : -1.0f);
                    r_a.data()[i] = (av / denom) * relevance_out.data()[i];
                    r_b.data()[i] = (bv / denom) * relevance_out.data()[i];
                } else {
                    r_a.data()[i] = 0.0f;
                    r_b.data()[i] = 0.0f;
                }
            }
            break;
        case TConorm::Godel:
            for (int64_t i = 0; i < n; ++i) {
                const bool a_wins = a.data()[i] >= b.data()[i];
                r_a.data()[i] = a_wins ? relevance_out.data()[i] : 0.0f;
                r_b.data()[i] = a_wins ? 0.0f : relevance_out.data()[i];
            }
            break;
    }

    return combine_operands(r_a, r_b, backend_);
}

}  // namespace pulsatrix
