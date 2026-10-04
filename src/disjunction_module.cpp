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
    const Shape operand_shape = drop_leading_dim(stacked.shape());
    const int64_t half = operand_shape.numel();
    // Device-generic (GPU-native-kernels Mission 3): each operand is a contiguous half of the
    // stacked buffer, so the split is two same-device copies.
    Tensor a(operand_shape, backend, stacked.device());
    Tensor b(operand_shape, backend, stacked.device());
    const CopyDirection dir =
        stacked.device() == DeviceType::Cpu ? CopyDirection::HostToHost : CopyDirection::DeviceToDevice;
    const size_t bytes = static_cast<size_t>(half) * sizeof(float);
    backend->copy(a.data(), stacked.data(), bytes, dir);
    backend->copy(b.data(), stacked.data() + half, bytes, dir);
    return {std::move(a), std::move(b)};
}

Tensor combine_operands(const Tensor& a, const Tensor& b, DeviceBackend* backend) {
    std::vector<int64_t> dims;
    dims.push_back(2);
    for (int64_t i = 0; i < a.rank(); ++i) {
        dims.push_back(a.shape().dim(static_cast<size_t>(i)));
    }
    Tensor result(Shape(dims), backend, a.device());
    const int64_t half = a.numel();
    const CopyDirection dir = a.device() == DeviceType::Cpu ? CopyDirection::HostToHost : CopyDirection::DeviceToDevice;
    const size_t bytes = static_cast<size_t>(half) * sizeof(float);
    backend->copy(result.data(), a.data(), bytes, dir);
    backend->copy(result.data() + half, b.data(), bytes, dir);
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

    auto [a, b] = split_operands(input, backend_, "DisjunctionModule::forward");

    Tensor y(a.shape(), backend_, input.device());
    backend_->logic_pointwise(LogicOp::DisjunctionForward, static_cast<int>(t_conorm_), a.data(), b.data(), nullptr,
                              nullptr, y.data(), nullptr, static_cast<size_t>(a.numel()), 0.0f);
    last_input_ = input;
    last_output_ = y;
    has_forwarded_ = true;
    return y;
}

Tensor DisjunctionModule::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "DisjunctionModule::backward");
    if (!has_forwarded_) {
        throw std::logic_error("DisjunctionModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_output_.shape()) {
        throw std::invalid_argument("DisjunctionModule::backward: grad_output must match the cached forward shape");
    }

    auto [a, b] = split_operands(last_input_, backend_, "DisjunctionModule::backward");

    Tensor grad_a(a.shape(), backend_, a.device());
    Tensor grad_b(b.shape(), backend_, b.device());
    backend_->logic_pointwise(LogicOp::DisjunctionBackward, static_cast<int>(t_conorm_), a.data(), b.data(),
                              grad_output.data(), nullptr, grad_a.data(), grad_b.data(), static_cast<size_t>(a.numel()),
                              0.0f);
    return combine_operands(grad_a, grad_b, backend_);
}

Tensor DisjunctionModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    require_device(relevance_out, *compute_device(), "DisjunctionModule::propagate_relevance");
    if (!has_forwarded_) {
        throw std::logic_error("DisjunctionModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.shape() != last_output_.shape()) {
        throw std::invalid_argument(
            "DisjunctionModule::propagate_relevance: relevance_out must match the cached forward shape");
    }

    auto [a, b] = split_operands(last_input_, backend_, "DisjunctionModule::propagate_relevance");

    Tensor r_a(a.shape(), backend_, a.device());
    Tensor r_b(b.shape(), backend_, b.device());
    backend_->logic_pointwise(LogicOp::DisjunctionLrp, static_cast<int>(t_conorm_), a.data(), b.data(),
                              relevance_out.data(), last_output_.data(), r_a.data(), r_b.data(),
                              static_cast<size_t>(a.numel()), config.epsilon);
    return combine_operands(r_a, r_b, backend_);
}

}  // namespace pulsatrix
