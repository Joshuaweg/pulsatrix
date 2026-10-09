#include "pulsatrix/residual_module.hpp"

#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {
namespace {

/**
 * @brief Splits R (relevance at y=a+b, weight 1 each) between a and b -- same rule as
 *        transformer_block.cpp's identically-named helper (LSTMModule's cell-carry split,
 *        GRUModule's candidate split, weight fixed at 1). Duplicated per this codebase's
 *        established convention of not sharing these small anonymous-namespace helpers
 *        across files (see multihead_attention_module.cpp's/swiglu_module.cpp's own
 *        `reshaped()` duplication).
 */
void residual_split(const Tensor& a, const Tensor& b, const Tensor& r, float epsilon, Tensor& r_a, Tensor& r_b,
                    DeviceBackend* backend) {
    // Device-generic (GPU-native-kernels Mission 3).
    backend->lrp_residual_split(a.data(), b.data(), r.data(), r_a.data(), r_b.data(), static_cast<size_t>(r.numel()),
                                epsilon);
}

}  // namespace

ResidualModule::ResidualModule(Module* inner, DeviceBackend* backend)
    : inner_(inner), backend_(backend), last_x_(Shape({0}), backend), last_f_x_(Shape({0}), backend) {
    // External boundary (constructor arguments can originate from Phase 5's Python
    // bindings with no upstream validation), same convention as SequentialModule's
    // null-entry check. backend is not null-checked -- see the header's own note.
    if (inner == nullptr) {
        throw std::invalid_argument("ResidualModule: inner must not be null");
    }
}

ResidualModule::ResidualModule(Module* inner, Module* shortcut, DeviceBackend* backend) : ResidualModule(inner, backend) {
    if (shortcut == nullptr) throw std::invalid_argument("ResidualModule: shortcut must not be null");
    shortcut_ = shortcut;
}

Tensor ResidualModule::forward_impl(const Tensor& input) {
    // Device-generic: the inner forward plus a DeviceBackend::add, no host dereference. Runs
    // on a GPU tensor whenever inner_ does (GPU-native-kernels Mission 0 O5).
    Tensor f_x = inner_->forward(input);
    const Tensor s = shortcut_ != nullptr ? shortcut_->forward(input) : input;
    if (s.shape() != f_x.shape()) {
        throw std::invalid_argument("ResidualModule::forward: the shortcut's and inner's outputs must have the same shape");
    }

    Tensor y(f_x.shape(), backend_);
    backend_->add(s.data(), f_x.data(), y.data(), static_cast<size_t>(y.numel()));

    last_x_ = s;
    last_f_x_ = f_x;
    has_forwarded_ = true;

    return y;
}

Tensor ResidualModule::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "ResidualModule::backward");
    if (!has_forwarded_) {
        throw std::logic_error("ResidualModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_x_.shape()) {
        throw std::invalid_argument("ResidualModule::backward: grad_output must match the cached forward shape");
    }
    // Device-generic: inner backward plus DeviceBackend::add (GPU-native-kernels Mission 1).
    Tensor grad_from_inner = inner_->backward(grad_output);
    const Tensor grad_from_shortcut = shortcut_ != nullptr ? shortcut_->backward(grad_output) : grad_output;

    Tensor grad_x(grad_from_inner.shape(), backend_);
    backend_->add(grad_from_shortcut.data(), grad_from_inner.data(), grad_x.data(), static_cast<size_t>(grad_x.numel()));

    return grad_x;
}

Tensor ResidualModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    require_device(relevance_out, *compute_device(), "ResidualModule::propagate_relevance");
    if (!has_forwarded_) {
        throw std::logic_error("ResidualModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.shape() != last_x_.shape()) {
        throw std::invalid_argument(
            "ResidualModule::propagate_relevance: relevance_out must match the cached forward shape");
    }
    // Device-generic (GPU-native-kernels Mission 3).

    Tensor r_x_direct(relevance_out.shape(), backend_);
    Tensor r_f_x(relevance_out.shape(), backend_);
    residual_split(last_x_, last_f_x_, relevance_out, config.epsilon, r_x_direct, r_f_x, backend_);

    Tensor r_x_from_inner = inner_->propagate_relevance(r_f_x, config);
    if (shortcut_ != nullptr) r_x_direct = shortcut_->propagate_relevance(r_x_direct, config);

    Tensor r_x(r_x_from_inner.shape(), backend_);
    backend_->add(r_x_direct.data(), r_x_from_inner.data(), r_x.data(), static_cast<size_t>(r_x.numel()));

    return r_x;
}

std::vector<NamedBufferRef> ResidualModule::named_buffers() {
    std::vector<NamedBufferRef> result;
    append_named_buffers(result, "inner", *inner_);
    if (shortcut_ != nullptr) append_named_buffers(result, "shortcut", *shortcut_);
    return result;
}

std::vector<NamedParamRef> ResidualModule::named_parameters() {
    std::vector<NamedParamRef> params;
    append_named_parameters(params, "inner", *inner_);
    if (shortcut_ != nullptr) append_named_parameters(params, "shortcut", *shortcut_);
    return params;
}

void ResidualModule::set_training(bool training) {
    Module::set_training(training);
    inner_->set_training(training);
    if (shortcut_ != nullptr) shortcut_->set_training(training);
}

}  // namespace pulsatrix
