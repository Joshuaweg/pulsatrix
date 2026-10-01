#include "pulsatrix/swiglu_module.hpp"

#include <stdexcept>
#include <vector>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {
namespace {

/**
 * @brief Flattens all leading dims of `shape` (last dim assumed == feature_dim) into a
 *        single batch dimension. Same technique MultiHeadAttentionModule's forward_impl
 *        Step 1 uses to feed rank-3 (N, L, d_model) input through a rank-2 LinearModule.
 */
[[nodiscard]] int64_t flatten_leading_dims(const Shape& shape, int64_t feature_dim) {
    PULSATRIX_ASSERT(shape.rank() >= 2);
    PULSATRIX_ASSERT(shape.dim(static_cast<size_t>(shape.rank() - 1)) == feature_dim);
    return shape.numel() / feature_dim;
}

/**
 * @brief A reshaped copy of t -- see multihead_attention_module.cpp's identical helper of
 *        the same name: Tensor::reshape is in-place/metadata-only on a contiguous row-major
 *        buffer, so this copy exists solely to keep the caller's tensor const.
 */
[[nodiscard]] Tensor reshaped(const Tensor& t, Shape new_shape) {
    Tensor out(t);
    out.reshape(std::move(new_shape));
    return out;
}

}  // namespace

SwiGLUModule::SwiGLUModule(int64_t d_model, int64_t d_ff, DeviceBackend* backend)
    : d_model_(d_model),
      d_ff_(d_ff),
      backend_(backend),
      gate_proj_(d_model, d_ff, backend),
      up_proj_(d_model, d_ff, backend),
      down_proj_(d_ff, d_model, backend),
      last_gate_pre_(Shape({0}), backend),
      last_gate_post_(Shape({0}), backend),
      last_up_(Shape({0}), backend) {
    // External boundary (construction arguments can originate from Phase 5's Python
    // bindings with no upstream validation).
    if (d_model <= 0) {
        throw std::invalid_argument("SwiGLUModule: d_model must be positive");
    }
    if (d_ff <= 0) {
        throw std::invalid_argument("SwiGLUModule: d_ff must be positive");
    }
}

Tensor SwiGLUModule::forward_impl(const Tensor& input) {
    // Device-generic: three LinearModule forwards plus DeviceBackend Silu and mul, no host
    // dereference (GPU-native-kernels Mission 0 O5). backward/propagate_relevance remain
    // host-only.
    if (input.rank() < 2 || input.shape().dim(static_cast<size_t>(input.rank() - 1)) != d_model_) {
        throw std::invalid_argument("SwiGLUModule::forward: input must be rank >= 2 with final dimension d_model");
    }

    last_input_shape_ = input.shape();
    const int64_t n_flat = flatten_leading_dims(last_input_shape_, d_model_);
    last_n_flat_ = n_flat;

    const Tensor flat_input = reshaped(input, Shape({n_flat, d_model_}));

    Tensor gate_pre = gate_proj_.forward(flat_input);  // (n_flat, d_ff)
    Tensor gate_post(gate_pre.shape(), backend_);
    backend_->elementwise(ElementwiseOp::Silu, gate_pre.data(), gate_post.data(),
                           static_cast<size_t>(gate_pre.numel()));

    Tensor up = up_proj_.forward(flat_input);  // (n_flat, d_ff)

    Tensor hidden(gate_post.shape(), backend_);
    backend_->mul(gate_post.data(), up.data(), hidden.data(), static_cast<size_t>(hidden.numel()));

    Tensor output_flat = down_proj_.forward(hidden);  // (n_flat, d_model)

    last_gate_pre_ = gate_pre;
    last_gate_post_ = gate_post;
    last_up_ = up;
    has_forwarded_ = true;

    return reshaped(output_flat, last_input_shape_);
}

Tensor SwiGLUModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("SwiGLUModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_input_shape_) {
        throw std::invalid_argument("SwiGLUModule::backward: grad_output must match the cached forward shape");
    }
    // Device-generic (GPU-native-kernels Mission 1): three LinearModule backwards, two muls
    // and the fused Silu derivative.
    const Tensor grad_output_flat = reshaped(grad_output, Shape({last_n_flat_, d_model_}));

    Tensor grad_hidden = down_proj_.backward(grad_output_flat);  // (n_flat, d_ff)

    const auto n = static_cast<size_t>(grad_hidden.numel());
    Tensor grad_gate_post(grad_hidden.shape(), backend_);
    backend_->mul(grad_hidden.data(), last_up_.data(), grad_gate_post.data(), n);
    Tensor grad_up(grad_hidden.shape(), backend_);
    backend_->mul(grad_hidden.data(), last_gate_post_.data(), grad_up.data(), n);

    // silu'(x) = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x)), from the cached pre-activation.
    Tensor grad_gate_pre(grad_hidden.shape(), backend_);
    backend_->elementwise_backward(ElementwiseOp::Silu, last_gate_pre_.data(), grad_gate_post.data(),
                                   grad_gate_pre.data(), n);

    Tensor grad_from_gate = gate_proj_.backward(grad_gate_pre);  // (n_flat, d_model)
    Tensor grad_from_up = up_proj_.backward(grad_up);            // (n_flat, d_model)

    Tensor grad_input_flat(grad_from_gate.shape(), backend_);
    backend_->add(grad_from_gate.data(), grad_from_up.data(), grad_input_flat.data(),
                  static_cast<size_t>(grad_input_flat.numel()));

    return reshaped(grad_input_flat, last_input_shape_);
}

Tensor SwiGLUModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) {
        throw std::logic_error("SwiGLUModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.shape() != last_input_shape_) {
        throw std::invalid_argument(
            "SwiGLUModule::propagate_relevance: relevance_out must match the cached forward shape");
    }
    // Device-generic (GPU-native-kernels Mission 3).

    const Tensor relevance_out_flat = reshaped(relevance_out, Shape({last_n_flat_, d_model_}));

    Tensor r_hidden = down_proj_.propagate_relevance(relevance_out_flat, config);  // (n_flat, d_ff)

    // Eq. 15 for the gate*up product: both operands receive the same share.
    const auto n = static_cast<size_t>(r_hidden.numel());
    Tensor r_gate_post(r_hidden.shape(), backend_, r_hidden.device());
    backend_->lrp_bilinear_elementwise(last_gate_post_.data(), last_up_.data(), r_hidden.data(), r_gate_post.data(), n,
                                       config.epsilon);
    Tensor r_up(r_gate_post);
    const Tensor& r_gate_pre = r_gate_post;

    Tensor r_from_gate = gate_proj_.propagate_relevance(r_gate_pre, config);  // (n_flat, d_model)
    Tensor r_from_up = up_proj_.propagate_relevance(r_up, config);            // (n_flat, d_model)

    Tensor relevance_in_flat(r_from_gate.shape(), backend_);
    backend_->add(r_from_gate.data(), r_from_up.data(), relevance_in_flat.data(),
                  static_cast<size_t>(relevance_in_flat.numel()));

    return reshaped(relevance_in_flat, last_input_shape_);
}

std::vector<ParamRef> SwiGLUModule::parameters() {
    std::vector<ParamRef> params;
    for (auto* module : {&gate_proj_, &up_proj_, &down_proj_}) {
        auto module_params = module->parameters();
        params.insert(params.end(), module_params.begin(), module_params.end());
    }
    return params;
}

}  // namespace pulsatrix
