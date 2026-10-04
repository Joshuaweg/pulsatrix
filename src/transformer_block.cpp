#include "pulsatrix/transformer_block.hpp"

#include <stdexcept>
#include <vector>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {
namespace {

/** @brief Same helper as multihead_attention_module.cpp/swiglu_module.cpp of the same name. */
[[nodiscard]] int64_t flatten_leading_dims(const Shape& shape, int64_t feature_dim) {
    PULSATRIX_ASSERT(shape.rank() >= 2);
    PULSATRIX_ASSERT(shape.dim(static_cast<size_t>(shape.rank() - 1)) == feature_dim);
    return shape.numel() / feature_dim;
}

/** @brief A reshaped copy of t -- see multihead_attention_module.cpp's identical helper. */
[[nodiscard]] Tensor reshaped(const Tensor& t, Shape new_shape) {
    Tensor out(t);
    out.reshape(std::move(new_shape));
    return out;
}

/**
 * @brief Splits R (relevance at y=a+b, weight 1 each) between a and b via this codebase's
 *        established two-term weighted-sum epsilon/z-rule (LSTMModule's cell-carry split,
 *        GRUModule's candidate split -- here with both weights fixed at 1).
 */
void residual_split(const Tensor& a, const Tensor& b, const Tensor& r, float epsilon, Tensor& r_a, Tensor& r_b,
                    DeviceBackend* backend) {
    // Device-generic (GPU-native-kernels Mission 3).
    backend->lrp_residual_split(a.data(), b.data(), r.data(), r_a.data(), r_b.data(), static_cast<size_t>(r.numel()),
                                epsilon);
}

}  // namespace

TransformerBlock::TransformerBlock(int64_t d_model, int64_t num_heads, int64_t d_ff, DeviceBackend* backend,
                                    bool use_rope, bool use_qk_norm)
    : d_model_(d_model),
      backend_(backend),
      norm1_(d_model, backend),
      mha_(d_model, num_heads, backend, use_rope, use_qk_norm),
      norm2_(d_model, backend),
      swiglu_(d_model, d_ff, backend),
      last_x_(Shape({0}), backend),
      last_attn_out_(Shape({0}), backend),
      last_y1_(Shape({0}), backend),
      last_ffn_out_(Shape({0}), backend) {}

Tensor TransformerBlock::forward_impl(const Tensor& input) {
    // Device-generic (GPU-native-kernels Mission 2): norms, attention, SwiGLU and the two
    // residual adds all run through DeviceBackend.

    if (input.rank() < 2 || input.shape().dim(static_cast<size_t>(input.rank() - 1)) != d_model_) {
        throw std::invalid_argument(
            "TransformerBlock::forward: input must be rank >= 2 with final dimension d_model");
    }

    last_input_shape_ = input.shape();
    const int64_t n_flat = flatten_leading_dims(last_input_shape_, d_model_);

    Tensor norm1_out_flat = norm1_.forward(reshaped(input, Shape({n_flat, d_model_})));
    Tensor norm1_out = reshaped(norm1_out_flat, last_input_shape_);
    Tensor attn_out = mha_.forward(norm1_out);

    Tensor y1(input.shape(), backend_);
    backend_->add(input.data(), attn_out.data(), y1.data(), static_cast<size_t>(y1.numel()));

    Tensor norm2_out_flat = norm2_.forward(reshaped(y1, Shape({n_flat, d_model_})));
    Tensor norm2_out = reshaped(norm2_out_flat, last_input_shape_);
    Tensor ffn_out = swiglu_.forward(norm2_out);

    Tensor y2(input.shape(), backend_);
    backend_->add(y1.data(), ffn_out.data(), y2.data(), static_cast<size_t>(y2.numel()));

    last_x_ = input;
    last_attn_out_ = attn_out;
    last_y1_ = y1;
    last_ffn_out_ = ffn_out;
    has_forwarded_ = true;

    return y2;
}

Tensor TransformerBlock::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "TransformerBlock::backward");
    if (!has_forwarded_) {
        throw std::logic_error("TransformerBlock::backward: called before any forward()");
    }
    if (grad_output.shape() != last_input_shape_) {
        throw std::invalid_argument("TransformerBlock::backward: grad_output must match the cached forward shape");
    }

    const int64_t n_flat = flatten_leading_dims(last_input_shape_, d_model_);

    // y2 = y1 + ffn_out: both branches receive grad_output unchanged (real gradient of a
    // plain sum, not the LRP epsilon split -- that only applies to propagate_relevance).
    Tensor grad_ffn_out = grad_output;
    Tensor grad_norm2_out = swiglu_.backward(grad_ffn_out);
    Tensor grad_y1_from_norm2 =
        reshaped(norm2_.backward(reshaped(grad_norm2_out, Shape({n_flat, d_model_}))), last_input_shape_);

    Tensor grad_y1(grad_output.shape(), backend_);
    backend_->add(grad_output.data(), grad_y1_from_norm2.data(), grad_y1.data(),
                  static_cast<size_t>(grad_y1.numel()));

    // y1 = x + attn_out: same structure, one level up.
    Tensor grad_attn_out = grad_y1;
    Tensor grad_norm1_out = mha_.backward(grad_attn_out);
    Tensor grad_x_from_norm1 =
        reshaped(norm1_.backward(reshaped(grad_norm1_out, Shape({n_flat, d_model_}))), last_input_shape_);

    Tensor grad_x(grad_output.shape(), backend_);
    backend_->add(grad_y1.data(), grad_x_from_norm1.data(), grad_x.data(), static_cast<size_t>(grad_x.numel()));

    return grad_x;
}

Tensor TransformerBlock::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    require_device(relevance_out, *compute_device(), "TransformerBlock::propagate_relevance");
    if (!has_forwarded_) {
        throw std::logic_error("TransformerBlock::propagate_relevance: called before any forward()");
    }
    if (relevance_out.shape() != last_input_shape_) {
        throw std::invalid_argument(
            "TransformerBlock::propagate_relevance: relevance_out must match the cached forward shape");
    }
    // Device-generic (GPU-native-kernels Mission 3).

    const int64_t n_flat = flatten_leading_dims(last_input_shape_, d_model_);

    // y2 = y1 + ffn_out.
    Tensor r_y1_direct(relevance_out.shape(), backend_);
    Tensor r_ffn_out(relevance_out.shape(), backend_);
    residual_split(last_y1_, last_ffn_out_, relevance_out, config.epsilon, r_y1_direct, r_ffn_out, backend_);

    Tensor r_norm2_out = swiglu_.propagate_relevance(r_ffn_out, config);
    Tensor r_y1_from_norm2 = reshaped(
        norm2_.propagate_relevance(reshaped(r_norm2_out, Shape({n_flat, d_model_})), config), last_input_shape_);

    Tensor r_y1(relevance_out.shape(), backend_);
    backend_->add(r_y1_direct.data(), r_y1_from_norm2.data(), r_y1.data(), static_cast<size_t>(r_y1.numel()));

    // y1 = x + attn_out.
    Tensor r_x_direct(relevance_out.shape(), backend_);
    Tensor r_attn_out(relevance_out.shape(), backend_);
    residual_split(last_x_, last_attn_out_, r_y1, config.epsilon, r_x_direct, r_attn_out, backend_);

    Tensor r_norm1_out = mha_.propagate_relevance(r_attn_out, config);
    Tensor r_x_from_norm1 = reshaped(
        norm1_.propagate_relevance(reshaped(r_norm1_out, Shape({n_flat, d_model_})), config), last_input_shape_);

    Tensor r_x(relevance_out.shape(), backend_);
    backend_->add(r_x_direct.data(), r_x_from_norm1.data(), r_x.data(), static_cast<size_t>(r_x.numel()));

    return r_x;
}

std::vector<NamedParamRef> TransformerBlock::named_parameters() {
    std::vector<NamedParamRef> params;
    append_named_parameters(params, "norm1", norm1_);
    append_named_parameters(params, "mha", mha_);
    append_named_parameters(params, "norm2", norm2_);
    append_named_parameters(params, "swiglu", swiglu_);
    return params;
}

void TransformerBlock::set_training(bool training) {
    Module::set_training(training);
    norm1_.set_training(training);
    mha_.set_training(training);
    norm2_.set_training(training);
    swiglu_.set_training(training);
}

}  // namespace pulsatrix
