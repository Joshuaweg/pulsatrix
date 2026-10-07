#include "pulsatrix/encoder_block.hpp"

#include <stdexcept>

#include "pulsatrix/feed_forward_module.hpp"
#include "pulsatrix/layer_norm_module.hpp"
#include "pulsatrix/rms_norm_module.hpp"

namespace pulsatrix {
namespace {

Tensor Reshaped(const Tensor& t, Shape shape) {
    Tensor out(t);
    out.reshape(std::move(shape));
    return out;
}

std::unique_ptr<Module> MakeNorm(const EncoderBlockOptions& o, int64_t d, DeviceBackend* backend) {
    const int64_t n = d > 0 ? d : 1;  // the attention's constructor reports a bad width
    if (o.norm == NormType::LayerNorm) return std::make_unique<LayerNormModule>(n, backend, backend->device(), o.norm_eps);
    return std::make_unique<RMSNormModule>(n, backend, backend->device(), o.norm_eps);
}

std::unique_ptr<Module> MakeMlp(const EncoderBlockOptions& o, int64_t d, int64_t d_ff, DeviceBackend* backend) {
    if (d_ff <= 0) throw std::invalid_argument("EncoderBlock: d_ff must be positive");
    if (o.mlp == MlpType::Plain) return std::make_unique<FeedForwardModule>(d, d_ff, backend, o.activation, o.mlp_bias);
    return std::make_unique<SwiGLUModule>(d, d_ff, backend, o.mlp_bias, o.gate_activation);
}

/** @brief The two-term epsilon rule for y = a + b, as TransformerBlock splits its residuals. */
void ResidualSplit(const Tensor& a, const Tensor& b, const Tensor& r, float epsilon, Tensor& r_a, Tensor& r_b, DeviceBackend* backend) {
    backend->lrp_residual_split(a.data(), b.data(), r.data(), r_a.data(), r_b.data(), static_cast<size_t>(r.numel()), epsilon);
}

Tensor Add(const Tensor& a, const Tensor& b, DeviceBackend* backend) {
    Tensor out(a.shape(), backend);
    backend->add(a.data(), b.data(), out.data(), static_cast<size_t>(out.numel()));
    return out;
}

}  // namespace

EncoderBlock::EncoderBlock(const AttentionConfig& attention, int64_t d_ff, DeviceBackend* backend, const EncoderBlockOptions& options)
    : d_model_(attention.d_model),
      backend_(backend),
      options_(options),
      norm1_(MakeNorm(options, attention.d_model, backend)),
      mha_(attention, backend),
      norm2_(MakeNorm(options, attention.d_model, backend)),
      mlp_(MakeMlp(options, attention.d_model, d_ff, backend)),
      last_x_(Shape({0}), backend),
      last_attn_out_(Shape({0}), backend),
      last_y1_(Shape({0}), backend),
      last_mlp_out_(Shape({0}), backend) {}

// Norm modules take (rows, d); the block's tensors are (N, L, d).
Tensor EncoderBlock::Norm(Module& norm, const Tensor& x) const {
    return Reshaped(norm.forward(Reshaped(x, Shape({x.numel() / d_model_, d_model_}))), x.shape());
}

Tensor EncoderBlock::NormBackward(Module& norm, const Tensor& g) const {
    return Reshaped(norm.backward(Reshaped(g, Shape({g.numel() / d_model_, d_model_}))), g.shape());
}

Tensor EncoderBlock::NormRelevance(Module& norm, const Tensor& r, const LRPRuleConfig& config) const {
    return Reshaped(norm.propagate_relevance(Reshaped(r, Shape({r.numel() / d_model_, d_model_})), config), r.shape());
}

Tensor EncoderBlock::forward_impl(const Tensor& input) {
    if (input.rank() != 3 || input.shape().dim(2) != d_model_) {
        throw std::invalid_argument("EncoderBlock::forward: input must be (N, L, d_model)");
    }
    last_input_shape_ = input.shape();
    const bool pre = options_.norm_position == NormPosition::Pre;
    last_x_ = input;
    last_attn_out_ = mha_.forward(pre ? Norm(*norm1_, input) : input);
    last_y1_ = pre ? Add(input, last_attn_out_, backend_) : Norm(*norm1_, Add(input, last_attn_out_, backend_));
    last_mlp_out_ = mlp_->forward(pre ? Norm(*norm2_, last_y1_) : last_y1_);
    has_forwarded_ = true;
    const Tensor sum = Add(last_y1_, last_mlp_out_, backend_);
    return pre ? sum : Norm(*norm2_, sum);
}

Tensor EncoderBlock::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "EncoderBlock::backward");
    if (!has_forwarded_) throw std::logic_error("EncoderBlock::backward: called before any forward()");
    if (grad_output.shape() != last_input_shape_) {
        throw std::invalid_argument("EncoderBlock::backward: grad_output must match the cached forward shape");
    }
    if (options_.norm_position == NormPosition::Pre) {
        const Tensor g_y1 = Add(grad_output, NormBackward(*norm2_, mlp_->backward(grad_output)), backend_);
        return Add(g_y1, NormBackward(*norm1_, mha_.backward(g_y1)), backend_);
    }
    const Tensor g_s2 = NormBackward(*norm2_, grad_output);
    const Tensor g_y1 = Add(g_s2, mlp_->backward(g_s2), backend_);
    const Tensor g_s1 = NormBackward(*norm1_, g_y1);
    return Add(g_s1, mha_.backward(g_s1), backend_);
}

Tensor EncoderBlock::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    require_device(relevance_out, *compute_device(), "EncoderBlock::propagate_relevance");
    if (!has_forwarded_) throw std::logic_error("EncoderBlock::propagate_relevance: called before any forward()");
    if (relevance_out.shape() != last_input_shape_) {
        throw std::invalid_argument("EncoderBlock::propagate_relevance: relevance_out must match the cached forward shape");
    }
    const bool pre = options_.norm_position == NormPosition::Pre;
    const Shape& shape = relevance_out.shape();
    // Second sublayer: y2 = y1 + mlp (pre), or y2 = norm2(y1 + mlp) (post).
    const Tensor r_s2 = pre ? relevance_out : NormRelevance(*norm2_, relevance_out, config);
    Tensor r_y1_direct(shape, backend_), r_mlp(shape, backend_);
    ResidualSplit(last_y1_, last_mlp_out_, r_s2, config.epsilon, r_y1_direct, r_mlp, backend_);
    Tensor r_mlp_in = mlp_->propagate_relevance(r_mlp, config);
    if (pre) r_mlp_in = NormRelevance(*norm2_, r_mlp_in, config);
    const Tensor r_y1 = Add(r_y1_direct, r_mlp_in, backend_);
    // First sublayer: y1 = x + attn (pre), or y1 = norm1(x + attn) (post).
    const Tensor r_s1 = pre ? r_y1 : NormRelevance(*norm1_, r_y1, config);
    Tensor r_x_direct(shape, backend_), r_attn(shape, backend_);
    ResidualSplit(last_x_, last_attn_out_, r_s1, config.epsilon, r_x_direct, r_attn, backend_);
    Tensor r_attn_in = mha_.propagate_relevance(r_attn, config);
    if (pre) r_attn_in = NormRelevance(*norm1_, r_attn_in, config);
    return Add(r_x_direct, r_attn_in, backend_);
}

std::vector<NamedParamRef> EncoderBlock::named_parameters() {
    std::vector<NamedParamRef> out;
    append_named_parameters(out, "norm1", *norm1_);
    append_named_parameters(out, "mha", mha_);
    append_named_parameters(out, "norm2", *norm2_);
    append_named_parameters(out, "mlp", *mlp_);
    return out;
}

std::vector<NamedBufferRef> EncoderBlock::named_buffers() {
    std::vector<NamedBufferRef> out;
    append_named_buffers(out, "norm1", *norm1_);
    append_named_buffers(out, "mha", mha_);
    append_named_buffers(out, "norm2", *norm2_);
    append_named_buffers(out, "mlp", *mlp_);
    return out;
}

void EncoderBlock::set_training(bool training) {
    Module::set_training(training);
    norm1_->set_training(training);
    mha_.set_training(training);
    norm2_->set_training(training);
    mlp_->set_training(training);
}

}  // namespace pulsatrix
