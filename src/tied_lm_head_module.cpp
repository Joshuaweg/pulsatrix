#include "pulsatrix/tied_lm_head_module.hpp"

#include <stdexcept>

#include "lrp_rules.hpp"

namespace pulsatrix {
namespace {

[[nodiscard]] Tensor reshaped(const Tensor& t, Shape new_shape) {
    Tensor out(t);
    out.reshape(std::move(new_shape));
    return out;
}

[[nodiscard]] ParamRef embedding_table(EmbeddingModule& embedding) {
    return embedding.named_parameters().front().ref;
}

}  // namespace

TiedLMHeadModule::TiedLMHeadModule(EmbeddingModule& embedding, DeviceBackend* backend)
    : backend_(backend),
      weight_(embedding_table(embedding).value),
      weight_grad_(embedding_table(embedding).grad),
      zero_bias_(Shape({weight_->shape().dim(0)}), backend, weight_->device()),
      last_input_(Shape({0}), backend),
      last_logits_(Shape({0}), backend) {}

Tensor TiedLMHeadModule::forward_impl(const Tensor& input) {
    const int64_t V = vocab_size();
    const int64_t D = d_model();
    if (input.rank() < 2 || input.shape().dim(static_cast<size_t>(input.rank() - 1)) != D) {
        throw std::invalid_argument("TiedLMHeadModule::forward: input must be rank >= 2 with shape (..., d_model)");
    }
    const int64_t M = input.numel() / D;
    std::vector<int64_t> out_dims;
    for (int64_t i = 0; i + 1 < input.rank(); ++i) {
        out_dims.push_back(input.shape().dim(static_cast<size_t>(i)));
    }
    out_dims.push_back(V);

    last_input_ = reshaped(input, Shape({M, D}));
    Tensor logits(Shape({M, V}), backend_, weight_->device());
    // E is stored (V, D); gemm_ex reads it transposed in place.
    backend_->gemm_ex(last_input_.data(), false, weight_->data(), true, logits.data(), static_cast<size_t>(M),
                      static_cast<size_t>(D), static_cast<size_t>(V), 0.0f);
    last_logits_ = logits;
    last_output_shape_ = Shape(out_dims);
    last_input_shape_ = input.shape();
    has_forwarded_ = true;
    return reshaped(logits, last_output_shape_);
}

Tensor TiedLMHeadModule::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "TiedLMHeadModule::backward");
    if (!has_forwarded_) {
        throw std::logic_error("TiedLMHeadModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_output_shape_) {
        throw std::invalid_argument("TiedLMHeadModule::backward: grad_output must match the cached output shape");
    }
    const auto m = static_cast<size_t>(last_input_.shape().dim(0));
    const auto v = static_cast<size_t>(vocab_size());
    const auto d = static_cast<size_t>(d_model());
    // grad_E (V, D) += grad^T (V, M) @ x (M, D), straight into the embedding's gradient.
    if (weight_->requires_grad()) {
        backend_->gemm_ex(grad_output.data(), true, last_input_.data(), false, weight_grad_->data(), v, m, d, 1.0f);
    }
    // grad_x (M, D) = grad (M, V) @ E (V, D).
    Tensor grad_input(Shape({static_cast<int64_t>(m), static_cast<int64_t>(d)}), backend_, weight_->device());
    backend_->gemm(grad_output.data(), weight_->data(), grad_input.data(), m, v, d);
    return reshaped(grad_input, last_input_shape_);
}

Tensor TiedLMHeadModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    require_device(relevance_out, *compute_device(), "TiedLMHeadModule::propagate_relevance");
    if (!has_forwarded_) {
        throw std::logic_error("TiedLMHeadModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.shape() != last_output_shape_) {
        throw std::invalid_argument(
            "TiedLMHeadModule::propagate_relevance: relevance_out must match the cached output shape");
    }
    lrp_rules::validate(config, "TiedLMHeadModule");

    const auto m = static_cast<size_t>(last_input_.shape().dim(0));
    const auto v = static_cast<size_t>(vocab_size());
    const auto d = static_cast<size_t>(d_model());
    DeviceBackend* be = backend_;
    lrp_rules::AffineOp op;
    op.backend = be;
    op.device = weight_->device();
    op.input_numel = m * d;
    op.output_numel = m * v;
    op.weight_numel = v * d;
    op.bias_numel = v;
    op.forward = [=](const float* x, const float* w, float* y) { be->gemm_ex(x, false, w, true, y, m, d, v, 0.0f); };
    op.backward = [=](const float* g, const float* w, float* gx) { be->gemm(g, w, gx, m, v, d); };
    op.add_bias = [=](const float* y, const float* b, float* o) { be->add_row_vector(y, b, o, m, v); };

    // With a zero bias, the pre-bias epsilon rule is the bias-in-denominator one.
    LRPRuleConfig rule = config;
    if (lrp_rules::is_legacy_epsilon(rule)) {
        rule.epsilon_bias_in_denominator = true;
    }
    Tensor relevance_in(Shape({static_cast<int64_t>(m), static_cast<int64_t>(d)}), backend_, weight_->device());
    lrp_rules::apply(op, last_input_.data(), weight_->data(), zero_bias_.data(), last_logits_.data(),
                     relevance_out.data(), relevance_in.data(), rule);
    return reshaped(relevance_in, last_input_shape_);
}

}  // namespace pulsatrix
