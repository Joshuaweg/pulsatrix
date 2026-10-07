#include "pulsatrix/feed_forward_module.hpp"

#include <stdexcept>

namespace pulsatrix {
namespace {

Tensor Reshaped(const Tensor& t, Shape shape) {
    Tensor out(t);
    out.reshape(std::move(shape));
    return out;
}

int64_t CheckWidth(int64_t d, const char* what) {
    if (d <= 0) throw std::invalid_argument(std::string("FeedForwardModule: ") + what + " must be positive");
    return d;
}

}  // namespace

FeedForwardModule::FeedForwardModule(int64_t d_model, int64_t d_ff, DeviceBackend* backend, ElementwiseOp activation, bool bias)
    : d_model_(CheckWidth(d_model, "d_model")),
      backend_(backend),
      fc1_(d_model, CheckWidth(d_ff, "d_ff"), backend, bias),
      act_(activation, backend),
      fc2_(d_ff, d_model, backend, bias) {}

Tensor FeedForwardModule::forward_impl(const Tensor& input) {
    if (input.rank() < 2 || input.shape().dim(static_cast<size_t>(input.rank() - 1)) != d_model_) {
        throw std::invalid_argument("FeedForwardModule::forward: input must be rank >= 2 with final dimension d_model");
    }
    last_input_shape_ = input.shape();
    has_forwarded_ = true;
    const int64_t rows = input.numel() / d_model_;
    return Reshaped(fc2_.forward(act_.forward(fc1_.forward(Reshaped(input, Shape({rows, d_model_}))))), last_input_shape_);
}

Tensor FeedForwardModule::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "FeedForwardModule::backward");
    if (!has_forwarded_) throw std::logic_error("FeedForwardModule::backward: called before any forward()");
    if (grad_output.shape() != last_input_shape_) {
        throw std::invalid_argument("FeedForwardModule::backward: grad_output must match the cached forward shape");
    }
    const int64_t rows = grad_output.numel() / d_model_;
    return Reshaped(fc1_.backward(act_.backward(fc2_.backward(Reshaped(grad_output, Shape({rows, d_model_}))))), last_input_shape_);
}

Tensor FeedForwardModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    require_device(relevance_out, *compute_device(), "FeedForwardModule::propagate_relevance");
    if (!has_forwarded_) throw std::logic_error("FeedForwardModule::propagate_relevance: called before any forward()");
    if (relevance_out.shape() != last_input_shape_) {
        throw std::invalid_argument("FeedForwardModule::propagate_relevance: relevance_out must match the cached forward shape");
    }
    const int64_t rows = relevance_out.numel() / d_model_;
    Tensor r = fc2_.propagate_relevance(Reshaped(relevance_out, Shape({rows, d_model_})), config);
    r = act_.propagate_relevance(r, config);
    return Reshaped(fc1_.propagate_relevance(r, config), last_input_shape_);
}

std::vector<NamedParamRef> FeedForwardModule::named_parameters() {
    std::vector<NamedParamRef> out;
    append_named_parameters(out, "fc1", fc1_);
    append_named_parameters(out, "fc2", fc2_);
    return out;
}

std::vector<NamedBufferRef> FeedForwardModule::named_buffers() {
    std::vector<NamedBufferRef> out;
    append_named_buffers(out, "fc1", fc1_);
    append_named_buffers(out, "fc2", fc2_);
    return out;
}

void FeedForwardModule::set_training(bool training) {
    Module::set_training(training);
    fc1_.set_training(training);
    act_.set_training(training);
    fc2_.set_training(training);
}

}  // namespace pulsatrix
