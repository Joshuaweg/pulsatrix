#include "pulsatrix/batch_norm_fold.hpp"

#include <cmath>
#include <stdexcept>

namespace pulsatrix {
namespace {

CopyDirection to_host_direction(const Tensor& t) {
    return t.device() == DeviceType::Cpu ? CopyDirection::HostToHost : CopyDirection::DeviceToHost;
}

std::vector<float> to_host(const Tensor& t) {
    std::vector<float> out(static_cast<size_t>(t.numel()));
    t.backend()->copy(out.data(), t.data(), out.size() * sizeof(float), to_host_direction(t));
    return out;
}

// Replaces a tensor's values, keeping its shape, device and requires_grad flag.
void assign(Tensor& t, const std::vector<float>& values) {
    t = Tensor(t.shape(), t.backend(), values, t.device());
}

// Conv2D's kernel and bias, through named_parameters() rather than its setters, so the
// device and the requires_grad flags are kept.
struct ConvParams {
    Tensor* kernel;
    Tensor* bias;
};

ConvParams conv_params(Conv2DModule& conv) {
    std::vector<NamedParamRef> params = conv.named_parameters();
    return {params[0].ref.value, params[1].ref.value};
}

}  // namespace

BatchNormFold::BatchNormFold(Conv2DModule& conv, BatchNormModule& bn) : conv_(conv), bn_(bn) {
    if (bn.is_training()) {
        throw std::invalid_argument("BatchNormFold: BatchNorm must be in eval mode (set_training(false))");
    }
    ConvParams p = conv_params(conv);
    const int64_t channels = p.kernel->shape().dim(0);
    if (bn.num_channels_ != channels) {
        throw std::invalid_argument("BatchNormFold: BatchNorm channels must equal the Conv2D's output channels");
    }
    if (bn.folded_) {
        throw std::logic_error("BatchNormFold: this BatchNorm is already folded");
    }

    original_kernel_ = to_host(*p.kernel);
    original_bias_ = to_host(*p.bias);
    const std::vector<float> gamma = to_host(bn.gamma_), beta = to_host(bn.beta_);
    const std::vector<float> mean = to_host(bn.running_mean_), var = to_host(bn.running_var_);

    std::vector<float> kernel = original_kernel_, bias = original_bias_;
    const size_t per_channel = kernel.size() / static_cast<size_t>(channels);
    for (size_t c = 0; c < static_cast<size_t>(channels); ++c) {
        const float scale = gamma[c] / std::sqrt(var[c] + bn.eps_);
        for (size_t i = 0; i < per_channel; ++i) {
            kernel[c * per_channel + i] *= scale;
        }
        bias[c] = (bias[c] - mean[c]) * scale + beta[c];
    }
    assign(*p.kernel, kernel);
    assign(*p.bias, bias);
    bn.folded_ = true;
}

BatchNormFold::~BatchNormFold() {
    ConvParams p = conv_params(conv_);
    assign(*p.kernel, original_kernel_);
    assign(*p.bias, original_bias_);
    bn_.folded_ = false;
}

}  // namespace pulsatrix
