#include "exai/linear_module.hpp"

#include "exai/assert.hpp"

namespace exai {

namespace {
// Transposes a (rows x cols) row-major buffer into a (cols x rows) row-major buffer.
// Only needed for LinearModule::backward()'s grad_x step -- CPUBackend::gemm has no
// transpose flag, and this is the one place in this module that needs an actual W^T
// rather than a layout chosen to avoid needing one.
Tensor transpose(const Tensor& m, int64_t rows, int64_t cols, DeviceBackend* backend) {
    Tensor out(Shape({cols, rows}), backend);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            out.data()[c * rows + r] = m.data()[r * cols + c];
        }
    }
    return out;
}
}  // namespace

LinearModule::LinearModule(int64_t in_features, int64_t out_features, DeviceBackend* backend)
    : in_features_(in_features),
      out_features_(out_features),
      backend_(backend),
      weight_(Shape({in_features, out_features}), backend),
      bias_(Shape({out_features}), backend),
      weight_grad_(Shape({in_features, out_features}), backend),
      bias_grad_(Shape({out_features}), backend),
      last_input_(Shape({in_features}), backend),
      last_pre_bias_output_(Shape({out_features}), backend) {}

void LinearModule::set_weight(std::initializer_list<float> values) {
    weight_ = Tensor(weight_.shape(), backend_, values);
}

void LinearModule::set_bias(std::initializer_list<float> values) {
    bias_ = Tensor(bias_.shape(), backend_, values);
}

Tensor LinearModule::forward_impl(const Tensor& input) {
    last_input_ = input;

    Tensor pre_bias(Shape({out_features_}), backend_);
    backend_->gemm(input.data(), weight_.data(), pre_bias.data(), 1, static_cast<size_t>(in_features_),
                    static_cast<size_t>(out_features_));
    last_pre_bias_output_ = pre_bias;  // cached for propagate_relevance's z_j

    Tensor output(pre_bias);
    output.accumulate(bias_);
    return output;
}

Tensor LinearModule::backward(const Tensor& grad_output) {
    // Dereferences Tensor::data() directly (via transpose()) -- not yet backend-generic.
    // See campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision and
    // mission_host_loop_guards.md.
    EXAI_ASSERT(grad_output.device() == DeviceType::Cpu);

    // grad_W = outer(last_input_, grad_output) = (in_features x 1) * (1 x out_features)
    Tensor local_weight_grad(Shape({in_features_, out_features_}), backend_);
    backend_->gemm(last_input_.data(), grad_output.data(), local_weight_grad.data(),
                    static_cast<size_t>(in_features_), 1, static_cast<size_t>(out_features_));
    weight_grad_.accumulate(local_weight_grad);

    bias_grad_.accumulate(grad_output);

    // grad_x = grad_output @ W^T = (1 x out_features) * (out_features x in_features)
    Tensor weight_t = transpose(weight_, in_features_, out_features_, backend_);
    Tensor grad_input(Shape({in_features_}), backend_);
    backend_->gemm(grad_output.data(), weight_t.data(), grad_input.data(), 1,
                    static_cast<size_t>(out_features_), static_cast<size_t>(in_features_));
    return grad_input;
}

Tensor LinearModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    // Dereferences Tensor::data() directly in a raw host loop -- not yet backend-generic.
    // See campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision and
    // mission_host_loop_guards.md.
    EXAI_ASSERT(relevance_out.device() == DeviceType::Cpu);

    Tensor relevance_in(Shape({in_features_}), backend_);
    relevance_in.fill(0.0f);

    for (int64_t j = 0; j < out_features_; ++j) {
        float z_j = last_pre_bias_output_.data()[j];
        float sign = (z_j >= 0.0f) ? 1.0f : -1.0f;
        float denom = z_j + config.epsilon * sign;
        float r_j = relevance_out.data()[j];

        for (int64_t i = 0; i < in_features_; ++i) {
            float w_ij = weight_.data()[i * out_features_ + j];
            float x_i = last_input_.data()[i];
            relevance_in.data()[i] += (x_i * w_ij / denom) * r_j;
        }
    }

    return relevance_in;
}

}  // namespace exai
