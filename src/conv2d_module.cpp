#include "pulsatrix/conv2d_module.hpp"

#include <stdexcept>

#include "lrp_rules.hpp"
#include "pulsatrix/assert.hpp"

namespace pulsatrix {

Conv2DModule::Conv2DModule(int64_t in_channels, int64_t out_channels, int64_t kernel_h, int64_t kernel_w,
                            DeviceBackend* backend)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_h_(kernel_h),
      kernel_w_(kernel_w),
      backend_(backend),
      kernel_(Shape({out_channels, in_channels, kernel_h, kernel_w}), backend),
      bias_(Shape({out_channels}), backend),
      kernel_grad_(Shape({out_channels, in_channels, kernel_h, kernel_w}), backend),
      bias_grad_(Shape({out_channels}), backend),
      last_input_(Shape({0}), backend),
      last_im2col_(Shape({0}), backend),
      last_pre_bias_output_(Shape({0}), backend) {}

void Conv2DModule::set_kernel(std::initializer_list<float> values) {
    kernel_ = Tensor(kernel_.shape(), backend_, values);
}

void Conv2DModule::set_bias(std::initializer_list<float> values) {
    bias_ = Tensor(bias_.shape(), backend_, values);
}

void Conv2DModule::set_kernel(const std::vector<float>& values) {
    kernel_ = Tensor(kernel_.shape(), backend_, values);
}

void Conv2DModule::set_bias(const std::vector<float>& values) {
    bias_ = Tensor(bias_.shape(), backend_, values);
}

Tensor Conv2DModule::forward_impl(const Tensor& input) {

    // External boundary (campaign_exai_dl_library_adversarial_hardening.md, Mission 1,
    // findings 1/6; shape generalized to (N, in_channels, H, W) by
    // campaign_exai_dl_library_batch_dimension_support): input can originate from Phase 5's
    // Python bindings with no upstream validation. Without this check, a kernel larger than
    // the input drives out_h/out_w negative -- which can silently produce a
    // wrong-but-valid-looking positive Q (e.g. out_h=-2, out_w=-3 -> Q=6) rather than erroring.
    if (input.rank() != 4 || input.shape().dim(1) != in_channels_) {
        throw std::invalid_argument("Conv2DModule::forward: input must be rank-4 (N, in_channels, H, W)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t H = input.shape().dim(2);
    const int64_t W = input.shape().dim(3);
    if (kernel_h_ > H || kernel_w_ > W) {
        throw std::invalid_argument("Conv2DModule::forward: kernel is larger than the input");
    }
    const int64_t out_h = H - kernel_h_ + 1;
    const int64_t out_w = W - kernel_w_ + 1;
    const int64_t P = in_channels_ * kernel_h_ * kernel_w_;
    const int64_t Q = out_h * out_w;
    const int64_t in_stride = in_channels_ * H * W;
    const int64_t out_stride = out_channels_ * Q;

    // Device-generic (GPU-native-kernels Mission 4): one im2col over the whole batch, a gemm
    // per example, then the per-channel bias broadcast.
    const DeviceType device = kernel_.device();
    const auto n = static_cast<size_t>(N), p = static_cast<size_t>(P), q = static_cast<size_t>(Q),
               oc = static_cast<size_t>(out_channels_);
    last_input_ = input;
    last_out_h_ = out_h;
    last_out_w_ = out_w;
    last_im2col_ = Tensor(Shape({N, P, Q}), backend_, device);
    backend_->im2col(input.data(), last_im2col_.data(), n, static_cast<size_t>(in_channels_), static_cast<size_t>(H),
                     static_cast<size_t>(W), static_cast<size_t>(kernel_h_), static_cast<size_t>(kernel_w_));
    last_pre_bias_output_ = Tensor(Shape({N, out_channels_, out_h, out_w}), backend_, device);
    for (int64_t e = 0; e < N; ++e) {
        backend_->gemm(kernel_.data(), last_im2col_.data() + e * P * Q, last_pre_bias_output_.data() + e * out_stride,
                       oc, p, q);
    }
    Tensor output(Shape({N, out_channels_, out_h, out_w}), backend_, device);
    backend_->add_channel_vector(last_pre_bias_output_.data(), bias_.data(), output.data(), n, oc, q);
    has_forwarded_ = true;
    return output;
}

Tensor Conv2DModule::backward(const Tensor& grad_output) {
    // Finding 12: calling backward() before any forward() previously silently computed a
    // meaningless answer from zero-initialized cached state instead of erroring.
    if (!has_forwarded_) {
        throw std::logic_error("Conv2DModule::backward: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    // Batch-size-mismatch is a new adversarial case introduced by the batch migration
    // (campaign_exai_dl_library_batch_dimension_support), same shape as LinearModule's.
    if (grad_output.rank() != 4 || grad_output.shape().dim(0) != N ||
        grad_output.shape().dim(1) != out_channels_ || grad_output.shape().dim(2) != last_out_h_ ||
        grad_output.shape().dim(3) != last_out_w_) {
        throw std::invalid_argument(
            "Conv2DModule::backward: grad_output must be rank-4 (N, out_channels, out_h, out_w) matching the "
            "cached forward shape");
    }

    const int64_t P = in_channels_ * kernel_h_ * kernel_w_;
    const int64_t Q = last_out_h_ * last_out_w_;
    const int64_t H = last_input_.shape().dim(2);
    const int64_t W = last_input_.shape().dim(3);
    const int64_t in_stride = in_channels_ * H * W;
    const int64_t out_stride = out_channels_ * Q;

    // Device-generic (GPU-native-kernels Mission 4). Per example, as before: the kernel and bias
    // gradients are computed into per-example temporaries and accumulated (the original
    // association); dY @ col^T and K^T @ dY read their transposed operand in place.
    const DeviceType device = kernel_.device();
    const auto p = static_cast<size_t>(P), q = static_cast<size_t>(Q), oc = static_cast<size_t>(out_channels_);
    Tensor ones(Shape({Q, 1}), backend_, device);
    ones.fill(1.0f);
    Tensor grad_cols(Shape({N, P, Q}), backend_, device);
    Tensor per_example_kernel_grad(kernel_.shape(), backend_, device);
    Tensor per_example_bias_grad(Shape({out_channels_}), backend_, device);
    for (int64_t e = 0; e < N; ++e) {
        const float* grad_out_e = grad_output.data() + e * out_stride;
        // Frozen parameters (FND-2) skip their gradient GEMMs; the input gradient below doesn't need them.
        if (kernel_.requires_grad()) {
            backend_->gemm_ex(grad_out_e, false, last_im2col_.data() + e * P * Q, true,
                              per_example_kernel_grad.data(), oc, q, p, 0.0f);
            kernel_grad_.accumulate(per_example_kernel_grad);
        }
        if (bias_.requires_grad()) {
            backend_->gemm_ex(grad_out_e, false, ones.data(), false, per_example_bias_grad.data(), oc, q, 1, 0.0f);
            bias_grad_.accumulate(per_example_bias_grad);
        }
        backend_->gemm_ex(kernel_.data(), true, grad_out_e, false, grad_cols.data() + e * P * Q, p, oc, q, 0.0f);
    }
    Tensor grad_input(last_input_.shape(), backend_, device);
    grad_input.fill(0.0f);
    backend_->col2im_add(grad_cols.data(), grad_input.data(), static_cast<size_t>(N),
                         static_cast<size_t>(in_channels_), static_cast<size_t>(H), static_cast<size_t>(W),
                         static_cast<size_t>(kernel_h_), static_cast<size_t>(kernel_w_));
    return grad_input;
}

Tensor Conv2DModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    // Finding 12: see backward()'s identical guard above.
    if (!has_forwarded_) {
        throw std::logic_error("Conv2DModule::propagate_relevance: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    if (relevance_out.rank() != 4 || relevance_out.shape().dim(0) != N ||
        relevance_out.shape().dim(1) != out_channels_ || relevance_out.shape().dim(2) != last_out_h_ ||
        relevance_out.shape().dim(3) != last_out_w_) {
        throw std::invalid_argument(
            "Conv2DModule::propagate_relevance: relevance_out must be rank-4 (N, out_channels, out_h, out_w) "
            "matching the cached forward shape");
    }


    const int64_t P = in_channels_ * kernel_h_ * kernel_w_;
    const int64_t Q = last_out_h_ * last_out_w_;
    const int64_t H = last_input_.shape().dim(2);
    const int64_t W = last_input_.shape().dim(3);
    const int64_t in_stride = in_channels_ * H * W;
    const int64_t out_stride = out_channels_ * Q;

    lrp_rules::validate(config, "Conv2DModule");

    // Device-generic (GPU-native-kernels Mission 4): the epsilon rule in patch space (one GPU
    // thread per patch element, output channels summed in the original order), then folded back.
    const DeviceType device = kernel_.device();
    Tensor relevance_cols(Shape({N, P, Q}), backend_, device);
    if (lrp_rules::is_legacy_epsilon(config)) {
        backend_->lrp_conv(last_im2col_.data(), kernel_.data(), last_pre_bias_output_.data(), relevance_out.data(),
                           relevance_cols.data(), static_cast<size_t>(N), static_cast<size_t>(out_channels_),
                           static_cast<size_t>(P), static_cast<size_t>(Q), config.epsilon);
    } else {
        // Zennit-compatible rules (LRP-rules Mission 2) in patch space: per example, the layer is
        // out_e = K (OC x P) @ col_e (P x Q) + b, so the patches are the rule's input operand.
        const auto n = static_cast<size_t>(N), p = static_cast<size_t>(P), q = static_cast<size_t>(Q),
                   oc = static_cast<size_t>(out_channels_);
        DeviceBackend* be = backend_;
        lrp_rules::AffineOp op;
        op.backend = be;
        op.device = device;
        op.input_numel = n * p * q;
        op.output_numel = n * oc * q;
        op.weight_numel = oc * p;
        op.bias_numel = oc;
        op.forward = [=](const float* col, const float* k, float* y) {
            for (size_t e = 0; e < n; ++e) {
                be->gemm(k, col + e * p * q, y + e * oc * q, oc, p, q);
            }
        };
        op.backward = [=](const float* g, const float* k, float* gcol) {
            for (size_t e = 0; e < n; ++e) {
                be->gemm_ex(k, true, g + e * oc * q, false, gcol + e * p * q, p, oc, q, 0.0f);
            }
        };
        op.add_bias = [=](const float* y, const float* b, float* o) { be->add_channel_vector(y, b, o, n, oc, q); };
        lrp_rules::apply(op, last_im2col_.data(), kernel_.data(), bias_.data(), last_pre_bias_output_.data(),
                         relevance_out.data(), relevance_cols.data(), config);
    }
    Tensor relevance_in(last_input_.shape(), backend_, device);
    relevance_in.fill(0.0f);
    backend_->col2im_add(relevance_cols.data(), relevance_in.data(), static_cast<size_t>(N),
                         static_cast<size_t>(in_channels_), static_cast<size_t>(H), static_cast<size_t>(W),
                         static_cast<size_t>(kernel_h_), static_cast<size_t>(kernel_w_));
    return relevance_in;
}

}  // namespace pulsatrix
