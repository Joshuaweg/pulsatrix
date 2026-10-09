#include "pulsatrix/conv2d_module.hpp"

#include <algorithm>
#include <stdexcept>

#include "lrp_rules.hpp"
#include "pulsatrix/assert.hpp"

namespace pulsatrix {

Conv2DModule::Conv2DModule(int64_t in_channels, int64_t out_channels, int64_t kernel_h, int64_t kernel_w,
                            DeviceBackend* backend, int64_t stride, int64_t padding)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_h_(kernel_h),
      kernel_w_(kernel_w),
      stride_(stride),
      padding_(padding),
      backend_(backend),
      kernel_(Shape({out_channels, in_channels, kernel_h, kernel_w}), backend),
      bias_(Shape({out_channels}), backend),
      kernel_grad_(Shape({out_channels, in_channels, kernel_h, kernel_w}), backend),
      bias_grad_(Shape({out_channels}), backend),
      last_input_(Shape({0}), backend),
      last_im2col_(Shape({0}), backend),
      last_pre_bias_output_(Shape({0}), backend) {
    if (stride < 1) {
        throw std::invalid_argument("Conv2DModule: stride must be >= 1");
    }
    if (padding < 0) {
        throw std::invalid_argument("Conv2DModule: padding must be >= 0");
    }
}

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
    if (kernel_h_ > H + 2 * padding_ || kernel_w_ > W + 2 * padding_) {
        throw std::invalid_argument("Conv2DModule::forward: kernel is larger than the padded input");
    }
    const int64_t out_h = (H + 2 * padding_ - kernel_h_) / stride_ + 1;
    const int64_t out_w = (W + 2 * padding_ - kernel_w_) / stride_ + 1;
    const int64_t P = in_channels_ * kernel_h_ * kernel_w_;
    const int64_t Q = out_h * out_w;
    const int64_t in_stride = in_channels_ * H * W;
    const int64_t out_stride = out_channels_ * Q;

    // Device-generic (GPU-native-kernels Mission 4): im2col, a gemm per example, then the
    // per-channel bias broadcast. HIP-7: the patches go through a buffer of at most
    // chunk_examples() examples, so their memory is bounded however large the batch is; when the
    // whole batch fits, they are kept for backward() and LRP instead of being rebuilt.
    const DeviceType device = kernel_.device();
    const auto n = static_cast<size_t>(N), p = static_cast<size_t>(P), q = static_cast<size_t>(Q),
               oc = static_cast<size_t>(out_channels_);
    last_input_ = input;
    last_out_h_ = out_h;
    last_out_w_ = out_w;
    const int64_t chunk = chunk_examples(N, P, Q);
    cached_cols_ = chunk == N;
    last_im2col_ = Tensor(Shape({chunk, P, Q}), backend_, device);
    last_pre_bias_output_ = Tensor(Shape({N, out_channels_, out_h, out_w}), backend_, device);
    for (int64_t e0 = 0; e0 < N; e0 += chunk) {
        const int64_t count = std::min(chunk, N - e0);
        backend_->im2col(input.data() + e0 * in_stride, last_im2col_.data(), static_cast<size_t>(count),
                         static_cast<size_t>(in_channels_), static_cast<size_t>(H), static_cast<size_t>(W), geometry());
        // HIP-6: the kernel against every example's patches in one batched GEMM.
        backend_->gemm_strided_batched(kernel_.data(), false, 0, last_im2col_.data(), false, p * q,
                                       last_pre_bias_output_.data() + e0 * out_stride, oc * q, oc, p, q,
                                       static_cast<size_t>(count));
    }
    if (!cached_cols_) last_im2col_ = Tensor(Shape({0}), backend_, device);  // free it until the next pass
    Tensor output(Shape({N, out_channels_, out_h, out_w}), backend_, device);
    backend_->add_channel_vector(last_pre_bias_output_.data(), bias_.data(), output.data(), n, oc, q);
    has_forwarded_ = true;
    return output;
}

int64_t Conv2DModule::chunk_examples(int64_t n, int64_t p, int64_t q) const {
    const auto per_example = static_cast<size_t>(p * q) * sizeof(float);
    const size_t fit = per_example == 0 ? static_cast<size_t>(n) : max_workspace_bytes_ / per_example;
    return std::clamp<int64_t>(static_cast<int64_t>(std::min<size_t>(fit, static_cast<size_t>(n))), 1, std::max<int64_t>(n, 1));
}

const float* Conv2DModule::patches(int64_t e0, int64_t count, Tensor& buffer) const {
    if (cached_cols_) {
        return last_im2col_.data() + e0 * in_channels_ * kernel_h_ * kernel_w_ * last_out_h_ * last_out_w_;
    }
    const int64_t H = last_input_.shape().dim(2), W = last_input_.shape().dim(3);
    backend_->im2col(last_input_.data() + e0 * in_channels_ * H * W, buffer.data(), static_cast<size_t>(count),
                     static_cast<size_t>(in_channels_), static_cast<size_t>(H), static_cast<size_t>(W), geometry());
    return buffer.data();
}

Tensor Conv2DModule::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "Conv2DModule::backward");
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

    // Device-generic (GPU-native-kernels Mission 4): the kernel and bias gradients are computed
    // per example and accumulated in example order (the original association); dY @ col^T and
    // K^T @ dY read their transposed operand in place.
    const DeviceType device = kernel_.device();
    const auto p = static_cast<size_t>(P), q = static_cast<size_t>(Q), oc = static_cast<size_t>(out_channels_);
    Tensor ones(Shape({Q, 1}), backend_, device);
    ones.fill(1.0f);
    // HIP-7: in chunks of examples, rebuilding the patches when forward() didn't keep them.
    const int64_t chunk = chunk_examples(N, P, Q);
    Tensor cols_buffer(cached_cols_ ? Shape({0}) : Shape({chunk, P, Q}), backend_, device);
    Tensor grad_cols(Shape({chunk, P, Q}), backend_, device);
    // HIP-6: one batched GEMM per chunk for each product. Per-example parameter gradients go to
    // part buffers and are added to the gradients in example order, the sums the per-example
    // accumulation gave. The kernel's parts are bounded by the workspace budget too.
    const int64_t kernel_numel = out_channels_ * P;
    const int64_t kernel_chunk = std::clamp<int64_t>(
        static_cast<int64_t>(max_workspace_bytes_ / (static_cast<size_t>(kernel_numel) * sizeof(float))), 1, chunk);
    Tensor kernel_parts(kernel_.requires_grad() ? Shape({kernel_chunk, out_channels_, P}) : Shape({0}), backend_, device);
    Tensor bias_parts(bias_.requires_grad() ? Shape({chunk, out_channels_}) : Shape({0}), backend_, device);
    Tensor grad_input(last_input_.shape(), backend_, device);
    grad_input.fill(0.0f);
    for (int64_t e0 = 0; e0 < N; e0 += chunk) {
        const int64_t count = std::min(chunk, N - e0);
        const float* grad_out = grad_output.data() + e0 * out_stride;
        // Frozen parameters (FND-2) skip their gradient GEMMs; the input gradient below doesn't need them.
        if (kernel_.requires_grad()) {
            const float* cols = patches(e0, count, cols_buffer);
            for (int64_t k0 = 0; k0 < count; k0 += kernel_chunk) {
                const int64_t kc = std::min(kernel_chunk, count - k0);
                backend_->gemm_strided_batched(grad_out + k0 * out_stride, false, oc * q, cols + k0 * P * Q, true, p * q,
                                               kernel_parts.data(), static_cast<size_t>(kernel_numel), oc, q, p,
                                               static_cast<size_t>(kc));
                backend_->accumulate_parts(kernel_parts.data(), static_cast<size_t>(kc),
                                           static_cast<size_t>(kernel_numel), kernel_grad_.data());
            }
        }
        if (bias_.requires_grad()) {
            backend_->gemm_strided_batched(grad_out, false, oc * q, ones.data(), false, 0, bias_parts.data(), oc, oc, q, 1,
                                           static_cast<size_t>(count));
            backend_->accumulate_parts(bias_parts.data(), static_cast<size_t>(count), oc, bias_grad_.data());
        }
        backend_->gemm_strided_batched(kernel_.data(), true, 0, grad_out, false, oc * q, grad_cols.data(), p * q, p, oc, q,
                                       static_cast<size_t>(count));
        backend_->col2im_add(grad_cols.data(), grad_input.data() + e0 * in_stride, static_cast<size_t>(count),
                             static_cast<size_t>(in_channels_), static_cast<size_t>(H), static_cast<size_t>(W),
                             geometry());
    }
    return grad_input;
}

Tensor Conv2DModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& outer) {
    // A composite's rule for convolutions wherever they are (KS-9) overrides the module's own.
    const LRPRuleConfig& config = outer.conv_rule ? *outer.conv_rule : outer;
    require_device(relevance_out, *compute_device(), "Conv2DModule::propagate_relevance");
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

    // Device-generic (GPU-native-kernels Mission 4): the rule in patch space, then folded back.
    // HIP-7: in chunks of examples; every rule is per example, so the result doesn't depend on
    // the chunk size.
    const DeviceType device = kernel_.device();
    const int64_t chunk = chunk_examples(N, P, Q);
    Tensor cols_buffer(cached_cols_ ? Shape({0}) : Shape({chunk, P, Q}), backend_, device);
    Tensor relevance_cols(Shape({chunk, P, Q}), backend_, device);
    Tensor relevance_in(last_input_.shape(), backend_, device);
    relevance_in.fill(0.0f);
    for (int64_t e0 = 0; e0 < N; e0 += chunk) {
        const int64_t count = std::min(chunk, N - e0);
        const float* cols = patches(e0, count, cols_buffer);
        const float* pre_bias = last_pre_bias_output_.data() + e0 * out_stride;
        const float* r_out = relevance_out.data() + e0 * out_stride;
        if (lrp_rules::is_legacy_epsilon(config)) {
            backend_->lrp_conv(cols, kernel_.data(), pre_bias, r_out, relevance_cols.data(), static_cast<size_t>(count),
                               static_cast<size_t>(out_channels_), static_cast<size_t>(P), static_cast<size_t>(Q),
                               config.epsilon);
        } else {
            // Zennit-compatible rules (LRP-rules Mission 2) in patch space: per example, the layer is
            // out_e = K (OC x P) @ col_e (P x Q) + b, so the patches are the rule's input operand.
            const auto n = static_cast<size_t>(count), p = static_cast<size_t>(P), q = static_cast<size_t>(Q),
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
                be->gemm_strided_batched(k, false, 0, col, false, p * q, y, oc * q, oc, p, q, n);
            };
            op.backward = [=](const float* g, const float* k, float* gcol) {
                be->gemm_strided_batched(k, true, 0, g, false, oc * q, gcol, p * q, p, oc, q, n);
            };
            op.add_bias = [=](const float* y, const float* b, float* o) { be->add_channel_vector(y, b, o, n, oc, q); };
            // ZBox bounds are images unfolded like the input, so padding taps get a zero bound.
            const Shape bound_shape({count, in_channels_, H, W});
            const auto in_c = static_cast<size_t>(in_channels_), h = static_cast<size_t>(H), w = static_cast<size_t>(W);
            const ConvGeometry g = geometry();
            op.fill_bound = [=](float value, float* col) {
                Tensor bound(bound_shape, be, device);
                bound.fill(value);
                be->im2col(bound.data(), col, n, in_c, h, w, g);
            };
            lrp_rules::apply(op, cols, kernel_.data(), bias_.data(), pre_bias, r_out, relevance_cols.data(), config);
        }
        backend_->col2im_add(relevance_cols.data(), relevance_in.data() + e0 * in_stride, static_cast<size_t>(count),
                             static_cast<size_t>(in_channels_), static_cast<size_t>(H), static_cast<size_t>(W),
                             geometry());
    }
    return relevance_in;
}

}  // namespace pulsatrix
