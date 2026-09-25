#include "pulsatrix/conv2d_module.hpp"

#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

namespace {

// input (C,H,W) -> matrix (C*kh*kw, out_h*out_w); column q holds the flattened receptive
// field for output position q, in (channel, row, col) order -- matching kernel_'s own
// row-major (out_channels, in_channels, kh, kw) layout so kernel_.data() can be read
// directly as an (out_channels, P) matrix with no reshape.
// @note Takes a raw pointer (not a Tensor&), operating on one example's worth of a
// possibly-batched buffer -- campaign_exai_dl_library_batch_dimension_support's batching
// approach (a per-example loop over this exact unbatched pipeline, not a new primitive).
Tensor im2col(const float* input_data, int64_t C, int64_t H, int64_t W, int64_t kh, int64_t kw, int64_t out_h,
              int64_t out_w, DeviceBackend* backend) {
    int64_t P = C * kh * kw;
    int64_t Q = out_h * out_w;
    Tensor col(Shape({P, Q}), backend);
    for (int64_t oh = 0; oh < out_h; ++oh) {
        for (int64_t ow = 0; ow < out_w; ++ow) {
            int64_t q = oh * out_w + ow;
            int64_t p = 0;
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t i = 0; i < kh; ++i) {
                    for (int64_t j = 0; j < kw; ++j) {
                        int64_t ih = oh + i;
                        int64_t iw = ow + j;
                        col.data()[p * Q + q] = input_data[(c * H + ih) * W + iw];
                        ++p;
                    }
                }
            }
        }
    }
    return col;
}

// Inverse of im2col -- scatter-accumulates a (P,Q) matrix into a caller-provided (C,H,W)
// output buffer (must already be zeroed -- this function only adds into it), SUMMING
// contributions at any input position touched by more than one output position (stride <
// kernel size always produces overlap). This overlap-summing is exactly what both
// gradient backprop and LRP relevance redistribution need on the way back to input space.
// @note Writes into out_data directly (rather than allocating and returning a small (C,H,W)
// Tensor) so a per-example batching loop can accumulate straight into the right slice of a
// batched (N,C,H,W) tensor with no extra copy.
void col2im_into(const float* col_data, float* out_data, int64_t C, int64_t H, int64_t W, int64_t kh, int64_t kw,
                  int64_t out_h, int64_t out_w) {
    int64_t Q = out_h * out_w;
    for (int64_t oh = 0; oh < out_h; ++oh) {
        for (int64_t ow = 0; ow < out_w; ++ow) {
            int64_t q = oh * out_w + ow;
            int64_t p = 0;
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t i = 0; i < kh; ++i) {
                    for (int64_t j = 0; j < kw; ++j) {
                        int64_t ih = oh + i;
                        int64_t iw = ow + j;
                        out_data[(c * H + ih) * W + iw] += col_data[p * Q + q];
                        ++p;
                    }
                }
            }
        }
    }
}

// Transposes a (rows x cols) row-major buffer into a (cols x rows) row-major buffer.
// Same helper shape as LinearModule's -- CPUBackend::gemm has no transpose flag.
Tensor transpose2d(const float* data, int64_t rows, int64_t cols, DeviceBackend* backend) {
    Tensor out(Shape({cols, rows}), backend);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            out.data()[c * rows + r] = data[r * cols + c];
        }
    }
    return out;
}

}  // namespace

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
    // Dereferences Tensor::data() directly (bias-add loop, plus im2col()) -- not yet
    // backend-generic. See campaign_exai_dl_library_phase1_5_cuda_backend.md's scope
    // decision and mission_host_loop_guards.md.
    PULSATRIX_ASSERT(input.device() == DeviceType::Cpu);

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

    last_input_ = input;
    last_out_h_ = out_h;
    last_out_w_ = out_w;
    last_im2col_ = Tensor(Shape({N, P, Q}), backend_);
    last_pre_bias_output_ = Tensor(Shape({N, out_channels_, out_h, out_w}), backend_);

    Tensor output(Shape({N, out_channels_, out_h, out_w}), backend_);

    // Per-example loop over the existing unbatched im2col/gemm pipeline -- each im2col
    // column is independent of every other, so this is exactly equivalent to a single
    // larger gemm with Q extended to N*Q, without needing a new backend primitive or a
    // batch-aware im2col.
    for (int64_t n = 0; n < N; ++n) {
        const float* input_n = input.data() + n * in_stride;
        Tensor col = im2col(input_n, in_channels_, H, W, kernel_h_, kernel_w_, out_h, out_w, backend_);
        for (int64_t i = 0; i < P * Q; ++i) {
            last_im2col_.data()[n * P * Q + i] = col.data()[i];
        }

        backend_->gemm(kernel_.data(), col.data(), last_pre_bias_output_.data() + n * out_stride,
                        static_cast<size_t>(out_channels_), static_cast<size_t>(P), static_cast<size_t>(Q));

        for (int64_t oc = 0; oc < out_channels_; ++oc) {
            float b = bias_.data()[oc];
            for (int64_t q = 0; q < Q; ++q) {
                int64_t idx = n * out_stride + oc * Q + q;
                output.data()[idx] = last_pre_bias_output_.data()[idx] + b;
            }
        }
    }

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
    // Dereferences Tensor::data() directly (transpose2d()/col2im_into() helpers, plus its
    // own bias-grad loop) -- not yet backend-generic. See
    // campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision and
    // mission_host_loop_guards.md.
    PULSATRIX_ASSERT(grad_output.device() == DeviceType::Cpu);

    const int64_t P = in_channels_ * kernel_h_ * kernel_w_;
    const int64_t Q = last_out_h_ * last_out_w_;
    const int64_t H = last_input_.shape().dim(2);
    const int64_t W = last_input_.shape().dim(3);
    const int64_t in_stride = in_channels_ * H * W;
    const int64_t out_stride = out_channels_ * Q;

    Tensor grad_input(last_input_.shape(), backend_);
    grad_input.fill(0.0f);

    Tensor kernel_t = transpose2d(kernel_.data(), out_channels_, P, backend_);  // (out_channels,P) -> (P,out_channels)

    for (int64_t n = 0; n < N; ++n) {
        const float* grad_out_n = grad_output.data() + n * out_stride;
        const float* col_n = last_im2col_.data() + n * P * Q;

        // grad_kernel += grad_out_n(out_channels,Q) @ col_n^T(Q,P) = (out_channels,P)
        Tensor col_t = transpose2d(col_n, P, Q, backend_);  // (P,Q) -> (Q,P)
        Tensor per_example_kernel_grad(Shape({out_channels_, in_channels_, kernel_h_, kernel_w_}), backend_);
        backend_->gemm(grad_out_n, col_t.data(), per_example_kernel_grad.data(), static_cast<size_t>(out_channels_),
                        static_cast<size_t>(Q), static_cast<size_t>(P));
        kernel_grad_.accumulate(per_example_kernel_grad);

        // grad_bias[oc] += sum over Q of grad_out_n[oc][q]
        Tensor per_example_bias_grad(Shape({out_channels_}), backend_);
        for (int64_t oc = 0; oc < out_channels_; ++oc) {
            float sum = 0.0f;
            for (int64_t q = 0; q < Q; ++q) {
                sum += grad_out_n[oc * Q + q];
            }
            per_example_bias_grad.data()[oc] = sum;
        }
        bias_grad_.accumulate(per_example_bias_grad);

        // grad_col = kernel^T(P,out_channels) @ grad_out_n(out_channels,Q) = (P,Q)
        Tensor grad_col(Shape({P, Q}), backend_);
        backend_->gemm(kernel_t.data(), grad_out_n, grad_col.data(), static_cast<size_t>(P),
                        static_cast<size_t>(out_channels_), static_cast<size_t>(Q));

        col2im_into(grad_col.data(), grad_input.data() + n * in_stride, in_channels_, H, W, kernel_h_, kernel_w_,
                    last_out_h_, last_out_w_);
    }

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

    // Dereferences Tensor::data() directly (its own loop, plus col2im_into()) -- not yet
    // backend-generic. See campaign_exai_dl_library_phase1_5_cuda_backend.md's scope
    // decision.
    PULSATRIX_ASSERT(relevance_out.device() == DeviceType::Cpu);

    const int64_t P = in_channels_ * kernel_h_ * kernel_w_;
    const int64_t Q = last_out_h_ * last_out_w_;
    const int64_t H = last_input_.shape().dim(2);
    const int64_t W = last_input_.shape().dim(3);
    const int64_t in_stride = in_channels_ * H * W;
    const int64_t out_stride = out_channels_ * Q;

    Tensor relevance_in(last_input_.shape(), backend_);
    relevance_in.fill(0.0f);

    // Applied independently per example -- each row's relevance redistribution uses only
    // that row's own cached z/x (last_pre_bias_output_/last_im2col_), no cross-example
    // coupling.
    for (int64_t n = 0; n < N; ++n) {
        const float* relevance_out_n = relevance_out.data() + n * out_stride;
        const float* pre_bias_n = last_pre_bias_output_.data() + n * out_stride;
        const float* col_n = last_im2col_.data() + n * P * Q;

        Tensor relevance_col(Shape({P, Q}), backend_);
        relevance_col.fill(0.0f);

        for (int64_t q = 0; q < Q; ++q) {
            for (int64_t oc = 0; oc < out_channels_; ++oc) {
                float z = pre_bias_n[oc * Q + q];
                float sign = (z >= 0.0f) ? 1.0f : -1.0f;
                float denom = z + config.epsilon * sign;
                float r = relevance_out_n[oc * Q + q];

                for (int64_t p = 0; p < P; ++p) {
                    float w = kernel_.data()[oc * P + p];
                    float a = col_n[p * Q + q];
                    relevance_col.data()[p * Q + q] += (a * w / denom) * r;
                }
            }
        }

        col2im_into(relevance_col.data(), relevance_in.data() + n * in_stride, in_channels_, H, W, kernel_h_,
                    kernel_w_, last_out_h_, last_out_w_);
    }

    return relevance_in;
}

}  // namespace pulsatrix
