#include "exai/conv2d_module.hpp"

namespace exai {

namespace {

// input (C,H,W) -> matrix (C*kh*kw, out_h*out_w); column q holds the flattened receptive
// field for output position q, in (channel, row, col) order -- matching kernel_'s own
// row-major (out_channels, in_channels, kh, kw) layout so kernel_.data() can be read
// directly as an (out_channels, P) matrix with no reshape.
Tensor im2col(const Tensor& input, int64_t C, int64_t H, int64_t W, int64_t kh, int64_t kw, int64_t out_h,
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
                        col.data()[p * Q + q] = input.data()[(c * H + ih) * W + iw];
                        ++p;
                    }
                }
            }
        }
    }
    return col;
}

// Inverse of im2col -- scatter-accumulates a (P,Q) matrix back into (C,H,W), SUMMING
// contributions at any input position touched by more than one output position (stride <
// kernel size always produces overlap). This overlap-summing is exactly what both
// gradient backprop and LRP relevance redistribution need on the way back to input space.
Tensor col2im(const Tensor& col, int64_t C, int64_t H, int64_t W, int64_t kh, int64_t kw, int64_t out_h,
              int64_t out_w, DeviceBackend* backend) {
    Tensor out(Shape({C, H, W}), backend);
    out.fill(0.0f);
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
                        out.data()[(c * H + ih) * W + iw] += col.data()[p * Q + q];
                        ++p;
                    }
                }
            }
        }
    }
    return out;
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

Tensor Conv2DModule::forward_impl(const Tensor& input) {
    int64_t H = input.shape().dim(1);
    int64_t W = input.shape().dim(2);
    int64_t out_h = H - kernel_h_ + 1;
    int64_t out_w = W - kernel_w_ + 1;
    int64_t P = in_channels_ * kernel_h_ * kernel_w_;
    int64_t Q = out_h * out_w;

    last_input_ = input;
    last_out_h_ = out_h;
    last_out_w_ = out_w;
    last_im2col_ = im2col(input, in_channels_, H, W, kernel_h_, kernel_w_, out_h, out_w, backend_);

    Tensor pre_bias(Shape({out_channels_, Q}), backend_);
    backend_->gemm(kernel_.data(), last_im2col_.data(), pre_bias.data(), static_cast<size_t>(out_channels_),
                    static_cast<size_t>(P), static_cast<size_t>(Q));
    last_pre_bias_output_ = pre_bias;
    last_pre_bias_output_.reshape(Shape({out_channels_, out_h, out_w}));

    Tensor output(pre_bias);
    for (int64_t oc = 0; oc < out_channels_; ++oc) {
        float b = bias_.data()[oc];
        for (int64_t q = 0; q < Q; ++q) {
            output.data()[oc * Q + q] += b;
        }
    }
    output.reshape(Shape({out_channels_, out_h, out_w}));
    return output;
}

Tensor Conv2DModule::backward(const Tensor& grad_output) {
    int64_t P = in_channels_ * kernel_h_ * kernel_w_;
    int64_t Q = last_out_h_ * last_out_w_;

    // grad_output is already (out_channels, out_h, out_w) row-major, identical bytes to
    // (out_channels, Q) -- read its buffer directly, no reshape of the const input needed.

    // grad_kernel = grad_output_matrix(out_channels,Q) @ im2col^T(Q,P) = (out_channels,P)
    Tensor col_t = transpose2d(last_im2col_.data(), P, Q, backend_);  // (P,Q) -> (Q,P)
    Tensor local_kernel_grad(Shape({out_channels_, in_channels_, kernel_h_, kernel_w_}), backend_);
    backend_->gemm(grad_output.data(), col_t.data(), local_kernel_grad.data(), static_cast<size_t>(out_channels_),
                    static_cast<size_t>(Q), static_cast<size_t>(P));
    kernel_grad_.accumulate(local_kernel_grad);

    // grad_bias[oc] = sum over Q of grad_output[oc][q]
    Tensor local_bias_grad(Shape({out_channels_}), backend_);
    local_bias_grad.fill(0.0f);
    for (int64_t oc = 0; oc < out_channels_; ++oc) {
        float sum = 0.0f;
        for (int64_t q = 0; q < Q; ++q) {
            sum += grad_output.data()[oc * Q + q];
        }
        local_bias_grad.data()[oc] = sum;
    }
    bias_grad_.accumulate(local_bias_grad);

    // grad_col = kernel^T(P,out_channels) @ grad_output_matrix(out_channels,Q) = (P,Q)
    Tensor kernel_t = transpose2d(kernel_.data(), out_channels_, P, backend_);  // (out_channels,P) -> (P,out_channels)
    Tensor grad_col(Shape({P, Q}), backend_);
    backend_->gemm(kernel_t.data(), grad_output.data(), grad_col.data(), static_cast<size_t>(P),
                    static_cast<size_t>(out_channels_), static_cast<size_t>(Q));

    int64_t H = last_input_.shape().dim(1);
    int64_t W = last_input_.shape().dim(2);
    return col2im(grad_col, in_channels_, H, W, kernel_h_, kernel_w_, last_out_h_, last_out_w_, backend_);
}

Tensor Conv2DModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    int64_t P = in_channels_ * kernel_h_ * kernel_w_;
    int64_t Q = last_out_h_ * last_out_w_;

    Tensor relevance_col(Shape({P, Q}), backend_);
    relevance_col.fill(0.0f);

    for (int64_t q = 0; q < Q; ++q) {
        for (int64_t oc = 0; oc < out_channels_; ++oc) {
            float z = last_pre_bias_output_.data()[oc * Q + q];
            float sign = (z >= 0.0f) ? 1.0f : -1.0f;
            float denom = z + config.epsilon * sign;
            float r = relevance_out.data()[oc * Q + q];

            for (int64_t p = 0; p < P; ++p) {
                float w = kernel_.data()[oc * P + p];
                float a = last_im2col_.data()[p * Q + q];
                relevance_col.data()[p * Q + q] += (a * w / denom) * r;
            }
        }
    }

    int64_t H = last_input_.shape().dim(1);
    int64_t W = last_input_.shape().dim(2);
    return col2im(relevance_col, in_channels_, H, W, kernel_h_, kernel_w_, last_out_h_, last_out_w_, backend_);
}

}  // namespace exai
