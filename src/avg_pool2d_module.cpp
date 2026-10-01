#include "pulsatrix/avg_pool2d_module.hpp"

#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

AvgPool2DModule::AvgPool2DModule(int64_t kernel_h, int64_t kernel_w, DeviceBackend* backend, float eps)
    : kernel_h_(kernel_h), kernel_w_(kernel_w), backend_(backend), eps_(eps), last_input_(Shape({0}), backend) {
    // External boundary (construction arguments can originate from Phase 5's Python
    // bindings with no upstream validation).
    if (kernel_h <= 0 || kernel_w <= 0) {
        throw std::invalid_argument("AvgPool2DModule: kernel_h and kernel_w must be positive");
    }
}

Tensor AvgPool2DModule::forward_impl(const Tensor& input) {
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    PULSATRIX_REQUIRE_HOST(input);

    if (input.rank() != 4) {
        throw std::invalid_argument("AvgPool2DModule::forward: input must be rank-4 (N, C, H, W)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t C = input.shape().dim(1);
    const int64_t H = input.shape().dim(2);
    const int64_t W = input.shape().dim(3);
    if (kernel_h_ > H || kernel_w_ > W) {
        throw std::invalid_argument("AvgPool2DModule::forward: kernel is larger than the input");
    }
    const int64_t out_h = (H - kernel_h_) / kernel_h_ + 1;
    const int64_t out_w = (W - kernel_w_) / kernel_w_ + 1;
    const int64_t in_plane = H * W;
    const int64_t out_plane = out_h * out_w;
    const int64_t K = kernel_h_ * kernel_w_;

    last_input_ = input;
    last_out_h_ = out_h;
    last_out_w_ = out_w;

    Tensor output(Shape({N, C, out_h, out_w}), backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(output);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const float* plane = input.data() + (n * C + c) * in_plane;
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    const int64_t ih0 = oh * kernel_h_;
                    const int64_t iw0 = ow * kernel_w_;
                    for (int64_t i = 0; i < kernel_h_; ++i) {
                        for (int64_t j = 0; j < kernel_w_; ++j) {
                            sum += plane[(ih0 + i) * W + (iw0 + j)];
                        }
                    }
                    output.data()[(n * C + c) * out_plane + oh * out_w + ow] = sum / static_cast<float>(K);
                }
            }
        }
    }

    has_forwarded_ = true;
    return output;
}

Tensor AvgPool2DModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("AvgPool2DModule::backward: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    const int64_t C = last_input_.shape().dim(1);
    if (grad_output.rank() != 4 || grad_output.shape().dim(0) != N || grad_output.shape().dim(1) != C ||
        grad_output.shape().dim(2) != last_out_h_ || grad_output.shape().dim(3) != last_out_w_) {
        throw std::invalid_argument(
            "AvgPool2DModule::backward: grad_output must be rank-4 (N, C, out_h, out_w) matching the cached "
            "forward shape");
    }
    PULSATRIX_REQUIRE_HOST(grad_output);

    const int64_t H = last_input_.shape().dim(2);
    const int64_t W = last_input_.shape().dim(3);
    const int64_t in_plane = H * W;
    const int64_t out_plane = last_out_h_ * last_out_w_;
    const int64_t K = kernel_h_ * kernel_w_;

    Tensor grad_input(last_input_.shape(), backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(grad_input);
    grad_input.fill(0.0f);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            float* in_plane_ptr = grad_input.data() + (n * C + c) * in_plane;
            const float* out_plane_ptr = grad_output.data() + (n * C + c) * out_plane;
            for (int64_t oh = 0; oh < last_out_h_; ++oh) {
                for (int64_t ow = 0; ow < last_out_w_; ++ow) {
                    float g = out_plane_ptr[oh * last_out_w_ + ow] / static_cast<float>(K);
                    const int64_t ih0 = oh * kernel_h_;
                    const int64_t iw0 = ow * kernel_w_;
                    // Non-overlapping windows -> direct assignment is correct (no
                    // scatter-sum needed, unlike Conv2DModule's overlapping-window case).
                    for (int64_t i = 0; i < kernel_h_; ++i) {
                        for (int64_t j = 0; j < kernel_w_; ++j) {
                            in_plane_ptr[(ih0 + i) * W + (iw0 + j)] = g;
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

Tensor AvgPool2DModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) {
        throw std::logic_error("AvgPool2DModule::propagate_relevance: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    const int64_t C = last_input_.shape().dim(1);
    if (relevance_out.rank() != 4 || relevance_out.shape().dim(0) != N || relevance_out.shape().dim(1) != C ||
        relevance_out.shape().dim(2) != last_out_h_ || relevance_out.shape().dim(3) != last_out_w_) {
        throw std::invalid_argument(
            "AvgPool2DModule::propagate_relevance: relevance_out must be rank-4 (N, C, out_h, out_w) matching the "
            "cached forward shape");
    }
    PULSATRIX_REQUIRE_HOST(relevance_out);

    const int64_t H = last_input_.shape().dim(2);
    const int64_t W = last_input_.shape().dim(3);
    const int64_t in_plane = H * W;
    const int64_t out_plane = last_out_h_ * last_out_w_;
    const int64_t K = kernel_h_ * kernel_w_;
    const float inv_k = 1.0f / static_cast<float>(K);

    Tensor relevance_in(last_input_.shape(), backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(relevance_in);
    relevance_in.fill(0.0f);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const float* in_plane_ptr = last_input_.data() + (n * C + c) * in_plane;
            float* relevance_plane_ptr = relevance_in.data() + (n * C + c) * in_plane;
            const float* out_plane_ptr = relevance_out.data() + (n * C + c) * out_plane;
            for (int64_t oh = 0; oh < last_out_h_; ++oh) {
                for (int64_t ow = 0; ow < last_out_w_; ++ow) {
                    const int64_t ih0 = oh * kernel_h_;
                    const int64_t iw0 = ow * kernel_w_;

                    // z = sum(x_i * (1/K)) -- this window's pre-activation, same value as
                    // forward()'s own output for this window.
                    float z = 0.0f;
                    for (int64_t i = 0; i < kernel_h_; ++i) {
                        for (int64_t j = 0; j < kernel_w_; ++j) {
                            z += in_plane_ptr[(ih0 + i) * W + (iw0 + j)] * inv_k;
                        }
                    }
                    float sign = (z >= 0.0f) ? 1.0f : -1.0f;
                    float denom = z + config.epsilon * sign;
                    float r = out_plane_ptr[oh * last_out_w_ + ow];

                    for (int64_t i = 0; i < kernel_h_; ++i) {
                        for (int64_t j = 0; j < kernel_w_; ++j) {
                            int64_t idx = (ih0 + i) * W + (iw0 + j);
                            float a = in_plane_ptr[idx];
                            relevance_plane_ptr[idx] = (a * inv_k / denom) * r;
                        }
                    }
                }
            }
        }
    }

    return relevance_in;
}

}  // namespace pulsatrix
