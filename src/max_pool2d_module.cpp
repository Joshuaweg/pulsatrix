#include "pulsatrix/max_pool2d_module.hpp"

#include <limits>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

MaxPool2DModule::MaxPool2DModule(int64_t kernel_h, int64_t kernel_w, DeviceBackend* backend)
    : kernel_h_(kernel_h), kernel_w_(kernel_w), backend_(backend) {
    // External boundary (construction arguments can originate from Phase 5's Python
    // bindings with no upstream validation).
    if (kernel_h <= 0 || kernel_w <= 0) {
        throw std::invalid_argument("MaxPool2DModule: kernel_h and kernel_w must be positive");
    }
}

Tensor MaxPool2DModule::forward_impl(const Tensor& input) {
    // Dereferences Tensor::data() directly -- not yet backend-generic. See
    // campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision.
    PULSATRIX_REQUIRE_HOST(input);

    // External boundary -- input can originate from Phase 5's Python bindings with no
    // upstream validation.
    if (input.rank() != 4) {
        throw std::invalid_argument("MaxPool2DModule::forward: input must be rank-4 (N, C, H, W)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t C = input.shape().dim(1);
    const int64_t H = input.shape().dim(2);
    const int64_t W = input.shape().dim(3);
    if (kernel_h_ > H || kernel_w_ > W) {
        throw std::invalid_argument("MaxPool2DModule::forward: kernel is larger than the input");
    }
    const int64_t out_h = (H - kernel_h_) / kernel_h_ + 1;
    const int64_t out_w = (W - kernel_w_) / kernel_w_ + 1;
    const int64_t in_plane = H * W;
    const int64_t out_plane = out_h * out_w;

    last_input_shape_ = input.shape();
    last_out_h_ = out_h;
    last_out_w_ = out_w;
    argmax_flat_index_.assign(static_cast<size_t>(N * C * out_plane), 0);

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
                    float best = -std::numeric_limits<float>::infinity();
                    int64_t best_flat = 0;
                    const int64_t ih0 = oh * kernel_h_;
                    const int64_t iw0 = ow * kernel_w_;
                    for (int64_t i = 0; i < kernel_h_; ++i) {
                        for (int64_t j = 0; j < kernel_w_; ++j) {
                            int64_t ih = ih0 + i;
                            int64_t iw = iw0 + j;
                            int64_t flat = ih * W + iw;
                            float v = plane[flat];
                            if (v > best) {
                                best = v;
                                best_flat = flat;
                            }
                        }
                    }
                    int64_t out_idx = (n * C + c) * out_plane + oh * out_w + ow;
                    output.data()[out_idx] = best;
                    argmax_flat_index_[static_cast<size_t>(out_idx)] = best_flat;
                }
            }
        }
    }

    has_forwarded_ = true;
    return output;
}

Tensor MaxPool2DModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("MaxPool2DModule::backward: called before any forward()");
    }
    const int64_t N = last_input_shape_.dim(0);
    const int64_t C = last_input_shape_.dim(1);
    if (grad_output.rank() != 4 || grad_output.shape().dim(0) != N || grad_output.shape().dim(1) != C ||
        grad_output.shape().dim(2) != last_out_h_ || grad_output.shape().dim(3) != last_out_w_) {
        throw std::invalid_argument(
            "MaxPool2DModule::backward: grad_output must be rank-4 (N, C, out_h, out_w) matching the cached "
            "forward shape");
    }
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    PULSATRIX_REQUIRE_HOST(grad_output);

    const int64_t H = last_input_shape_.dim(2);
    const int64_t W = last_input_shape_.dim(3);
    const int64_t in_plane = H * W;
    const int64_t out_plane = last_out_h_ * last_out_w_;

    Tensor grad_input(last_input_shape_, backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(grad_input);
    grad_input.fill(0.0f);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            float* in_plane_ptr = grad_input.data() + (n * C + c) * in_plane;
            const float* out_plane_ptr = grad_output.data() + (n * C + c) * out_plane;
            for (int64_t q = 0; q < out_plane; ++q) {
                int64_t out_idx = (n * C + c) * out_plane + q;
                int64_t flat = argmax_flat_index_[static_cast<size_t>(out_idx)];
                // Non-overlapping windows (stride == kernel) -> each input position belongs
                // to exactly one window, so direct assignment is correct (no scatter-sum
                // needed, unlike Conv2DModule's overlapping-window col2im_into).
                in_plane_ptr[flat] = out_plane_ptr[q];
            }
        }
    }

    return grad_input;
}

Tensor MaxPool2DModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    if (!has_forwarded_) {
        throw std::logic_error("MaxPool2DModule::propagate_relevance: called before any forward()");
    }
    const int64_t N = last_input_shape_.dim(0);
    const int64_t C = last_input_shape_.dim(1);
    if (relevance_out.rank() != 4 || relevance_out.shape().dim(0) != N || relevance_out.shape().dim(1) != C ||
        relevance_out.shape().dim(2) != last_out_h_ || relevance_out.shape().dim(3) != last_out_w_) {
        throw std::invalid_argument(
            "MaxPool2DModule::propagate_relevance: relevance_out must be rank-4 (N, C, out_h, out_w) matching the "
            "cached forward shape");
    }
    PULSATRIX_REQUIRE_HOST(relevance_out);

    const int64_t H = last_input_shape_.dim(2);
    const int64_t W = last_input_shape_.dim(3);
    const int64_t in_plane = H * W;
    const int64_t out_plane = last_out_h_ * last_out_w_;

    Tensor relevance_in(last_input_shape_, backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(relevance_in);
    relevance_in.fill(0.0f);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            float* in_plane_ptr = relevance_in.data() + (n * C + c) * in_plane;
            const float* out_plane_ptr = relevance_out.data() + (n * C + c) * out_plane;
            for (int64_t q = 0; q < out_plane; ++q) {
                int64_t out_idx = (n * C + c) * out_plane + q;
                int64_t flat = argmax_flat_index_[static_cast<size_t>(out_idx)];
                // Winner-take-all: the argmax position receives all of this window's
                // relevance; every other position stays 0. Conserves trivially.
                in_plane_ptr[flat] = out_plane_ptr[q];
            }
        }
    }

    return relevance_in;
}

}  // namespace pulsatrix
