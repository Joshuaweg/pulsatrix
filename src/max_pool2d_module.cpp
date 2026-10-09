#include "pulsatrix/max_pool2d_module.hpp"

#include <limits>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

MaxPool2DModule::MaxPool2DModule(int64_t kernel_h, int64_t kernel_w, DeviceBackend* backend)
    : MaxPool2DModule(kernel_h, kernel_w, kernel_h, kernel_w, 0, 0, backend) {}

MaxPool2DModule::MaxPool2DModule(int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w, int64_t pad_h,
                                 int64_t pad_w, DeviceBackend* backend)
    : kernel_h_(kernel_h), kernel_w_(kernel_w), stride_h_(stride_h), stride_w_(stride_w), pad_h_(pad_h), pad_w_(pad_w),
      backend_(backend) {
    // External boundary (construction arguments can originate from Phase 5's Python
    // bindings with no upstream validation).
    if (kernel_h <= 0 || kernel_w <= 0) {
        throw std::invalid_argument("MaxPool2DModule: kernel_h and kernel_w must be positive");
    }
    if (stride_h <= 0 || stride_w <= 0) throw std::invalid_argument("MaxPool2DModule: the stride must be positive");
    if (pad_h < 0 || pad_w < 0 || 2 * pad_h > kernel_h || 2 * pad_w > kernel_w) {
        throw std::invalid_argument("MaxPool2DModule: padding must be between 0 and half the kernel");
    }
}

Tensor MaxPool2DModule::forward_impl(const Tensor& input) {

    // External boundary -- input can originate from Phase 5's Python bindings with no
    // upstream validation.
    if (input.rank() != 4) {
        throw std::invalid_argument("MaxPool2DModule::forward: input must be rank-4 (N, C, H, W)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t C = input.shape().dim(1);
    const int64_t H = input.shape().dim(2);
    const int64_t W = input.shape().dim(3);
    if (kernel_h_ > H + 2 * pad_h_ || kernel_w_ > W + 2 * pad_w_) {
        throw std::invalid_argument("MaxPool2DModule::forward: kernel is larger than the padded input");
    }
    const int64_t out_h = (H + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    const int64_t out_w = (W + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

    last_input_shape_ = input.shape();
    last_out_h_ = out_h;
    last_out_w_ = out_w;
    // Device-generic (GPU-native-kernels Mission 4): one thread per pooled element.
    const auto planes = static_cast<size_t>(N * C);
    argmax_flat_index_ = Tensor(Shape({N, C, out_h, out_w}), backend_, input.device());
    Tensor output(Shape({N, C, out_h, out_w}), backend_, input.device());
    backend_->max_pool_forward(input.data(), output.data(), argmax_flat_index_.data(), planes, static_cast<size_t>(H),
                               static_cast<size_t>(W), static_cast<size_t>(kernel_h_), static_cast<size_t>(kernel_w_),
                               static_cast<size_t>(stride_h_), static_cast<size_t>(stride_w_), static_cast<size_t>(pad_h_),
                               static_cast<size_t>(pad_w_));
    has_forwarded_ = true;
    return output;
}

Tensor MaxPool2DModule::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "MaxPool2DModule::backward");
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

    const int64_t H = last_input_shape_.dim(2);
    const int64_t W = last_input_shape_.dim(3);

    // Everything to each window's argmax, nothing elsewhere; overlapping wins add up.
    Tensor grad_input(last_input_shape_, backend_, grad_output.device());
    backend_->max_unpool(grad_output.data(), argmax_flat_index_.data(), grad_input.data(), static_cast<size_t>(N * C),
                         static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(kernel_h_),
                         static_cast<size_t>(kernel_w_), static_cast<size_t>(stride_h_), static_cast<size_t>(stride_w_),
                         static_cast<size_t>(pad_h_), static_cast<size_t>(pad_w_));
    return grad_input;
}

Tensor MaxPool2DModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    require_device(relevance_out, *compute_device(), "MaxPool2DModule::propagate_relevance");
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

    const int64_t H = last_input_shape_.dim(2);
    const int64_t W = last_input_shape_.dim(3);

    // Everything to each window's argmax, nothing elsewhere; overlapping wins add up.
    Tensor relevance_in(last_input_shape_, backend_, relevance_out.device());
    backend_->max_unpool(relevance_out.data(), argmax_flat_index_.data(), relevance_in.data(), static_cast<size_t>(N * C),
                         static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(kernel_h_),
                         static_cast<size_t>(kernel_w_), static_cast<size_t>(stride_h_), static_cast<size_t>(stride_w_),
                         static_cast<size_t>(pad_h_), static_cast<size_t>(pad_w_));
    return relevance_in;
}

}  // namespace pulsatrix
