#include "pulsatrix/softmax_module.hpp"

#include <algorithm>
#include <cmath>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {
namespace {

/**
 * @brief Splits a shape into (number of independent softmax rows, row length) -- every fixed
 *        combination of all dimensions except the last is one row.
 * @note Rank-agnostic on purpose: rank 1 is a single row, rank 2 is (N, C), rank 4 attention
 *       scores are (N * num_heads * L) rows of L. Row-major layout means a row is a
 *       contiguous span, so no stride table is needed -- only the split point.
 */
struct RowLayout {
    int64_t num_rows;
    int64_t row_len;
};

[[nodiscard]] RowLayout row_layout_of(const Shape& shape) {
    PULSATRIX_ASSERT(shape.rank() >= 1);
    const int64_t row_len = shape.dim(static_cast<size_t>(shape.rank() - 1));
    PULSATRIX_ASSERT(row_len > 0);
    return RowLayout{shape.numel() / row_len, row_len};
}

}  // namespace

SoftmaxModule::SoftmaxModule(DeviceBackend* backend)
    : backend_(backend), last_input_(Shape({0}), backend), last_output_(Shape({0}), backend) {}
// Both caches start as zero-element placeholders -- softmax has no fixed shape (unlike
// LinearModule's in_features/out_features), so the real shape is only known once
// forward_impl() is first called and reassigns them wholesale. Same pattern as ReluModule,
// except two caches: Eq. 13 needs the pre-softmax input x as well as the output s.

Tensor SoftmaxModule::forward_impl(const Tensor& input) {
    // Device-generic (GPU-native-kernels Mission 1): max-subtracted row softmax on the
    // input's own device.
    const RowLayout layout = row_layout_of(input.shape());

    Tensor output(input.shape(), backend_, input.device());
    backend_->softmax_rows(input.data(), output.data(), static_cast<size_t>(layout.num_rows),
                           static_cast<size_t>(layout.row_len));

    last_input_ = input;
    last_output_ = output;
    return output;
}

Tensor SoftmaxModule::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "SoftmaxModule::backward");
    PULSATRIX_ASSERT(grad_output.shape() == last_output_.shape());

    const RowLayout layout = row_layout_of(grad_output.shape());

    // dx = y * (dy - <y, dy>) per row, computed from the cached output.
    Tensor grad_input(grad_output.shape(), backend_, grad_output.device());
    backend_->softmax_rows_backward(last_output_.data(), grad_output.data(), grad_input.data(),
                                    static_cast<size_t>(layout.num_rows), static_cast<size_t>(layout.row_len));
    return grad_input;
}

Tensor SoftmaxModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    require_device(relevance_out, *compute_device(), "SoftmaxModule::propagate_relevance");
    PULSATRIX_ASSERT(relevance_out.shape() == last_output_.shape());

    // Device-generic (GPU-native-kernels Mission 3).
    const RowLayout layout = row_layout_of(relevance_out.shape());
    Tensor relevance_in(relevance_out.shape(), backend_, relevance_out.device());
    backend_->lrp_softmax_rows(last_input_.data(), last_output_.data(), relevance_out.data(), relevance_in.data(),
                               static_cast<size_t>(layout.num_rows), static_cast<size_t>(layout.row_len));
    return relevance_in;
}

void SoftmaxModule::release_activations() {
    release_tensor(last_input_);
    release_tensor(last_output_);
}

}  // namespace pulsatrix
