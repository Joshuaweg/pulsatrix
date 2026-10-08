#include "pulsatrix/rope_module.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {
namespace {

/**
 * @brief Splits a `(..., L, head_dim)` shape into (independent slices, positions per slice).
 * @note Rank-agnostic on purpose: everything in front of the last two axes is just "more
 *       slices" -- `(N, L, head_dim)` gives N slices, `(N, num_heads, L, head_dim)` gives
 *       N*num_heads. Row-major layout means one slice is a contiguous `L * head_dim` span
 *       and one position is a contiguous `head_dim` span, so no stride table is needed.
 */
struct SliceLayout {
    int64_t num_matrices;
    int64_t seq_len;
};

[[nodiscard]] SliceLayout slice_layout_of(const Shape& shape, int64_t head_dim) {
    PULSATRIX_ASSERT(shape.rank() >= 2);
    const int64_t seq_len = shape.dim(static_cast<size_t>(shape.rank() - 2));
    PULSATRIX_ASSERT(seq_len > 0);
    PULSATRIX_ASSERT(head_dim > 0);
    return SliceLayout{shape.numel() / (seq_len * head_dim), seq_len};
}


}  // namespace

RoPEModule::RoPEModule(int64_t head_dim, DeviceBackend* backend, std::vector<double> inverse_frequencies,
                       RoPELayout layout)
    : RoPEModule(head_dim, backend, 10000.0f, layout) {
    if (static_cast<int64_t>(inverse_frequencies.size()) != head_dim / 2) {
        throw std::invalid_argument("RoPEModule: needs head_dim / 2 inverse frequencies");
    }
    for (double f : inverse_frequencies) {
        if (!(f > 0.0) || !std::isfinite(f)) {
            throw std::invalid_argument("RoPEModule: inverse frequencies must be positive and finite");
        }
    }
    inv_freq_ = std::move(inverse_frequencies);
    base_ = 0.0f;  // not a geometric schedule
}

RoPEModule::RoPEModule(int64_t head_dim, DeviceBackend* backend, float base, RoPELayout layout)
    : head_dim_(head_dim),
      base_(base),
      layout_(layout),
      backend_(backend),
      last_input_(Shape({0}), backend),
      last_output_(Shape({0}), backend),
      cos_table_(Shape({0}), backend),
      sin_table_(Shape({0}), backend) {
    // External boundary (construction arguments can originate from Phase 5's Python
    // bindings with no upstream validation).
    if (head_dim <= 0) {
        throw std::invalid_argument("RoPEModule: head_dim must be positive");
    }
    if (head_dim % 2 != 0) {
        throw std::invalid_argument("RoPEModule: head_dim must be even (the rotation acts on adjacent pairs)");
    }
    for (int64_t i = 0; i < head_dim / 2; ++i) {
        inv_freq_.push_back(std::pow(static_cast<double>(base), -2.0 * static_cast<double>(i) / static_cast<double>(head_dim)));
    }
}
// Both caches start as zero-element placeholders -- only head_dim is fixed at construction;
// the leading dims and L are known only once forward_impl() first runs. Same pattern as
// SoftmaxModule's two caches: the epsilon rule needs the input x (numerators) as well as
// the output y (denominators).

void RoPEModule::set_position_offset(int64_t offset) {
    if (offset < 0) {
        throw std::invalid_argument("RoPEModule::set_position_offset: offset must be non-negative");
    }
    position_offset_ = offset;
}

void RoPEModule::ensure_tables(int64_t seq_len, int64_t offset) {
    if (seq_len == table_seq_len_ && offset == table_offset_) {
        return;
    }
    const int64_t half = head_dim_ / 2;
    std::vector<float> cos_values(static_cast<size_t>(seq_len * half));
    std::vector<float> sin_values(static_cast<size_t>(seq_len * half));
    for (int64_t pos = 0; pos < seq_len; ++pos) {
        for (int64_t i = 0; i < half; ++i) {
            const double theta = static_cast<double>(offset + pos) * inv_freq_[static_cast<size_t>(i)];
            cos_values[static_cast<size_t>(pos * half + i)] = static_cast<float>(std::cos(theta));
            sin_values[static_cast<size_t>(pos * half + i)] = static_cast<float>(std::sin(theta));
        }
    }
    cos_table_ = Tensor(Shape({seq_len, half}), backend_, cos_values);
    sin_table_ = Tensor(Shape({seq_len, half}), backend_, sin_values);
    table_seq_len_ = seq_len;
    table_offset_ = offset;
}

Tensor RoPEModule::forward_impl(const Tensor& input) {
    if (input.rank() < 2 || input.shape().dim(static_cast<size_t>(input.rank() - 1)) != head_dim_) {
        throw std::invalid_argument("RoPEModule::forward: input must be rank >= 2 with shape (..., L, head_dim)");
    }

    // Device-generic (GPU-native-kernels Mission 2): precomputed tables + one rotate kernel.
    const SliceLayout layout = slice_layout_of(input.shape(), head_dim_);
    ensure_tables(layout.seq_len, position_offset_);
    Tensor output(input.shape(), backend_, input.device());
    backend_->rope_rotate(input.data(), cos_table_.data(), sin_table_.data(), output.data(),
                          static_cast<size_t>(layout.num_matrices), static_cast<size_t>(layout.seq_len),
                          static_cast<size_t>(head_dim_), /*inverse=*/false, layout_ == RoPELayout::RotateHalf);

    last_offset_ = position_offset_;
    last_input_ = input;
    last_output_ = output;
    has_forwarded_ = true;
    return output;
}

Tensor RoPEModule::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "RoPEModule::backward");
    if (!has_forwarded_) {
        throw std::logic_error("RoPEModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_input_.shape()) {
        throw std::invalid_argument("RoPEModule::backward: grad_output must match the cached forward shape");
    }

    // The rotation is orthogonal, so its gradient is the transpose rotation.
    const SliceLayout layout = slice_layout_of(grad_output.shape(), head_dim_);
    ensure_tables(layout.seq_len, last_offset_);
    Tensor grad_input(grad_output.shape(), backend_, grad_output.device());
    backend_->rope_rotate(grad_output.data(), cos_table_.data(), sin_table_.data(), grad_input.data(),
                          static_cast<size_t>(layout.num_matrices), static_cast<size_t>(layout.seq_len),
                          static_cast<size_t>(head_dim_), /*inverse=*/true, layout_ == RoPELayout::RotateHalf);
    return grad_input;
}

Tensor RoPEModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    require_device(relevance_out, *compute_device(), "RoPEModule::propagate_relevance");
    if (!has_forwarded_) {
        throw std::logic_error("RoPEModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.shape() != last_output_.shape()) {
        throw std::invalid_argument(
            "RoPEModule::propagate_relevance: relevance_out must match the cached forward shape");
    }
    // Device-generic (GPU-native-kernels Mission 3): same cached cos/sin tables as forward.
    const SliceLayout layout = slice_layout_of(relevance_out.shape(), head_dim_);
    ensure_tables(layout.seq_len, last_offset_);
    Tensor relevance_in(relevance_out.shape(), backend_, relevance_out.device());
    backend_->lrp_rope(last_input_.data(), last_output_.data(), relevance_out.data(), cos_table_.data(),
                       sin_table_.data(), relevance_in.data(), static_cast<size_t>(layout.num_matrices),
                       static_cast<size_t>(layout.seq_len), static_cast<size_t>(head_dim_), config.epsilon,
                       layout_ == RoPELayout::RotateHalf);
    return relevance_in;
}

void RoPEModule::release_activations() {
    release_tensor(last_input_);
    release_tensor(last_output_);
    has_forwarded_ = false;
}

}  // namespace pulsatrix
