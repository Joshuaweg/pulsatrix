#include "pulsatrix/rope_module.hpp"

#include <cmath>
#include <stdexcept>

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

/**
 * @brief The rotation angle for feature pair `i` at sequence position `pos`:
 *        `theta_i = pos * base^(-2i/head_dim)`.
 * @note Computed in double and narrowed once at the end -- `std::pow` in float loses
 *       enough precision at large `head_dim` to show up against hand-computed references.
 *       At `pos == 0` this is exactly 0.0 for every pair (the identity case), which is why
 *       position 0 comes back bit-identical rather than merely close.
 */
[[nodiscard]] double rope_angle(int64_t pos, int64_t i, int64_t head_dim, float base) {
    const double exponent = -2.0 * static_cast<double>(i) / static_cast<double>(head_dim);
    return static_cast<double>(pos) * std::pow(static_cast<double>(base), exponent);
}

}  // namespace

RoPEModule::RoPEModule(int64_t head_dim, DeviceBackend* backend, float base)
    : head_dim_(head_dim),
      base_(base),
      backend_(backend),
      last_input_(Shape({0}), backend),
      last_output_(Shape({0}), backend) {
    // External boundary (construction arguments can originate from Phase 5's Python
    // bindings with no upstream validation).
    if (head_dim <= 0) {
        throw std::invalid_argument("RoPEModule: head_dim must be positive");
    }
    if (head_dim % 2 != 0) {
        throw std::invalid_argument("RoPEModule: head_dim must be even (the rotation acts on adjacent pairs)");
    }
}
// Both caches start as zero-element placeholders -- only head_dim is fixed at construction;
// the leading dims and L are known only once forward_impl() first runs. Same pattern as
// SoftmaxModule's two caches: the epsilon rule needs the input x (numerators) as well as
// the output y (denominators).

Tensor RoPEModule::forward_impl(const Tensor& input) {
    // Dereferences Tensor::data() directly in a raw host loop -- not backend-generic.
    // See mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(input);

    if (input.rank() < 2 || input.shape().dim(static_cast<size_t>(input.rank() - 1)) != head_dim_) {
        throw std::invalid_argument("RoPEModule::forward: input must be rank >= 2 with shape (..., L, head_dim)");
    }

    const SliceLayout layout = slice_layout_of(input.shape(), head_dim_);
    const int64_t half = head_dim_ / 2;

    Tensor output(input.shape(), backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(output);
    for (int64_t m = 0; m < layout.num_matrices; ++m) {
        for (int64_t pos = 0; pos < layout.seq_len; ++pos) {
            const int64_t base_off = (m * layout.seq_len + pos) * head_dim_;
            for (int64_t i = 0; i < half; ++i) {
                const double theta = rope_angle(pos, i, head_dim_, base_);
                const float c = static_cast<float>(std::cos(theta));
                const float s = static_cast<float>(std::sin(theta));

                const float x0 = input.data()[base_off + 2 * i];
                const float x1 = input.data()[base_off + 2 * i + 1];
                output.data()[base_off + 2 * i] = x0 * c - x1 * s;
                output.data()[base_off + 2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }

    last_input_ = input;
    last_output_ = output;
    has_forwarded_ = true;
    return output;
}

Tensor RoPEModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("RoPEModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_input_.shape()) {
        throw std::invalid_argument("RoPEModule::backward: grad_output must match the cached forward shape");
    }
    // Raw host loop -- see the header's note and mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(grad_output);

    const SliceLayout layout = slice_layout_of(grad_output.shape(), head_dim_);
    const int64_t half = head_dim_ / 2;

    Tensor grad_input(grad_output.shape(), backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(grad_input);
    for (int64_t m = 0; m < layout.num_matrices; ++m) {
        for (int64_t pos = 0; pos < layout.seq_len; ++pos) {
            const int64_t base_off = (m * layout.seq_len + pos) * head_dim_;
            for (int64_t i = 0; i < half; ++i) {
                const double theta = rope_angle(pos, i, head_dim_, base_);
                const float c = static_cast<float>(std::cos(theta));
                const float s = static_cast<float>(std::sin(theta));

                // Inverse rotation (R^{-1} = R^T for an orthogonal rotation) -- the forward
                // formula with the sin terms' signs swapped.
                const float g0 = grad_output.data()[base_off + 2 * i];
                const float g1 = grad_output.data()[base_off + 2 * i + 1];
                grad_input.data()[base_off + 2 * i] = g0 * c + g1 * s;
                grad_input.data()[base_off + 2 * i + 1] = -g0 * s + g1 * c;
            }
        }
    }
    return grad_input;
}

Tensor RoPEModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) {
        throw std::logic_error("RoPEModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.shape() != last_output_.shape()) {
        throw std::invalid_argument(
            "RoPEModule::propagate_relevance: relevance_out must match the cached forward shape");
    }
    // Raw host loop -- see the header's note and mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(relevance_out);

    const SliceLayout layout = slice_layout_of(relevance_out.shape(), head_dim_);
    const int64_t half = head_dim_ / 2;

    Tensor relevance_in(relevance_out.shape(), backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(relevance_in);
    // Zero-filled up front precisely because every element below is written with `+=`, not
    // `=` -- see the two-contribution note at the accumulation site.
    relevance_in.fill(0.0f);

    for (int64_t m = 0; m < layout.num_matrices; ++m) {
        for (int64_t pos = 0; pos < layout.seq_len; ++pos) {
            const int64_t base_off = (m * layout.seq_len + pos) * head_dim_;
            for (int64_t i = 0; i < half; ++i) {
                const double theta = rope_angle(pos, i, head_dim_, base_);
                const float c = static_cast<float>(std::cos(theta));
                const float s = static_cast<float>(std::sin(theta));

                const int64_t lo = base_off + 2 * i;      // flat index of x[2i]   / y[2i]
                const int64_t hi = base_off + 2 * i + 1;  // flat index of x[2i+1] / y[2i+1]

                const float x0 = last_input_.data()[lo];
                const float x1 = last_input_.data()[hi];
                const float y0 = last_output_.data()[lo];
                const float y1 = last_output_.data()[hi];

                // Epsilon-stabilized denominators, signed to match z_j (same sign(0) == +1
                // convention as RNNModule/LinearModule).
                const float denom0 = y0 + config.epsilon * ((y0 >= 0.0f) ? 1.0f : -1.0f);
                const float denom1 = y1 + config.epsilon * ((y1 >= 0.0f) ? 1.0f : -1.0f);

                const float r0 = relevance_out.data()[lo];
                const float r1 = relevance_out.data()[hi];

                // Source 1: y[2i] = x[2i]*cos + x[2i+1]*(-sin), redistributed across both
                // inputs of this pair.
                relevance_in.data()[lo] += (x0 * c / denom0) * r0;
                relevance_in.data()[hi] += (x1 * -s / denom0) * r0;

                // Source 2: y[2i+1] = x[2i]*sin + x[2i+1]*cos. These MUST accumulate on top
                // of source 1's writes above -- each x-component of the pair receives
                // relevance from BOTH output components, so `lo` and `hi` each get exactly
                // two `+=` writes per (matrix, position, pair). Turning either of these four
                // into `=` silently drops half the relevance and breaks conservation
                // (structurally the same trap as GRUModule's two-path accumulator).
                relevance_in.data()[lo] += (x0 * s / denom1) * r1;
                relevance_in.data()[hi] += (x1 * c / denom1) * r1;
            }
        }
    }
    return relevance_in;
}

}  // namespace pulsatrix
