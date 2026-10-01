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
    // Raw host loop (std::exp per element) -- not backend-generic. There is no Exp
    // elementwise op and adding one is out of this mission's scope (CrossEntropyLoss
    // computes its softmax the same way). See mission_host_loop_guards.md for the guard.
    PULSATRIX_REQUIRE_HOST(input);

    const RowLayout layout = row_layout_of(input.shape());

    Tensor output(input.shape(), backend_);
    for (int64_t row = 0; row < layout.num_rows; ++row) {
        const int64_t base = row * layout.row_len;

        // Numerically stable: subtract this row's own max before exponentiating, same
        // convention as CrossEntropyLoss::forward, applied per row rather than globally.
        float row_max = input.data()[base];
        for (int64_t j = 1; j < layout.row_len; ++j) {
            row_max = std::max(row_max, input.data()[base + j]);
        }

        float exp_sum = 0.0f;
        for (int64_t j = 0; j < layout.row_len; ++j) {
            const float e = std::exp(input.data()[base + j] - row_max);
            output.data()[base + j] = e;
            exp_sum += e;
        }
        for (int64_t j = 0; j < layout.row_len; ++j) {
            output.data()[base + j] /= exp_sum;
        }
    }

    last_input_ = input;
    last_output_ = output;
    return output;
}

Tensor SoftmaxModule::backward(const Tensor& grad_output) {
    // Raw host loop -- see the header's note and mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(grad_output);
    PULSATRIX_ASSERT(grad_output.shape() == last_output_.shape());

    const RowLayout layout = row_layout_of(grad_output.shape());

    Tensor grad_input(grad_output.shape(), backend_);
    for (int64_t row = 0; row < layout.num_rows; ++row) {
        const int64_t base = row * layout.row_len;

        // dot = sum_j(s[j] * grad_out[j]) -- the softmax Jacobian's rank-1 correction term.
        float dot = 0.0f;
        for (int64_t j = 0; j < layout.row_len; ++j) {
            dot += last_output_.data()[base + j] * grad_output.data()[base + j];
        }
        for (int64_t i = 0; i < layout.row_len; ++i) {
            grad_input.data()[base + i] = last_output_.data()[base + i] * (grad_output.data()[base + i] - dot);
        }
    }
    return grad_input;
}

Tensor SoftmaxModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    // Raw host loop -- see the header's note and mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(relevance_out);
    PULSATRIX_ASSERT(relevance_out.shape() == last_output_.shape());

    const RowLayout layout = row_layout_of(relevance_out.shape());

    Tensor relevance_in(relevance_out.shape(), backend_);
    for (int64_t row = 0; row < layout.num_rows; ++row) {
        const int64_t base = row * layout.row_len;

        // Per-row relevance total -- each row's own sum, never a global one.
        float relevance_sum = 0.0f;
        for (int64_t j = 0; j < layout.row_len; ++j) {
            relevance_sum += relevance_out.data()[base + j];
        }
        // AttnLRP Eq. 13: R_in[i] = x[i] * (R_out[i] - s[i] * sum_j(R_out[j])).
        // No epsilon stabilizer and no rescaling -- this rule does not conserve relevance by
        // construction and must not be "corrected" into conserving. See the header.
        for (int64_t i = 0; i < layout.row_len; ++i) {
            relevance_in.data()[base + i] =
                last_input_.data()[base + i] * (relevance_out.data()[base + i] -
                                                last_output_.data()[base + i] * relevance_sum);
        }
    }
    return relevance_in;
}

}  // namespace pulsatrix
