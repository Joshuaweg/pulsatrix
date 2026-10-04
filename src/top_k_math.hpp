// Top-k selection for one row, shared verbatim by CPUBackend (a loop over rows) and the GPU
// kernels (one thread per row), so every backend selects identically. Roadmap FND-3.
// - Order: NaN ranks above every number; equal values keep the lower column index first.
// - Insertion into the caller's k-long output rows, which double as the working buffer: no
//   allocation, so it runs unchanged in device code. O(cols * k) per row, which suits the small
//   k of TopK featurizers, routers and sampling; a large-k device path can come later.
// - Pure selection: values are copied, never computed, so backends agree bit for bit.
// Private to src/.
#pragma once

#include <cstdint>

#include "pointwise_math.hpp"  // PULSATRIX_HOST_DEVICE

namespace pulsatrix {
namespace topk {

// Strict total order with NaN above every number (a != a is the device-safe isnan).
PULSATRIX_HOST_DEVICE inline bool ranks_above(float a, float b) { return (a != a && b == b) || a > b; }

// Whether a belongs strictly before b in the output. Strict, so ties keep scan (index) order.
PULSATRIX_HOST_DEVICE inline bool beats(float a, float b, bool largest) {
    return largest ? ranks_above(a, b) : ranks_above(b, a);
}

// The k best entries of in[0, cols) in rank order, with their column indices as whole-number
// floats. Preconditions (validated by top_k()): 1 <= k <= cols.
PULSATRIX_HOST_DEVICE inline void row(const float* in, float* values, float* indices, int64_t cols, int64_t k,
                                      bool largest) {
    int64_t filled = 0;
    for (int64_t j = 0; j < cols; ++j) {
        const float x = in[j];
        if (filled == k && !beats(x, values[k - 1], largest)) {
            continue;
        }
        // Open a slot at the end (dropping the current k-th entry once full), then shift x
        // left past every entry it strictly beats.
        int64_t p = filled < k ? filled++ : k - 1;
        while (p > 0 && beats(x, values[p - 1], largest)) {
            values[p] = values[p - 1];
            indices[p] = indices[p - 1];
            --p;
        }
        values[p] = x;
        indices[p] = static_cast<float>(j);
    }
}

}  // namespace topk
}  // namespace pulsatrix
