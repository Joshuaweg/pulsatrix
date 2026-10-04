#include "pulsatrix/top_k.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace pulsatrix {

TopKResult top_k(const Tensor& input, int64_t k, bool largest) {
    if (input.rank() < 1 || input.numel() <= 0) {
        throw std::invalid_argument("top_k: input must be non-empty with rank >= 1");
    }
    const int64_t cols = input.shape().dim(input.rank() - 1);
    if (k < 1 || k > cols) {
        throw std::invalid_argument("top_k: k must be in [1, " + std::to_string(cols) + "], got " +
                                    std::to_string(k));
    }
    // Indices are whole-number floats: exact only up to 2^24.
    if (cols > (int64_t{1} << 24)) {
        throw std::invalid_argument("top_k: last dimension exceeds 2^24, beyond exact float indices");
    }
    const int64_t rows = input.numel() / cols;

    std::vector<int64_t> dims;
    for (int64_t d = 0; d + 1 < input.rank(); ++d) {
        dims.push_back(input.shape().dim(d));
    }
    dims.push_back(k);
    DeviceBackend* backend = input.backend();
    TopKResult result{Tensor(Shape(dims), backend, input.device()), Tensor(Shape(dims), backend, input.device())};
    backend->top_k_rows(input.data(), result.values.data(), result.indices.data(), static_cast<size_t>(rows),
                        static_cast<size_t>(cols), static_cast<size_t>(k), largest);
    return result;
}

}  // namespace pulsatrix
