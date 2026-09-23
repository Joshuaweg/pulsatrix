#include "exai/sinusoidal_timestep_embedding.hpp"

#include <cmath>
#include <stdexcept>

#include "exai/shape.hpp"

namespace exai {

Tensor SinusoidalTimestepEmbedding(int64_t t, int64_t embedding_dim, DeviceBackend* backend, float base) {
    if (embedding_dim <= 0) {
        throw std::invalid_argument("SinusoidalTimestepEmbedding: embedding_dim must be positive");
    }
    if (embedding_dim % 2 != 0) {
        throw std::invalid_argument("SinusoidalTimestepEmbedding: embedding_dim must be even");
    }

    Tensor embedding(Shape({1, embedding_dim}), backend);
    const float t_value = static_cast<float>(t);
    for (int64_t i = 0; i < embedding_dim / 2; ++i) {
        // Frequency of pair i: t / base^(2i/embedding_dim). Computed in double and narrowed
        // once, rather than in float: base^(2i/d) spans 1 .. base across the pairs, and the
        // float exponentiation of a large base at a fractional exponent is where the
        // encoding's high-index pairs would lose their (already small) angular resolution.
        const double exponent = static_cast<double>(2 * i) / static_cast<double>(embedding_dim);
        const double frequency = static_cast<double>(t_value) / std::pow(static_cast<double>(base), exponent);
        embedding.data()[2 * i] = static_cast<float>(std::sin(frequency));
        embedding.data()[2 * i + 1] = static_cast<float>(std::cos(frequency));
    }
    return embedding;
}

}  // namespace exai
