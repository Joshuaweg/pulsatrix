// A portable random generator for results that must match across platforms (std::
// distributions differ between libstdc++ and MSVC). SplitMix64 for bits; Box-Muller for
// Gaussians. Private to src/.
#pragma once

#include <cmath>
#include <cstdint>

namespace pulsatrix {

struct PortableRng {
    uint64_t state;

    uint64_t next() {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    /** @brief Uniform in (0, 1), never exactly 0 (so it's safe for log). */
    double uniform() { return (static_cast<double>(next() >> 11) + 0.5) / static_cast<double>(1ULL << 53); }

    double gaussian() {
        constexpr double kTwoPi = 6.28318530717958647692;
        return std::sqrt(-2.0 * std::log(uniform())) * std::cos(kTwoPi * uniform());
    }
};

}  // namespace pulsatrix
