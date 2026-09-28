#include "pulsatrix/explainer_stability.hpp"

#include <stdexcept>

namespace pulsatrix {

StabilityResult ComputeAttributionStability(const std::vector<Attribution>& repeated_runs) {
    if (repeated_runs.empty()) {
        throw std::invalid_argument("ComputeAttributionStability: repeated_runs must be non-empty");
    }

    int64_t n = repeated_runs.front().values.numel();
    for (const Attribution& run : repeated_runs) {
        if (run.values.numel() != n) {
            throw std::invalid_argument("ComputeAttributionStability: all runs must share the same element count");
        }
    }

    // Exact-equality check first: a variance computation over bit-identical values can still
    // accumulate ~1e-15 floating-point summation-order noise, which would incorrectly report
    // a "deterministic" explainer as having nonzero variance.
    bool all_identical = true;
    for (const Attribution& run : repeated_runs) {
        for (int64_t i = 0; i < n; ++i) {
            if (run.values.data()[i] != repeated_runs.front().values.data()[i]) {
                all_identical = false;
                break;
            }
        }
        if (!all_identical) break;
    }
    if (all_identical) {
        return StabilityResult{0.0f, true};
    }

    float total_variance = 0.0f;
    float run_count = static_cast<float>(repeated_runs.size());
    for (int64_t element = 0; element < n; ++element) {
        float mean = 0.0f;
        for (const Attribution& run : repeated_runs) {
            mean += run.values.data()[element];
        }
        mean /= run_count;

        float variance = 0.0f;
        for (const Attribution& run : repeated_runs) {
            float diff = run.values.data()[element] - mean;
            variance += diff * diff;
        }
        variance /= run_count;
        total_variance += variance;
    }

    return StabilityResult{total_variance / static_cast<float>(n), false};
}

}  // namespace pulsatrix
