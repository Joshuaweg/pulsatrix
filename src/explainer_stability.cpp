#include "pulsatrix/explainer_stability.hpp"

#include <stdexcept>
#include <vector>

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

    // A diagnostic over a handful of attribution maps: copy each to the host once (any
    // device) and run the statistics there (GPU-native-kernels Mission 3).
    std::vector<std::vector<float>> values;
    values.reserve(repeated_runs.size());
    for (const Attribution& run : repeated_runs) {
        std::vector<float> host(static_cast<size_t>(n));
        run.values.backend()->copy(host.data(), run.values.data(), host.size() * sizeof(float),
                                   run.values.device() == DeviceType::Cpu ? CopyDirection::HostToHost
                                                                         : CopyDirection::DeviceToHost);
        values.push_back(std::move(host));
    }

    // Exact-equality check first: a variance computation over bit-identical values can still
    // accumulate ~1e-15 floating-point summation-order noise, which would incorrectly report
    // a "deterministic" explainer as having nonzero variance.
    bool all_identical = true;
    for (const std::vector<float>& run : values) {
        for (int64_t i = 0; i < n; ++i) {
            if (run[static_cast<size_t>(i)] != values.front()[static_cast<size_t>(i)]) {
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
        for (const std::vector<float>& run : values) {
            mean += run[static_cast<size_t>(element)];
        }
        mean /= run_count;

        float variance = 0.0f;
        for (const std::vector<float>& run : values) {
            float diff = run[static_cast<size_t>(element)] - mean;
            variance += diff * diff;
        }
        variance /= run_count;
        total_variance += variance;
    }

    return StabilityResult{total_variance / static_cast<float>(n), false};
}

}  // namespace pulsatrix
