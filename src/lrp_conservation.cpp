#include "pulsatrix/lrp_conservation.hpp"

#include <cmath>

namespace pulsatrix {

float ConservationResult::delta() const { return std::abs(relevance_in_sum - relevance_out_sum); }

ConservationResult ComputeConservation(const Tensor& relevance_in, const Tensor& relevance_out) {
    // Device-generic (GPU-native-kernels Mission 3): CPUBackend::sum adds in index order, as
    // the original loops did; only the two scalars reach the host.
    const float sum_in = relevance_in.backend()->sum(relevance_in.data(), static_cast<size_t>(relevance_in.numel()));
    const float sum_out =
        relevance_out.backend()->sum(relevance_out.data(), static_cast<size_t>(relevance_out.numel()));
    return ConservationResult{sum_in, sum_out};
}

}  // namespace pulsatrix
