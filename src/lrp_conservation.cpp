#include "pulsatrix/lrp_conservation.hpp"

#include <cmath>

namespace pulsatrix {

float ConservationResult::delta() const { return std::abs(relevance_in_sum - relevance_out_sum); }

ConservationResult ComputeConservation(const Tensor& relevance_in, const Tensor& relevance_out) {
    // Dereferences Tensor::data() directly in a raw host loop -- not yet backend-generic
    // (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(relevance_in);
    PULSATRIX_REQUIRE_HOST(relevance_out);

    float sum_in = 0.0f;
    for (int64_t i = 0; i < relevance_in.numel(); ++i) {
        sum_in += relevance_in.data()[i];
    }
    float sum_out = 0.0f;
    for (int64_t i = 0; i < relevance_out.numel(); ++i) {
        sum_out += relevance_out.data()[i];
    }
    return ConservationResult{sum_in, sum_out};
}

}  // namespace pulsatrix
