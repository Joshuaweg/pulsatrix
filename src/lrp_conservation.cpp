#include "pulsatrix/lrp_conservation.hpp"

#include <cmath>

namespace pulsatrix {

float ConservationResult::delta() const { return std::abs(relevance_in_sum - relevance_out_sum); }

ConservationResult ComputeConservation(const Tensor& relevance_in, const Tensor& relevance_out) {
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
