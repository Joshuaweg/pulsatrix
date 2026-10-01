#include "pulsatrix/satisfaction_loss.hpp"

#include <stdexcept>

namespace pulsatrix {

SatisfactionLoss::SatisfactionLoss(DeviceBackend* backend, float p) : backend_(backend), aggregator_(backend, p) {}

float SatisfactionLoss::forward(const Tensor& truth_values) {
    if (truth_values.rank() != 1) {
        throw std::invalid_argument(
            "SatisfactionLoss::forward: truth_values must be rank 1 (a single formula's per-grounding truth "
            "degrees)");
    }
    Tensor sat = aggregator_.forward(truth_values);  // rank 0 (a scalar), numel() == 1
    // Reads Tensor::data() directly on the host -- not yet backend-generic
    // (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(sat);
    has_forwarded_ = true;
    return 1.0f - sat.data()[0];
}

Tensor SatisfactionLoss::backward() {
    if (!has_forwarded_) {
        throw std::logic_error("SatisfactionLoss::backward: called before any forward()");
    }
    Tensor grad_sat(Shape({}), backend_, {-1.0f});
    return aggregator_.backward(grad_sat);
}

}  // namespace pulsatrix
