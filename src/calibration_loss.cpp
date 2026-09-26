#include "pulsatrix/calibration_loss.hpp"

#include <stdexcept>

namespace pulsatrix {

CalibrationLoss::CalibrationLoss(DeviceBackend* backend)
    : backend_(backend), last_probs_(Shape({0}), backend) {}

float CalibrationLoss::forward(const Tensor&, const Tensor&) {
    throw std::logic_error("CalibrationLoss::forward not yet implemented");
}

Tensor CalibrationLoss::backward() const {
    throw std::logic_error("CalibrationLoss::backward not yet implemented");
}

}  // namespace pulsatrix
