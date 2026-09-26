#include "pulsatrix/trajectory_balance_loss.hpp"

#include <stdexcept>

namespace pulsatrix {

float TrajectoryBalanceLoss::forward(float, float, float, float) {
    throw std::logic_error("TrajectoryBalanceLoss::forward not yet implemented");
}

void TrajectoryBalanceLoss::require_forwarded() const {
    throw std::logic_error("TrajectoryBalanceLoss::require_forwarded not yet implemented");
}

float TrajectoryBalanceLoss::delta() const {
    throw std::logic_error("TrajectoryBalanceLoss::delta not yet implemented");
}

float TrajectoryBalanceLoss::grad_log_z() const {
    throw std::logic_error("TrajectoryBalanceLoss::grad_log_z not yet implemented");
}

float TrajectoryBalanceLoss::grad_weight_for_log_pf() const {
    throw std::logic_error("TrajectoryBalanceLoss::grad_weight_for_log_pf not yet implemented");
}

}  // namespace pulsatrix
