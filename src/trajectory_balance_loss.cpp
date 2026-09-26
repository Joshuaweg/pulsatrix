#include "pulsatrix/trajectory_balance_loss.hpp"

#include <stdexcept>

namespace pulsatrix {

float TrajectoryBalanceLoss::forward(float sum_log_pf, float sum_log_pb, float log_reward, float log_z) {
    last_delta_ = log_z + sum_log_pf - log_reward - sum_log_pb;
    has_forwarded_ = true;
    return last_delta_ * last_delta_;
}

void TrajectoryBalanceLoss::require_forwarded() const {
    if (!has_forwarded_) {
        throw std::logic_error("TrajectoryBalanceLoss: called before forward");
    }
}

float TrajectoryBalanceLoss::delta() const {
    require_forwarded();
    return last_delta_;
}

float TrajectoryBalanceLoss::grad_log_z() const {
    require_forwarded();
    return 2.0f * last_delta_;
}

float TrajectoryBalanceLoss::grad_weight_for_log_pf() const {
    require_forwarded();
    return -2.0f * last_delta_;
}

}  // namespace pulsatrix
