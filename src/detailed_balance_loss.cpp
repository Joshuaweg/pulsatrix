#include "pulsatrix/detailed_balance_loss.hpp"

#include <stdexcept>

namespace pulsatrix {

float DetailedBalanceLoss::forward(float, float, float, float) {
    throw std::logic_error("DetailedBalanceLoss::forward not yet implemented");
}

void DetailedBalanceLoss::require_forwarded() const {
    throw std::logic_error("DetailedBalanceLoss::require_forwarded not yet implemented");
}

float DetailedBalanceLoss::delta() const {
    throw std::logic_error("DetailedBalanceLoss::delta not yet implemented");
}

float DetailedBalanceLoss::grad_log_flow_s() const {
    throw std::logic_error("DetailedBalanceLoss::grad_log_flow_s not yet implemented");
}

float DetailedBalanceLoss::grad_log_flow_s_next() const {
    throw std::logic_error("DetailedBalanceLoss::grad_log_flow_s_next not yet implemented");
}

float DetailedBalanceLoss::grad_weight_for_log_pf() const {
    throw std::logic_error("DetailedBalanceLoss::grad_weight_for_log_pf not yet implemented");
}

}  // namespace pulsatrix
