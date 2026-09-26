#include "pulsatrix/subtb_loss.hpp"

#include <stdexcept>

namespace pulsatrix {

float SubTBLoss::forward(float, float, float, float, float) {
    throw std::logic_error("SubTBLoss::forward not yet implemented");
}

void SubTBLoss::require_forwarded() const {
    throw std::logic_error("SubTBLoss::require_forwarded not yet implemented");
}

float SubTBLoss::delta() const {
    throw std::logic_error("SubTBLoss::delta not yet implemented");
}

float SubTBLoss::grad_log_flow_i() const {
    throw std::logic_error("SubTBLoss::grad_log_flow_i not yet implemented");
}

float SubTBLoss::grad_log_flow_j() const {
    throw std::logic_error("SubTBLoss::grad_log_flow_j not yet implemented");
}

float SubTBLoss::grad_weight_for_log_pf_range() const {
    throw std::logic_error("SubTBLoss::grad_weight_for_log_pf_range not yet implemented");
}

}  // namespace pulsatrix
