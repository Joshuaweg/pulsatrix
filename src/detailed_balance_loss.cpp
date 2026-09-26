#include "pulsatrix/detailed_balance_loss.hpp"

#include <stdexcept>

namespace pulsatrix {

float DetailedBalanceLoss::forward(float log_flow_s, float log_pf, float log_flow_s_next, float log_pb) {
    last_delta_ = log_flow_s + log_pf - log_flow_s_next - log_pb;
    has_forwarded_ = true;
    return last_delta_ * last_delta_;
}

void DetailedBalanceLoss::require_forwarded() const {
    if (!has_forwarded_) {
        throw std::logic_error("DetailedBalanceLoss: called before forward");
    }
}

float DetailedBalanceLoss::delta() const {
    require_forwarded();
    return last_delta_;
}

float DetailedBalanceLoss::grad_log_flow_s() const {
    require_forwarded();
    return 2.0f * last_delta_;
}

float DetailedBalanceLoss::grad_log_flow_s_next() const {
    require_forwarded();
    return -2.0f * last_delta_;
}

float DetailedBalanceLoss::grad_weight_for_log_pf() const {
    require_forwarded();
    return -2.0f * last_delta_;
}

}  // namespace pulsatrix
