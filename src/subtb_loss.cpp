#include "pulsatrix/subtb_loss.hpp"

#include <stdexcept>

namespace pulsatrix {

float SubTBLoss::forward(float log_flow_i, float sum_log_pf, float log_flow_j, float sum_log_pb,
                          float pair_weight_ratio) {
    last_delta_ = log_flow_i + sum_log_pf - log_flow_j - sum_log_pb;
    last_pair_weight_ratio_ = pair_weight_ratio;
    has_forwarded_ = true;
    return pair_weight_ratio * last_delta_ * last_delta_;
}

void SubTBLoss::require_forwarded() const {
    if (!has_forwarded_) {
        throw std::logic_error("SubTBLoss: called before forward");
    }
}

float SubTBLoss::delta() const {
    require_forwarded();
    return last_delta_;
}

float SubTBLoss::grad_log_flow_i() const {
    require_forwarded();
    return last_pair_weight_ratio_ * 2.0f * last_delta_;
}

float SubTBLoss::grad_log_flow_j() const {
    require_forwarded();
    return last_pair_weight_ratio_ * -2.0f * last_delta_;
}

float SubTBLoss::grad_weight_for_log_pf_range() const {
    require_forwarded();
    return last_pair_weight_ratio_ * -2.0f * last_delta_;
}

}  // namespace pulsatrix
