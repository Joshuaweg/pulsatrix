#include "pulsatrix/gflownet_forward_policy.hpp"

#include <stdexcept>

namespace pulsatrix {
namespace {

int64_t validated_action_dim(Module* policy_network, int64_t action_dim) {
    if (policy_network == nullptr) {
        throw std::invalid_argument("GFlowNetForwardPolicy: policy_network must not be null");
    }
    if (action_dim <= 0) {
        throw std::invalid_argument("GFlowNetForwardPolicy: action_dim must be >= 1");
    }
    return action_dim;
}

}  // namespace

GFlowNetForwardPolicy::GFlowNetForwardPolicy(Module* policy_network, int64_t action_dim, DeviceBackend* backend,
                                              uint32_t seed)
    : policy_network_(policy_network),
      action_dim_(validated_action_dim(policy_network, action_dim)),
      backend_(backend),
      lcg_state_(seed) {}

float GFlowNetForwardPolicy::next_unit() {
    throw std::logic_error("GFlowNetForwardPolicy::next_unit not yet implemented");
}

GFlowNetSampledAction GFlowNetForwardPolicy::sample(const Tensor&, const std::vector<bool>&) {
    throw std::logic_error("GFlowNetForwardPolicy::sample not yet implemented");
}

}  // namespace pulsatrix
