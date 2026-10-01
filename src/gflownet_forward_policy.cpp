#include "pulsatrix/gflownet_forward_policy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/shape.hpp"

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
    // Numerical Recipes LCG constants -- byte-for-byte CategoricalPolicyAgent::next_unit,
    // deliberately not <random>. See the header note on why this class doesn't just reuse
    // CategoricalPolicyAgent directly (masking).
    lcg_state_ = lcg_state_ * 1664525u + 1013904223u;
    const uint32_t draw = (lcg_state_ >> 8) & 0xFFFFu;
    return static_cast<float>(draw) / 65536.0f;
}

GFlowNetSampledAction GFlowNetForwardPolicy::sample(const Tensor& observation,
                                                     const std::vector<bool>& valid_actions) {
    PULSATRIX_REQUIRE_HOST(observation);

    if (static_cast<int64_t>(valid_actions.size()) != action_dim_) {
        throw std::invalid_argument("GFlowNetForwardPolicy::sample: valid_actions must have size action_dim()");
    }
    if (std::none_of(valid_actions.begin(), valid_actions.end(), [](bool v) { return v; })) {
        throw std::invalid_argument("GFlowNetForwardPolicy::sample: valid_actions must have at least one true entry");
    }

    const Tensor logits = policy_network_->forward(observation);
    if (logits.rank() != 2 || logits.shape().dim(0) != 1 || logits.shape().dim(1) != action_dim_) {
        throw std::invalid_argument("GFlowNetForwardPolicy: policy_network must produce output of shape "
                                     "(1, action_dim)");
    }
    // The network may run on a GPU backend even when observation is host-resident; the raw
    // host loop over logits below would then be UB (GPU-native-kernels campaign,
    // Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(logits);

    // Additive-mask technique: an invalid action's logit is driven to the lowest representable
    // float before softmax, giving it ~0 probability without risking a NaN from actual -infinity
    // arithmetic (unlike CategoricalPolicyAgent, which has no invalid actions to mask).
    std::vector<float> masked_logits(static_cast<size_t>(action_dim_));
    for (int64_t a = 0; a < action_dim_; ++a) {
        masked_logits[static_cast<size_t>(a)] =
            valid_actions[static_cast<size_t>(a)] ? logits.data()[a] : std::numeric_limits<float>::lowest();
    }

    // Numerically stable softmax over the masked logits -- same pattern as
    // CategoricalPolicyAgent::act() / CrossEntropyLoss::forward.
    float max_logit = masked_logits[0];
    for (int64_t a = 1; a < action_dim_; ++a) {
        max_logit = std::max(max_logit, masked_logits[static_cast<size_t>(a)]);
    }
    float exp_sum = 0.0f;
    for (int64_t a = 0; a < action_dim_; ++a) {
        exp_sum += std::exp(masked_logits[static_cast<size_t>(a)] - max_logit);
    }
    const float log_exp_sum = std::log(exp_sum);

    std::vector<float> log_softmax(static_cast<size_t>(action_dim_));
    for (int64_t a = 0; a < action_dim_; ++a) {
        log_softmax[static_cast<size_t>(a)] = masked_logits[static_cast<size_t>(a)] - max_logit - log_exp_sum;
    }

    const float u = next_unit();

    // Clamp default is the *last valid* action, not simply action_dim_-1: the final entry
    // could itself be masked out, and the cumulative sum reaching 1.0 only up to float
    // rounding must never fall through to an invalid action.
    int64_t action = action_dim_ - 1;
    for (int64_t a = action_dim_ - 1; a >= 0; --a) {
        if (valid_actions[static_cast<size_t>(a)]) {
            action = a;
            break;
        }
    }
    // Strict '>', not '>=': next_unit() can return exactly 0.0 (a real, reachable LCG draw,
    // ~1-in-65536 odds) -- with '>=', a u of exactly 0.0 would satisfy `cumulative(0) >= 0` at
    // the very *first* scanned action regardless of its probability, silently selecting a
    // masked (probability ~0) action if it happens to be first. Found via a real training-run
    // crash ("illegal off-grid increment"), not by inspection -- see
    // mission_detailed_balance_loss.md's AAR. Safe to tighten (the pre-computed last-valid-action
    // default above already covers the "cumulative sum never quite reaches 1.0" float-rounding
    // case the original '>=' was chosen for).
    float cumulative = 0.0f;
    for (int64_t a = 0; a < action_dim_; ++a) {
        cumulative += std::exp(log_softmax[static_cast<size_t>(a)]);
        if (cumulative > u) {
            action = a;
            break;
        }
    }

    return GFlowNetSampledAction{Tensor(Shape({1, 1}), backend_, {static_cast<float>(action)}),
                                  log_softmax[static_cast<size_t>(action)]};
}

std::vector<float> GFlowNetForwardPolicy::masked_probs(const Tensor& observation,
                                                        const std::vector<bool>& valid_actions) {
    PULSATRIX_REQUIRE_HOST(observation);

    if (static_cast<int64_t>(valid_actions.size()) != action_dim_) {
        throw std::invalid_argument("GFlowNetForwardPolicy::masked_probs: valid_actions must have size action_dim()");
    }
    if (std::none_of(valid_actions.begin(), valid_actions.end(), [](bool v) { return v; })) {
        throw std::invalid_argument(
            "GFlowNetForwardPolicy::masked_probs: valid_actions must have at least one true entry");
    }

    const Tensor logits = policy_network_->forward(observation);
    if (logits.rank() != 2 || logits.shape().dim(0) != 1 || logits.shape().dim(1) != action_dim_) {
        throw std::invalid_argument("GFlowNetForwardPolicy: policy_network must produce output of shape "
                                     "(1, action_dim)");
    }
    // The network may run on a GPU backend even when observation is host-resident; the raw
    // host loop over logits below would then be UB (GPU-native-kernels campaign,
    // Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(logits);

    std::vector<float> masked_logits(static_cast<size_t>(action_dim_));
    for (int64_t a = 0; a < action_dim_; ++a) {
        masked_logits[static_cast<size_t>(a)] =
            valid_actions[static_cast<size_t>(a)] ? logits.data()[a] : std::numeric_limits<float>::lowest();
    }

    float max_logit = masked_logits[0];
    for (int64_t a = 1; a < action_dim_; ++a) {
        max_logit = std::max(max_logit, masked_logits[static_cast<size_t>(a)]);
    }
    float exp_sum = 0.0f;
    std::vector<float> exp_vals(static_cast<size_t>(action_dim_));
    for (int64_t a = 0; a < action_dim_; ++a) {
        exp_vals[static_cast<size_t>(a)] = std::exp(masked_logits[static_cast<size_t>(a)] - max_logit);
        exp_sum += exp_vals[static_cast<size_t>(a)];
    }

    std::vector<float> probs(static_cast<size_t>(action_dim_));
    for (int64_t a = 0; a < action_dim_; ++a) {
        probs[static_cast<size_t>(a)] = exp_vals[static_cast<size_t>(a)] / exp_sum;
    }
    return probs;
}

}  // namespace pulsatrix
