#include "exai/categorical_policy_agent.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "exai/assert.hpp"
#include "exai/shape.hpp"

namespace exai {
namespace {

// Validates the constructor's arguments and echoes action_dim back, so it can be called from
// the *initializer list* -- action_dim_ is a member initialized before the constructor body
// would ever get a chance to reject it. Same shape as DQNAgent's validated_action_dim.
int64_t validated_action_dim(Module* policy_network, int64_t action_dim) {
    if (policy_network == nullptr) {
        throw std::invalid_argument("CategoricalPolicyAgent: policy_network must not be null");
    }
    if (action_dim <= 0) {
        throw std::invalid_argument("CategoricalPolicyAgent: action_dim must be >= 1");
    }
    return action_dim;
}

}  // namespace

CategoricalPolicyAgent::CategoricalPolicyAgent(Module* policy_network, int64_t action_dim, DeviceBackend* backend,
                                               uint32_t seed)
    : policy_network_(policy_network),
      action_dim_(validated_action_dim(policy_network, action_dim)),
      backend_(backend),
      lcg_state_(seed) {}

float CategoricalPolicyAgent::next_unit() {
    // Numerical Recipes LCG constants -- byte-for-byte the generator CartPoleEnv::reset(),
    // ReplayBuffer::sample() and DQNAgent::next_unit() already use, deliberately not <random>,
    // whose engine outputs beyond mt19937 are implementation-defined.
    lcg_state_ = lcg_state_ * 1664525u + 1013904223u;
    const uint32_t draw = (lcg_state_ >> 8) & 0xFFFFu;  // [0, 65535]
    // Divided by 65536, not 65535: half-open [0, 1). An inclusive draw could hand the
    // inverse-CDF scan a u == 1.0f that the final cumulative probability -- 1.0 only up to
    // float rounding -- might fail to reach, pushing the scan into its clamp for no reason.
    return static_cast<float>(draw) / 65536.0f;
}

Tensor CategoricalPolicyAgent::policy_logits(const Tensor& observation) {
    Tensor logits = policy_network_->forward(observation);
    if (logits.rank() != 2 || logits.shape().dim(0) != 1 || logits.shape().dim(1) != action_dim_) {
        throw std::invalid_argument("CategoricalPolicyAgent: policy_network must produce output of shape "
                                    "(1, action_dim)");
    }
    return logits;
}

Tensor CategoricalPolicyAgent::act(const Tensor& observation) {
    // The softmax and the inverse-CDF scan below are raw host loops over Tensor::data(); the
    // network output inherits its device from this observation. Undefined behavior on a
    // CUDA-backed Tensor -- see mission_host_loop_guards.md.
    EXAI_ASSERT(observation.device() == DeviceType::Cpu);

    const Tensor logits = policy_logits(observation);

    // Numerically stable softmax: subtract the row max before exponentiating, exactly the
    // pattern CrossEntropyLoss::forward established.
    float max_logit = logits.data()[0];
    for (int64_t a = 1; a < action_dim_; ++a) {
        max_logit = std::max(max_logit, logits.data()[a]);
    }
    float exp_sum = 0.0f;
    for (int64_t a = 0; a < action_dim_; ++a) {
        exp_sum += std::exp(logits.data()[a] - max_logit);
    }
    const float log_exp_sum = std::log(exp_sum);

    std::vector<float> log_softmax(static_cast<size_t>(action_dim_));
    for (int64_t a = 0; a < action_dim_; ++a) {
        log_softmax[static_cast<size_t>(a)] = logits.data()[a] - max_logit - log_exp_sum;
    }

    // The draw is taken unconditionally, before the scan, so the stream advances by exactly one
    // step per act() call no matter which action comes out.
    const float u = next_unit();

    // Inverse-CDF categorical sampling: the smallest index whose cumulative probability reaches
    // u. The final index is the clamp -- the cumulative sum reaches 1.0 only up to float
    // rounding, so a u just below 1 could otherwise fall off the end of the scan. Initializing
    // to the last index and breaking early expresses that clamp without a special case.
    int64_t action = action_dim_ - 1;
    float cumulative = 0.0f;
    for (int64_t a = 0; a < action_dim_; ++a) {
        cumulative += std::exp(log_softmax[static_cast<size_t>(a)]);
        if (cumulative >= u) {
            action = a;
            break;
        }
    }

    // Cached from the stable log-softmax, not re-derived as log(p[action]): one rounding path,
    // and no risk of logging a probability that underflowed to zero.
    last_log_prob_ = log_softmax[static_cast<size_t>(action)];
    has_acted_ = true;

    return Tensor(Shape({1, 1}), backend_, {static_cast<float>(action)});
}

Tensor CategoricalPolicyAgent::act_greedy(const Tensor& observation) {
    EXAI_ASSERT(observation.device() == DeviceType::Cpu);

    const Tensor logits = policy_logits(observation);

    // argmax over logits == argmax over softmax(logits): softmax is strictly monotone, so no
    // exponentiation is needed. Ties resolve to the lowest index (strict >), the same rule
    // DQNAgent::greedy_action uses.
    int64_t best = 0;
    for (int64_t a = 1; a < action_dim_; ++a) {
        if (logits.data()[a] > logits.data()[best]) {
            best = a;
        }
    }
    // Deliberately no last_log_prob_ / has_acted_ update: nothing was sampled here.
    return Tensor(Shape({1, 1}), backend_, {static_cast<float>(best)});
}

float CategoricalPolicyAgent::log_prob() const {
    if (!has_acted_) {
        throw std::logic_error("CategoricalPolicyAgent::log_prob called before act");
    }
    return last_log_prob_;
}

}  // namespace exai
