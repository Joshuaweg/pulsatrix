#include "pulsatrix/dqn_agent.hpp"

#include <stdexcept>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/shape.hpp"

namespace pulsatrix {
namespace {

// Validates the constructor's scalar arguments and echoes action_dim back, so it can be
// called from the *initializer list* -- epsilon_ and action_dim_ are members initialized
// before the constructor body would ever get a chance to reject them.
int64_t validated_action_dim(Module* q_network, int64_t action_dim, float epsilon) {
    if (q_network == nullptr) {
        throw std::invalid_argument("DQNAgent: q_network must not be null");
    }
    if (action_dim <= 0) {
        throw std::invalid_argument("DQNAgent: action_dim must be >= 1");
    }
    if (!(epsilon >= 0.0f && epsilon <= 1.0f)) {
        throw std::invalid_argument("DQNAgent: epsilon must be in [0, 1]");
    }
    return action_dim;
}

}  // namespace

DQNAgent::DQNAgent(Module* q_network, int64_t action_dim, float epsilon, DeviceBackend* backend, uint32_t seed)
    : q_network_(q_network),
      action_dim_(validated_action_dim(q_network, action_dim, epsilon)),
      epsilon_(epsilon),
      backend_(backend),
      lcg_state_(seed) {}

float DQNAgent::next_unit() {
    // Numerical Recipes LCG constants -- byte-for-byte the generator CartPoleEnv::reset() and
    // ReplayBuffer::sample() already use, deliberately not <random>, whose engine outputs
    // beyond mt19937 are implementation-defined.
    lcg_state_ = lcg_state_ * 1664525u + 1013904223u;
    const uint32_t draw = (lcg_state_ >> 8) & 0xFFFFu;  // [0, 65535]
    // Divided by 65536, not 65535: this must be a *half-open* [0, 1) draw. Dividing by 65535
    // (CartPoleEnv's choice, which wants an inclusive interval to centre on zero) would let
    // `u == 1.0f` occur, and `u < epsilon` would then be false on that draw even at
    // epsilon == 1.0 -- an exploration policy that occasionally acts greedily despite being
    // told to explore always. The epsilon == 1 boundary test would catch it, eventually.
    return static_cast<float>(draw) / 65536.0f;
}

int64_t DQNAgent::next_index(int64_t bound) {
    lcg_state_ = lcg_state_ * 1664525u + 1013904223u;
    const int64_t draw = static_cast<int64_t>((lcg_state_ >> 8) & 0xFFFFu);  // [0, 65535]
    // Modulo folding of a 16-bit draw, exactly as ReplayBuffer::next_index does: the residual
    // bias is bounded by bound/65536 and is irrelevant for exploration, while rejection
    // sampling would make the number of LCG steps content-dependent and the stream far harder
    // to reason about in a determinism test.
    return draw % bound;
}

Tensor DQNAgent::greedy_action(const Tensor& observation) {
    const Tensor q_values = q_network_->forward(observation);
    if (q_values.rank() != 2 || q_values.shape().dim(0) != 1 || q_values.shape().dim(1) != action_dim_) {
        throw std::invalid_argument("DQNAgent: q_network must produce output of shape (1, action_dim)");
    }

    // Ties resolve to the lowest index (strict >), the same rule ComputeDQNTarget and
    // ComputeDoubleDQNTarget use, so policy and target agree on a tied row by construction.
    int64_t best = 0;
    for (int64_t a = 1; a < action_dim_; ++a) {
        if (q_values.data()[a] > q_values.data()[best]) {
            best = a;
        }
    }
    return Tensor(Shape({1, 1}), backend_, {static_cast<float>(best)});
}

Tensor DQNAgent::act(const Tensor& observation) {
    // The argmax below is a raw host loop over Tensor::data(); the network output inherits
    // its device from this observation. Undefined behavior on a CUDA-backed Tensor -- see
    // mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(observation);

    // The coin flip is drawn unconditionally, before the branch, so the stream advances by a
    // known amount regardless of which way it goes.
    const float u = next_unit();
    if (u < epsilon_) {
        return Tensor(Shape({1, 1}), backend_, {static_cast<float>(next_index(action_dim_))});
    }
    return greedy_action(observation);
}

Tensor DQNAgent::act_greedy(const Tensor& observation) {
    PULSATRIX_REQUIRE_HOST(observation);
    return greedy_action(observation);
}

void DQNAgent::set_epsilon(float epsilon) {
    if (!(epsilon >= 0.0f && epsilon <= 1.0f)) {
        throw std::invalid_argument("DQNAgent::set_epsilon: epsilon must be in [0, 1]");
    }
    epsilon_ = epsilon;
}

}  // namespace pulsatrix
