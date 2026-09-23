#include "exai/dqn_target.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "exai/assert.hpp"
#include "exai/shape.hpp"

namespace exai {
namespace {

// Renders a Shape as "(a, b, ...)" for an error message. Local to this TU: the only place in
// the codebase that currently needs to *name* a mismatched shape rather than just reject it.
std::string shape_to_string(const Shape& shape) {
    std::string out = "(";
    for (int64_t i = 0; i < shape.rank(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(shape.dim(static_cast<size_t>(i)));
    }
    return out + ")";
}

// Validates a (N, action_dim) Q-value block and returns its dimensions.
void require_q_block(const Tensor& tensor, const char* function_name, const char* what, int64_t& batch_size,
                     int64_t& action_dim) {
    if (tensor.rank() != 2) {
        throw std::invalid_argument(std::string(function_name) + ": " + what +
                                    " must have shape (N, action_dim)");
    }
    batch_size = tensor.shape().dim(0);
    action_dim = tensor.shape().dim(1);
    if (batch_size < 1 || action_dim < 1) {
        throw std::invalid_argument(std::string(function_name) + ": " + what +
                                    " must have N >= 1 and action_dim >= 1");
    }
}

// Validates a (N, 1) column of per-transition scalars.
void require_column(const Tensor& tensor, int64_t batch_size, const char* function_name, const char* what) {
    if (tensor.rank() != 2 || tensor.shape().dim(0) != batch_size || tensor.shape().dim(1) != 1) {
        throw std::invalid_argument(std::string(function_name) + ": " + what + " must have shape (" +
                                    std::to_string(batch_size) + ", 1)");
    }
}

void require_gamma(float gamma, const char* function_name) {
    if (!(gamma >= 0.0f && gamma <= 1.0f)) {
        throw std::invalid_argument(std::string(function_name) + ": gamma must be in [0, 1]");
    }
}

// Index of the largest element in row `b` of a (N, action_dim) block. Ties resolve to the
// lowest index (strict >), the same tie rule DQNAgent's argmax uses.
int64_t argmax_in_row(const Tensor& block, int64_t b, int64_t action_dim) {
    const float* row = block.data() + b * action_dim;
    int64_t best = 0;
    for (int64_t a = 1; a < action_dim; ++a) {
        if (row[a] > row[best]) {
            best = a;
        }
    }
    return best;
}

}  // namespace

Tensor ComputeDQNTarget(const Tensor& next_q_target, const Tensor& rewards, const Tensor& dones, float gamma,
                        DeviceBackend* backend) {
    // Raw host loop over Tensor::data() (a row-wise max has no DeviceBackend primitive) --
    // undefined behavior on a CUDA-backed Tensor. See mission_host_loop_guards.md.
    EXAI_ASSERT(next_q_target.device() == DeviceType::Cpu);
    EXAI_ASSERT(rewards.device() == DeviceType::Cpu);
    EXAI_ASSERT(dones.device() == DeviceType::Cpu);

    int64_t batch_size = 0;
    int64_t action_dim = 0;
    require_q_block(next_q_target, "ComputeDQNTarget", "next_q_target", batch_size, action_dim);
    require_column(rewards, batch_size, "ComputeDQNTarget", "rewards");
    require_column(dones, batch_size, "ComputeDQNTarget", "dones");
    require_gamma(gamma, "ComputeDQNTarget");

    std::vector<float> targets(static_cast<size_t>(batch_size));
    for (int64_t b = 0; b < batch_size; ++b) {
        const float best_value = next_q_target.data()[b * action_dim + argmax_in_row(next_q_target, b, action_dim)];
        // (1 - done) zeroes the bootstrapped term *exactly* on a terminal transition: with
        // done == 1.0f the whole product is 0.0f and the target is the bare reward, bit for
        // bit, not merely close to it.
        const float bootstrap = gamma * (1.0f - dones.data()[b]) * best_value;
        targets[static_cast<size_t>(b)] = rewards.data()[b] + bootstrap;
    }
    return Tensor(Shape({batch_size, 1}), backend, targets);
}

Tensor ComputeDoubleDQNTarget(const Tensor& next_q_online, const Tensor& next_q_target, const Tensor& rewards,
                              const Tensor& dones, float gamma, DeviceBackend* backend) {
    EXAI_ASSERT(next_q_online.device() == DeviceType::Cpu);
    EXAI_ASSERT(next_q_target.device() == DeviceType::Cpu);
    EXAI_ASSERT(rewards.device() == DeviceType::Cpu);
    EXAI_ASSERT(dones.device() == DeviceType::Cpu);

    int64_t batch_size = 0;
    int64_t action_dim = 0;
    require_q_block(next_q_online, "ComputeDoubleDQNTarget", "next_q_online", batch_size, action_dim);
    if (next_q_target.shape() != next_q_online.shape()) {
        throw std::invalid_argument("ComputeDoubleDQNTarget: next_q_target must have the same shape as next_q_online");
    }
    require_column(rewards, batch_size, "ComputeDoubleDQNTarget", "rewards");
    require_column(dones, batch_size, "ComputeDoubleDQNTarget", "dones");
    require_gamma(gamma, "ComputeDoubleDQNTarget");

    std::vector<float> targets(static_cast<size_t>(batch_size));
    for (int64_t b = 0; b < batch_size; ++b) {
        // Selection from the online network, evaluation from the target network -- the whole
        // of van Hasselt et al. 2016. Note which tensor each of the two lines reads.
        const int64_t best_action = argmax_in_row(next_q_online, b, action_dim);
        const float evaluated = next_q_target.data()[b * action_dim + best_action];
        const float bootstrap = gamma * (1.0f - dones.data()[b]) * evaluated;
        targets[static_cast<size_t>(b)] = rewards.data()[b] + bootstrap;
    }
    return Tensor(Shape({batch_size, 1}), backend, targets);
}

void SyncTargetNetwork(Module& source, Module& destination) {
    std::vector<ParamRef> source_params = source.parameters();
    std::vector<ParamRef> destination_params = destination.parameters();

    if (source_params.size() != destination_params.size()) {
        throw std::invalid_argument("SyncTargetNetwork: source exposes " + std::to_string(source_params.size()) +
                                    " parameters but destination exposes " +
                                    std::to_string(destination_params.size()) +
                                    " -- the two networks have different architectures");
    }

    for (size_t i = 0; i < source_params.size(); ++i) {
        const Tensor& from = *source_params[i].value;
        Tensor& into = *destination_params[i].value;
        if (from.shape() != into.shape()) {
            throw std::invalid_argument("SyncTargetNetwork: parameter " + std::to_string(i) +
                                        " has shape " + shape_to_string(from.shape()) + " in source but " +
                                        shape_to_string(into.shape()) + " in destination");
        }
        // Element-wise into the *existing* buffer, never `into = from`: destination's
        // parameter Tensors must remain the same objects its own parameters() -- and any
        // optimizer already holding ParamRefs into them -- point at.
        for (int64_t e = 0; e < from.numel(); ++e) {
            into.data()[e] = from.data()[e];
        }
    }
}

}  // namespace exai
