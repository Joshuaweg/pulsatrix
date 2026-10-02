#include "pulsatrix/dqn_target.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/shape.hpp"

namespace pulsatrix {
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

// Copies every element of `from` into `into`'s existing buffer, on any devices.
void copy_parameter(const Tensor& from, Tensor& into) {
    const size_t bytes = static_cast<size_t>(from.numel()) * sizeof(float);
    if (bytes == 0 || from.data() == into.data()) {
        return;  // nothing to copy, or a network synced onto itself
    }
    const bool into_is_host = into.device() == DeviceType::Cpu;
    if (from.device() == into.device()) {
        into.backend()->copy(into.data(), from.data(), bytes,
                             into_is_host ? CopyDirection::HostToHost : CopyDirection::DeviceToDevice);
        return;
    }
    const std::vector<float> staged = from.to_host_vector();
    into.backend()->copy(into.data(), staged.data(), bytes,
                         into_is_host ? CopyDirection::HostToHost : CopyDirection::HostToDevice);
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

// One rl_rows(DqnTarget) lane per transition: argmax of the selection row (ties to the lowest
// index, strict >, the same tie rule DQNAgent's argmax uses), evaluated in the evaluation row.
Tensor dqn_target_rows(const Tensor& q_select, const Tensor& q_eval, const Tensor& rewards, const Tensor& dones,
                       float gamma, int64_t batch_size, int64_t action_dim, DeviceBackend* backend) {
    Tensor targets(Shape({batch_size, 1}), backend);
    RlRowArgs args;
    args.in[0] = q_select.data();
    args.in[1] = q_eval.data();
    args.in[2] = rewards.data();
    args.in[3] = dones.data();
    args.out[0] = targets.data();
    args.rows = batch_size;
    args.cols = action_dim;
    args.gamma = gamma;
    backend->rl_rows(RlRowOp::DqnTarget, args);
    return targets;
}

}  // namespace

Tensor ComputeDQNTarget(const Tensor& next_q_target, const Tensor& rewards, const Tensor& dones, float gamma,
                        DeviceBackend* backend) {
    int64_t batch_size = 0;
    int64_t action_dim = 0;
    require_q_block(next_q_target, "ComputeDQNTarget", "next_q_target", batch_size, action_dim);
    require_column(rewards, batch_size, "ComputeDQNTarget", "rewards");
    require_column(dones, batch_size, "ComputeDQNTarget", "dones");
    require_gamma(gamma, "ComputeDQNTarget");

    // Device-generic (GPU-native-kernels Mission 7): the max is the target network's value at
    // its own argmax. (1 - done) zeroes the bootstrapped term *exactly* on a terminal
    // transition: with done == 1.0f the whole product is 0.0f and the target is the bare reward,
    // bit for bit, not merely close to it.
    return dqn_target_rows(next_q_target, next_q_target, rewards, dones, gamma, batch_size, action_dim, backend);
}

Tensor ComputeDoubleDQNTarget(const Tensor& next_q_online, const Tensor& next_q_target, const Tensor& rewards,
                              const Tensor& dones, float gamma, DeviceBackend* backend) {
    int64_t batch_size = 0;
    int64_t action_dim = 0;
    require_q_block(next_q_online, "ComputeDoubleDQNTarget", "next_q_online", batch_size, action_dim);
    if (next_q_target.shape() != next_q_online.shape()) {
        throw std::invalid_argument("ComputeDoubleDQNTarget: next_q_target must have the same shape as next_q_online");
    }
    require_column(rewards, batch_size, "ComputeDoubleDQNTarget", "rewards");
    require_column(dones, batch_size, "ComputeDoubleDQNTarget", "dones");
    require_gamma(gamma, "ComputeDoubleDQNTarget");

    // Selection from the online network, evaluation from the target network -- the whole of
    // van Hasselt et al. 2016. Note which tensor fills which slot.
    return dqn_target_rows(next_q_online, next_q_target, rewards, dones, gamma, batch_size, action_dim, backend);
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
        // Into the *existing* buffer, never `into = from`: destination's parameter Tensors must
        // remain the same objects its own parameters() -- and any optimizer already holding
        // ParamRefs into them -- point at. Device-generic (GPU-native-kernels Mission 7): one
        // buffer copy through the destination's backend, staged through the host only when the
        // two networks live on different devices.
        copy_parameter(from, into);
    }
}

}  // namespace pulsatrix
