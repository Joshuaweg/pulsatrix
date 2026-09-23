#include "exai/gae.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "exai/assert.hpp"
#include "exai/shape.hpp"

namespace exai {
namespace {

// Validates a (N, 1) column of per-step scalars. Local to this TU, mirroring
// dqn_target.cpp's own require_column -- deliberately not shared, each is that
// translation unit's private validation detail.
void require_column(const Tensor& tensor, int64_t batch_size, const char* what) {
    if (tensor.rank() != 2 || tensor.shape().dim(0) != batch_size || tensor.shape().dim(1) != 1) {
        throw std::invalid_argument(std::string("ComputeGAE: ") + what + " must have shape (" +
                                    std::to_string(batch_size) + ", 1)");
    }
}

void require_unit_interval(float value, const char* what) {
    if (!(value >= 0.0f && value <= 1.0f)) {
        throw std::invalid_argument(std::string("ComputeGAE: ") + what + " must be in [0, 1]");
    }
}

}  // namespace

GAEResult ComputeGAE(const Tensor& rewards, const Tensor& dones, const Tensor& values, float bootstrap_value,
                     float gamma, float lambda, DeviceBackend* backend) {
    // Raw host loop over Tensor::data() (a reverse-order recursion has no DeviceBackend
    // primitive) -- undefined behavior on a CUDA-backed Tensor. See mission_host_loop_guards.md.
    EXAI_ASSERT(rewards.device() == DeviceType::Cpu);
    EXAI_ASSERT(dones.device() == DeviceType::Cpu);
    EXAI_ASSERT(values.device() == DeviceType::Cpu);

    if (rewards.rank() != 2 || rewards.shape().dim(1) != 1) {
        throw std::invalid_argument("ComputeGAE: rewards must have shape (N, 1)");
    }
    const int64_t batch_size = rewards.shape().dim(0);
    if (batch_size < 1) {
        throw std::invalid_argument("ComputeGAE: rewards must have N >= 1");
    }
    require_column(dones, batch_size, "dones");
    require_column(values, batch_size, "values");
    require_unit_interval(gamma, "gamma");
    require_unit_interval(lambda, "lambda");

    std::vector<float> advantages(static_cast<size_t>(batch_size));
    std::vector<float> returns(static_cast<size_t>(batch_size));

    // One reverse pass. `running` is advantages[t+1]; it starts at 0 because nothing exists
    // past the end of the rollout, and the (1 - done) factor below cuts it to exactly 0 again
    // at every terminal step -- an advantage must never propagate backwards across a terminal
    // state. Same reverse-pass/done-boundary shape as RolloutBuffer::compute_returns(), a
    // different recurrence inside it.
    float running = 0.0f;
    for (int64_t t = batch_size - 1; t >= 0; --t) {
        const float not_done = 1.0f - dones.data()[t];
        // V_next: the next stored step's own value for every step but the last, and the
        // caller-supplied bootstrap for the last -- RolloutBuffer stores no next_observation,
        // so the final successor's value can only come from outside.
        const float next_value = (t + 1 < batch_size) ? values.data()[t + 1] : bootstrap_value;
        const float delta = rewards.data()[t] + gamma * not_done * next_value - values.data()[t];
        running = delta + gamma * lambda * not_done * running;
        advantages[static_cast<size_t>(t)] = running;
        // The GAE identity: the critic regresses onto its own current estimate corrected by
        // the advantage, not onto a raw Monte-Carlo return.
        returns[static_cast<size_t>(t)] = running + values.data()[t];
    }

    return GAEResult{Tensor(Shape({batch_size, 1}), backend, advantages),
                     Tensor(Shape({batch_size, 1}), backend, returns)};
}

}  // namespace exai
