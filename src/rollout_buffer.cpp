#include "exai/rollout_buffer.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "exai/assert.hpp"
#include "exai/shape.hpp"

namespace exai {
namespace {

// Validates the three dimension arguments and echoes the first one back, so it can be called
// from the constructor's *initializer list*. The storage Tensors are members and are
// therefore constructed before the constructor body ever runs; a negative max_length would
// reach Shape's own "dimensions must be non-negative" throw first, reporting the wrong error
// from the wrong class. Validating inside the initializer list keeps RolloutBuffer's own
// message authoritative. Same shape as ReplayBuffer's validated_capacity().
int64_t validated_max_length(int64_t max_length, int64_t observation_dim, int64_t action_dim) {
    if (max_length <= 0) {
        throw std::invalid_argument("RolloutBuffer: max_length must be >= 1");
    }
    if (observation_dim <= 0) {
        throw std::invalid_argument("RolloutBuffer: observation_dim must be >= 1");
    }
    if (action_dim <= 0) {
        throw std::invalid_argument("RolloutBuffer: action_dim must be >= 1");
    }
    return max_length;
}

// Rejects anything that is not exactly a (1, expected_width) row.
void require_row_shape(const Tensor& tensor, int64_t expected_width, const char* what) {
    if (tensor.rank() != 2 || tensor.shape().dim(0) != 1 || tensor.shape().dim(1) != expected_width) {
        throw std::invalid_argument(std::string("RolloutBuffer::add: ") + what + " must have shape (1, " +
                                    std::to_string(expected_width) + ")");
    }
}

// Copies one (1, width) row into row `row` of a (max_length, width) storage block.
void write_row(Tensor& storage, int64_t row, const Tensor& source, int64_t width) {
    float* destination = storage.data() + row * width;
    const float* values = source.data();
    for (int64_t i = 0; i < width; ++i) {
        destination[i] = values[i];
    }
}

}  // namespace

RolloutBuffer::RolloutBuffer(int64_t max_length, int64_t observation_dim, int64_t action_dim, DeviceBackend* backend)
    : max_length_(validated_max_length(max_length, observation_dim, action_dim)),
      observation_dim_(observation_dim),
      action_dim_(action_dim),
      backend_(backend),
      observations_(Shape({max_length, observation_dim}), backend),
      actions_(Shape({max_length, action_dim}), backend),
      rewards_(Shape({max_length, 1}), backend),
      log_probs_(Shape({max_length, 1}), backend),
      dones_(Shape({max_length, 1}), backend) {}

void RolloutBuffer::add(const Tensor& observation, const Tensor& action, float reward, float log_prob, bool done) {
    // Raw host-loop row copies over Tensor::data() -- undefined behavior on a CUDA-backed
    // Tensor. See mission_host_loop_guards.md; same guard as every prior host-loop site.
    EXAI_ASSERT(observation.device() == DeviceType::Cpu);
    EXAI_ASSERT(action.device() == DeviceType::Cpu);

    // Before the shape checks, because it is the stronger statement: when the rollout is
    // full, *no* add() can be correct, whatever the argument shapes are. Reporting a shape
    // problem first would point the caller at the wrong fix.
    if (size_ >= max_length_) {
        throw std::logic_error("RolloutBuffer::add: rollout is already at max_length (" +
                               std::to_string(max_length_) +
                               "); call compute_returns() and clear() before collecting the next rollout");
    }

    require_row_shape(observation, observation_dim_, "observation");
    require_row_shape(action, action_dim_, "action");

    write_row(observations_, size_, observation, observation_dim_);
    write_row(actions_, size_, action, action_dim_);
    rewards_.data()[size_] = reward;
    log_probs_.data()[size_] = log_prob;
    dones_.data()[size_] = done ? 1.0f : 0.0f;

    ++size_;
}

RolloutBatch RolloutBuffer::compute_returns(float gamma) const {
    if (!(gamma > 0.0f) || gamma > 1.0f) {
        // Written as !(gamma > 0) rather than gamma <= 0 so a NaN gamma is rejected too:
        // every comparison against NaN is false, so a NaN would slip through `gamma <= 0 ||
        // gamma > 1` and silently poison every return.
        throw std::invalid_argument("RolloutBuffer::compute_returns: gamma must be in (0, 1]");
    }

    std::vector<float> observations(static_cast<size_t>(size_ * observation_dim_));
    std::vector<float> actions(static_cast<size_t>(size_ * action_dim_));
    std::vector<float> returns(static_cast<size_t>(size_));
    std::vector<float> log_probs(static_cast<size_t>(size_));

    // Single reverse pass: G_t = r_t + gamma * G_{t+1}, which is the return-to-go definition
    // sum_{k=t}^{T-1} gamma^(k-t) r_k unrolled, in O(T) rather than the O(T^2) of summing
    // each suffix independently.
    float running_return = 0.0f;
    for (int64_t t = size_ - 1; t >= 0; --t) {
        if (dones_.data()[t] != 0.0f) {
            // Episode boundary: step t is terminal, so there is no G_{t+1} to discount into
            // it -- the successor step belongs to the *next* episode. Reset before adding
            // r_t, never after, or r_t itself would be discarded. Dropping this reset is the
            // classic silent bug here: episode k+1's return leaks backwards across a
            // terminal state into episode k's last steps, and every value in the first
            // episode comes out too large while the test "returns decrease" still passes.
            running_return = 0.0f;
        }
        running_return = rewards_.data()[t] + gamma * running_return;
        returns[static_cast<size_t>(t)] = running_return;
        log_probs[static_cast<size_t>(t)] = log_probs_.data()[t];
    }

    // Straight prefix copies, not per-row gathers: unlike ReplayBuffer::sample(), which
    // gathers scattered random indices, a rollout is handed back in stored order, so rows
    // [0, size_) are already contiguous in the storage block.
    for (int64_t i = 0; i < size_ * observation_dim_; ++i) {
        observations[static_cast<size_t>(i)] = observations_.data()[i];
    }
    for (int64_t i = 0; i < size_ * action_dim_; ++i) {
        actions[static_cast<size_t>(i)] = actions_.data()[i];
    }

    return RolloutBatch{Tensor(Shape({size_, observation_dim_}), backend_, observations),
                        Tensor(Shape({size_, action_dim_}), backend_, actions),
                        Tensor(Shape({size_, 1}), backend_, returns),
                        Tensor(Shape({size_, 1}), backend_, log_probs)};
}

void RolloutBuffer::clear() {
    // Length only. Every read path (compute_returns, and add()'s write index) is bounded by
    // size_, so the stale rows beyond it are unreachable; zero-filling max_length rows on
    // every rollout would be work no observer can detect.
    size_ = 0;
}

}  // namespace exai
