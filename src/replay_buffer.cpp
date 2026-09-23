#include "exai/replay_buffer.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "exai/assert.hpp"
#include "exai/shape.hpp"

namespace exai {
namespace {

// Validates the three dimension arguments and echoes the first one back, so it can be
// called from the constructor's *initializer list*. The storage Tensors are members and are
// therefore constructed before the constructor body ever runs; a negative capacity would
// reach Shape's own "dimensions must be non-negative" throw first, reporting the wrong error
// from the wrong class. Validating inside the initializer list keeps ReplayBuffer's own
// message authoritative.
int64_t validated_capacity(int64_t capacity, int64_t observation_dim, int64_t action_dim) {
    if (capacity <= 0) {
        throw std::invalid_argument("ReplayBuffer: capacity must be >= 1");
    }
    if (observation_dim <= 0) {
        throw std::invalid_argument("ReplayBuffer: observation_dim must be >= 1");
    }
    if (action_dim <= 0) {
        throw std::invalid_argument("ReplayBuffer: action_dim must be >= 1");
    }
    return capacity;
}

// Rejects anything that is not exactly a (1, expected_width) row.
void require_row_shape(const Tensor& tensor, int64_t expected_width, const char* what) {
    if (tensor.rank() != 2 || tensor.shape().dim(0) != 1 || tensor.shape().dim(1) != expected_width) {
        throw std::invalid_argument(std::string("ReplayBuffer::add: ") + what + " must have shape (1, " +
                                    std::to_string(expected_width) + ")");
    }
}

// Copies one (1, width) row into row `row` of a (capacity, width) storage block.
void write_row(Tensor& storage, int64_t row, const Tensor& source, int64_t width) {
    float* destination = storage.data() + row * width;
    const float* values = source.data();
    for (int64_t i = 0; i < width; ++i) {
        destination[i] = values[i];
    }
}

// Copies row `row` of a (capacity, width) storage block into `out` at offset `out_row`.
void gather_row(std::vector<float>& out, int64_t out_row, const Tensor& storage, int64_t row, int64_t width) {
    const float* source = storage.data() + row * width;
    float* destination = out.data() + out_row * width;
    for (int64_t i = 0; i < width; ++i) {
        destination[i] = source[i];
    }
}

}  // namespace

ReplayBuffer::ReplayBuffer(int64_t capacity, int64_t observation_dim, int64_t action_dim, DeviceBackend* backend,
                           uint32_t seed)
    : capacity_(validated_capacity(capacity, observation_dim, action_dim)),
      observation_dim_(observation_dim),
      action_dim_(action_dim),
      backend_(backend),
      lcg_state_(seed),
      observations_(Shape({capacity, observation_dim}), backend),
      actions_(Shape({capacity, action_dim}), backend),
      rewards_(Shape({capacity, 1}), backend),
      next_observations_(Shape({capacity, observation_dim}), backend),
      dones_(Shape({capacity, 1}), backend) {}

int64_t ReplayBuffer::next_index(int64_t bound) {
    // Numerical Recipes LCG constants -- byte-for-byte the generator CartPoleEnv::reset()
    // and this codebase's reproducible-randomness tests already use, deliberately not
    // <random>, whose engine outputs beyond mt19937 are implementation-defined.
    lcg_state_ = lcg_state_ * 1664525u + 1013904223u;
    const int64_t draw = static_cast<int64_t>((lcg_state_ >> 8) & 0xFFFFu);  // [0, 65535]
    // Modulo folding of a 16-bit draw. The residual bias is bounded by bound/65536 and is
    // irrelevant for replay sampling (an epsilon-scale non-uniformity in which stored
    // transition is revisited), while the alternative -- rejection sampling -- would make
    // the number of LCG steps content-dependent and the stream harder to reason about in a
    // determinism test. Same low-bit-discarding shift CartPoleEnv uses.
    return draw % bound;
}

void ReplayBuffer::add(const Tensor& observation, const Tensor& action, float reward, const Tensor& next_observation,
                       bool done) {
    // Raw host-loop row copies over Tensor::data() -- undefined behavior on a CUDA-backed
    // Tensor. See mission_host_loop_guards.md; same guard as every prior host-loop site.
    EXAI_ASSERT(observation.device() == DeviceType::Cpu);
    EXAI_ASSERT(action.device() == DeviceType::Cpu);
    EXAI_ASSERT(next_observation.device() == DeviceType::Cpu);

    require_row_shape(observation, observation_dim_, "observation");
    require_row_shape(action, action_dim_, "action");
    require_row_shape(next_observation, observation_dim_, "next_observation");

    write_row(observations_, write_index_, observation, observation_dim_);
    write_row(actions_, write_index_, action, action_dim_);
    write_row(next_observations_, write_index_, next_observation, observation_dim_);
    rewards_.data()[write_index_] = reward;
    dones_.data()[write_index_] = done ? 1.0f : 0.0f;

    write_index_ = (write_index_ + 1) % capacity_;
    if (size_ < capacity_) {
        ++size_;
    }
}

ReplayBatch ReplayBuffer::sample(int64_t batch_size) {
    if (batch_size <= 0) {
        throw std::invalid_argument("ReplayBuffer::sample: batch_size must be >= 1");
    }
    if (batch_size > size_) {
        throw std::invalid_argument("ReplayBuffer::sample: batch_size (" + std::to_string(batch_size) +
                                    ") exceeds the number of stored transitions (" + std::to_string(size_) + ")");
    }

    std::vector<float> observations(static_cast<size_t>(batch_size * observation_dim_));
    std::vector<float> actions(static_cast<size_t>(batch_size * action_dim_));
    std::vector<float> rewards(static_cast<size_t>(batch_size));
    std::vector<float> next_observations(static_cast<size_t>(batch_size * observation_dim_));
    std::vector<float> dones(static_cast<size_t>(batch_size));

    for (int64_t b = 0; b < batch_size; ++b) {
        // Uniform over [0, size_), *with* replacement -- so a batch may repeat a transition,
        // and batch_size == size() is a legitimate full-content draw rather than a
        // permutation. Never over [0, capacity_): a partially-filled buffer's unwritten
        // slots are zeros, not transitions, and must not be sampled.
        const int64_t index = next_index(size_);
        gather_row(observations, b, observations_, index, observation_dim_);
        gather_row(actions, b, actions_, index, action_dim_);
        gather_row(next_observations, b, next_observations_, index, observation_dim_);
        rewards[static_cast<size_t>(b)] = rewards_.data()[index];
        dones[static_cast<size_t>(b)] = dones_.data()[index];
    }

    return ReplayBatch{Tensor(Shape({batch_size, observation_dim_}), backend_, observations),
                       Tensor(Shape({batch_size, action_dim_}), backend_, actions),
                       Tensor(Shape({batch_size, 1}), backend_, rewards),
                       Tensor(Shape({batch_size, observation_dim_}), backend_, next_observations),
                       Tensor(Shape({batch_size, 1}), backend_, dones)};
}

}  // namespace exai
