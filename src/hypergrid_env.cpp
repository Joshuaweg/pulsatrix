#include "pulsatrix/hypergrid_env.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/shape.hpp"

namespace pulsatrix {
namespace {

// How far a coordinate's float encoding may sit from a whole number before it is rejected --
// same tolerance and rationale as CartPoleEnv's discrete-action decoding.
constexpr float kIntegerTolerance = 1e-4f;

}  // namespace

HyperGridEnv::HyperGridEnv(DeviceBackend* backend, int64_t ndim, int64_t side_length, float r0, float r1, float r2,
                            int64_t max_steps)
    : backend_(backend), ndim_(ndim), side_length_(side_length), r0_(r0), r1_(r1), r2_(r2), max_steps_(max_steps) {
    if (ndim < 1) {
        throw std::invalid_argument("HyperGridEnv: ndim must be >= 1");
    }
    if (side_length < 4) {
        throw std::invalid_argument("HyperGridEnv: side_length must be >= 4");
    }
    if (r0 < 0.0f || r1 < 0.0f || r2 < 0.0f) {
        throw std::invalid_argument("HyperGridEnv: reward constants must be >= 0");
    }
    if (max_steps < 1) {
        throw std::invalid_argument("HyperGridEnv: max_steps must be >= 1");
    }
    coord_.assign(static_cast<size_t>(ndim), 0);
}

Tensor HyperGridEnv::observation() const {
    std::vector<float> values(coord_.size());
    for (size_t i = 0; i < coord_.size(); ++i) {
        values[i] = static_cast<float>(coord_[i]);
    }
    return Tensor(Shape({1, ndim_}), backend_, values);
}

void HyperGridEnv::decode_state(const Tensor& state, std::vector<int64_t>& out) const {
    if (state.rank() != 2 || state.shape().dim(0) != 1 || state.shape().dim(1) != ndim_) {
        throw std::invalid_argument("HyperGridEnv: state must have shape (1, ndim())");
    }
    PULSATRIX_ASSERT(state.device() == DeviceType::Cpu);

    out.assign(static_cast<size_t>(ndim_), 0);
    for (int64_t i = 0; i < ndim_; ++i) {
        const float encoded = state.data()[i];
        const float rounded = std::round(encoded);
        if (std::abs(encoded - rounded) > kIntegerTolerance) {
            throw std::invalid_argument("HyperGridEnv: state coordinates must be integer-valued");
        }
        const int64_t value = static_cast<int64_t>(rounded);
        if (value < 0 || value > side_length_ - 1) {
            throw std::invalid_argument("HyperGridEnv: state coordinate out of range [0, side_length()-1]");
        }
        out[static_cast<size_t>(i)] = value;
    }
}

Tensor HyperGridEnv::reset() {
    coord_.assign(static_cast<size_t>(ndim_), 0);
    step_count_ = 0;
    has_reset_ = true;
    done_ = false;
    return observation();
}

Tensor HyperGridEnv::reset(const Tensor& initial_state) {
    decode_state(initial_state, coord_);
    step_count_ = 0;
    has_reset_ = true;
    done_ = false;
    return observation();
}

float HyperGridEnv::reward(const Tensor& state) const {
    std::vector<int64_t> point;
    decode_state(state, point);

    bool near_corner = true;
    bool exact_corner = true;
    for (int64_t i = 0; i < ndim_; ++i) {
        const float u = static_cast<float>(point[static_cast<size_t>(i)]) / static_cast<float>(side_length_ - 1);
        if (!(u <= 0.25f || u >= 0.75f)) {
            near_corner = false;
        }
        if (point[static_cast<size_t>(i)] != 0 && point[static_cast<size_t>(i)] != side_length_ - 1) {
            exact_corner = false;
        }
    }

    float r = r0_;
    if (near_corner) {
        r += r1_;
    }
    if (exact_corner) {
        r += r2_;
    }
    return r;
}

float HyperGridEnv::backward_log_prob(const Tensor& state, int64_t action) const {
    std::vector<int64_t> point;
    decode_state(state, point);

    if (action < 0 || action >= ndim_) {
        throw std::invalid_argument("HyperGridEnv::backward_log_prob: action out of range [0, ndim())");
    }
    if (point[static_cast<size_t>(action)] == 0) {
        throw std::invalid_argument("HyperGridEnv::backward_log_prob: action's coordinate is already 0 -- not a "
                                     "valid parent transition");
    }

    int64_t nonzero_count = 0;
    for (int64_t i = 0; i < ndim_; ++i) {
        if (point[static_cast<size_t>(i)] != 0) {
            ++nonzero_count;
        }
    }
    return -std::log(static_cast<float>(nonzero_count));
}

std::vector<bool> HyperGridEnv::valid_actions_mask(const Tensor&) const {
    throw std::logic_error("HyperGridEnv::valid_actions_mask not yet implemented");
}

StepResult HyperGridEnv::step(const Tensor& action) {
    PULSATRIX_ASSERT(action.device() == DeviceType::Cpu);

    if (!has_reset_) {
        throw std::invalid_argument("HyperGridEnv::step called before reset");
    }
    if (done_) {
        throw std::logic_error("HyperGridEnv::step called after the episode already terminated");
    }
    if (action.rank() != 2 || action.shape().dim(0) != 1 || action.shape().dim(1) != 1) {
        throw std::invalid_argument("HyperGridEnv::step: action must have shape (1, 1)");
    }

    const float encoded = action.data()[0];
    const float rounded = std::round(encoded);
    if (std::abs(encoded - rounded) > kIntegerTolerance) {
        throw std::invalid_argument("HyperGridEnv::step: action must encode a whole-number action index");
    }
    const int64_t index = static_cast<int64_t>(rounded);
    if (index < 0 || index >= action_dim()) {
        throw std::invalid_argument("HyperGridEnv::step: action index out of range [0, action_dim())");
    }

    if (index == ndim_) {
        // stop
        ++step_count_;
        done_ = true;
        return StepResult{observation(), reward(observation()), true};
    }

    if (coord_[static_cast<size_t>(index)] >= side_length_ - 1) {
        throw std::invalid_argument("HyperGridEnv::step: illegal off-grid increment");
    }
    ++coord_[static_cast<size_t>(index)];
    ++step_count_;

    const bool capped = step_count_ >= max_steps_;
    done_ = capped;
    const float step_reward = capped ? reward(observation()) : 0.0f;
    return StepResult{observation(), step_reward, capped};
}

}  // namespace pulsatrix
