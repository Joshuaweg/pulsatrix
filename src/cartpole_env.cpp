#include "pulsatrix/cartpole_env.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/shape.hpp"

namespace pulsatrix {
namespace {

// Standard CartPole-v1 constants (Barto, Sutton & Anderson 1983). Fixed by design -- see
// the class note in cartpole_env.hpp on why these are not constructor parameters.
constexpr double kGravity = 9.8;
constexpr double kMassPole = 0.1;
constexpr double kTotalMass = 1.1;      // masscart (1.0) + masspole (0.1)
constexpr double kLength = 0.5;         // half the pole's length
constexpr double kPoleMassLength = 0.05;  // masspole * length
constexpr double kForceMag = 10.0;
constexpr double kTau = 0.02;  // integration timestep, seconds

// How far a discrete action's float encoding may sit from a whole number before it is
// rejected. Loose enough to tolerate a policy's float round-trip, tight enough that a
// genuinely fractional "action" (an un-argmaxed probability, say) is caught.
constexpr float kActionIntegerTolerance = 1e-4f;

}  // namespace

CartPoleEnv::CartPoleEnv(DeviceBackend* backend, int64_t max_steps, uint32_t seed)
    : backend_(backend), max_steps_(max_steps), lcg_state_(seed) {
    if (max_steps < 1) {
        throw std::invalid_argument("CartPoleEnv: max_steps must be >= 1");
    }
}

double CartPoleEnv::next_uniform() {
    // Numerical Recipes LCG constants -- the same generator this codebase's own tests use
    // for reproducible pseudo-randomness (see gan_integration_test.cpp), deliberately not
    // <random>, whose engine outputs beyond mt19937 are implementation-defined.
    lcg_state_ = lcg_state_ * 1664525u + 1013904223u;
    const double unit = static_cast<double>((lcg_state_ >> 8) & 0xFFFFu) / 65535.0;  // [0, 1]
    return (unit - 0.5) * 0.1;                                                       // [-0.05, 0.05]
}

Tensor CartPoleEnv::observation() const {
    return Tensor(Shape({1, 4}), backend_,
                  {static_cast<float>(x_), static_cast<float>(x_dot_), static_cast<float>(theta_),
                   static_cast<float>(theta_dot_)});
}

Tensor CartPoleEnv::reset() {
    x_ = next_uniform();
    x_dot_ = next_uniform();
    theta_ = next_uniform();
    theta_dot_ = next_uniform();
    step_count_ = 0;
    has_reset_ = true;
    return observation();
}

Tensor CartPoleEnv::reset(const Tensor& initial_state) {
    if (initial_state.rank() != 2 || initial_state.shape().dim(0) != 1 || initial_state.shape().dim(1) != 4) {
        throw std::invalid_argument("CartPoleEnv::reset: initial_state must have shape (1, 4)");
    }
    PULSATRIX_REQUIRE_HOST(initial_state);

    x_ = static_cast<double>(initial_state.data()[0]);
    x_dot_ = static_cast<double>(initial_state.data()[1]);
    theta_ = static_cast<double>(initial_state.data()[2]);
    theta_dot_ = static_cast<double>(initial_state.data()[3]);
    step_count_ = 0;
    has_reset_ = true;
    return observation();
}

StepResult CartPoleEnv::step(const Tensor& action) {
    // Raw host loop over Tensor::data() -- undefined behavior on a CUDA-backed Tensor.
    // See mission_host_loop_guards.md; same guard as every prior module.
    PULSATRIX_REQUIRE_HOST(action);

    if (!has_reset_) {
        throw std::invalid_argument("CartPoleEnv::step called before reset");
    }
    if (action.rank() != 2 || action.shape().dim(0) != 1 || action.shape().dim(1) != 1) {
        throw std::invalid_argument("CartPoleEnv::step: action must have shape (1, 1)");
    }

    const float encoded = action.data()[0];
    const float rounded = std::round(encoded);
    if (std::abs(encoded - rounded) > kActionIntegerTolerance) {
        throw std::invalid_argument("CartPoleEnv::step: action must encode a whole-number action index");
    }
    const int64_t index = static_cast<int64_t>(rounded);
    if (index < 0 || index >= action_dim()) {
        throw std::invalid_argument("CartPoleEnv::step: action index out of range [0, action_dim())");
    }

    const double force = (index == 1) ? kForceMag : -kForceMag;
    const double costheta = std::cos(theta_);
    const double sintheta = std::sin(theta_);

    const double temp = (force + kPoleMassLength * theta_dot_ * theta_dot_ * sintheta) / kTotalMass;
    const double thetaacc =
        (kGravity * sintheta - costheta * temp) / (kLength * (4.0 / 3.0 - kMassPole * costheta * costheta / kTotalMass));
    const double xacc = temp - kPoleMassLength * thetaacc * costheta / kTotalMass;

    // Explicit (forward) Euler, in Gym's exact order: positions advance using the
    // *pre-update* velocities. Not semi-implicit Euler -- see the header's note.
    x_ += kTau * x_dot_;
    x_dot_ += kTau * xacc;
    theta_ += kTau * theta_dot_;
    theta_dot_ += kTau * thetaacc;

    ++step_count_;

    const bool done = std::abs(x_) > kXThreshold || std::abs(theta_) > kThetaThreshold || step_count_ >= max_steps_;

    // Reward 1.0 on every step, the terminating one included -- Gym's own convention: the
    // agent is paid for having survived through this step, not for what follows it.
    return StepResult{observation(), 1.0f, done};
}

}  // namespace pulsatrix
