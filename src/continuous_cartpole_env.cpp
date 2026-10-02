#include "pulsatrix/continuous_cartpole_env.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/shape.hpp"

namespace pulsatrix {
namespace {

// Standard CartPole-v1 constants (Barto, Sutton & Anderson 1983) -- byte-for-byte the values
// src/cartpole_env.cpp uses. Copied deliberately rather than shared: see the header's note on
// why this is a sibling class, and note that ActionPlusOneReproducesDiscreteRightPush /
// ActionMinusOneReproducesDiscreteLeftPush compare full trajectories bit-exactly against
// CartPoleEnv, so they fail the instant any of these drifts from the discrete original's.
constexpr double kGravity = 9.8;
constexpr double kMassPole = 0.1;
constexpr double kTotalMass = 1.1;        // masscart (1.0) + masspole (0.1)
constexpr double kLength = 0.5;           // half the pole's length
constexpr double kPoleMassLength = 0.05;  // masspole * length
constexpr double kForceMag = 10.0;
constexpr double kTau = 0.02;  // integration timestep, seconds

}  // namespace

ContinuousCartPoleEnv::ContinuousCartPoleEnv(DeviceBackend* backend, int64_t max_steps, uint32_t seed)
    : backend_(backend), max_steps_(max_steps), lcg_state_(seed) {
    if (max_steps < 1) {
        throw std::invalid_argument("ContinuousCartPoleEnv: max_steps must be >= 1");
    }
}

double ContinuousCartPoleEnv::next_uniform() {
    // Numerical Recipes LCG constants -- the same generator CartPoleEnv and this codebase's
    // own tests use for reproducible pseudo-randomness, deliberately not <random>, whose
    // engine outputs beyond mt19937 are implementation-defined.
    lcg_state_ = lcg_state_ * 1664525u + 1013904223u;
    const double unit = static_cast<double>((lcg_state_ >> 8) & 0xFFFFu) / 65535.0;  // [0, 1]
    return (unit - 0.5) * 0.1;                                                       // [-0.05, 0.05]
}

Tensor ContinuousCartPoleEnv::observation() const {
    return Tensor(Shape({1, 4}), backend_,
                  {static_cast<float>(x_), static_cast<float>(x_dot_), static_cast<float>(theta_),
                   static_cast<float>(theta_dot_)});
}

Tensor ContinuousCartPoleEnv::reset() {
    x_ = next_uniform();
    x_dot_ = next_uniform();
    theta_ = next_uniform();
    theta_dot_ = next_uniform();
    step_count_ = 0;
    has_reset_ = true;
    return observation();
}

Tensor ContinuousCartPoleEnv::reset(const Tensor& initial_state) {
    if (initial_state.rank() != 2 || initial_state.shape().dim(0) != 1 || initial_state.shape().dim(1) != 4) {
        throw std::invalid_argument("ContinuousCartPoleEnv::reset: initial_state must have shape (1, 4)");
    }
    // Host boundary (GPU-native-kernels Mission 7): one device->host copy of the 4 values.
    const std::vector<float> state = initial_state.to_host_vector();

    x_ = static_cast<double>(state[0]);
    x_dot_ = static_cast<double>(state[1]);
    theta_ = static_cast<double>(state[2]);
    theta_dot_ = static_cast<double>(state[3]);
    step_count_ = 0;
    has_reset_ = true;
    return observation();
}

StepResult ContinuousCartPoleEnv::step(const Tensor& action) {
    if (!has_reset_) {
        throw std::invalid_argument("ContinuousCartPoleEnv::step called before reset");
    }
    if (action.rank() != 2 || action.shape().dim(0) != 1 || action.shape().dim(1) != 1) {
        throw std::invalid_argument("ContinuousCartPoleEnv::step: action must have shape (1, 1)");
    }

    // Widened to double before the range check: 1.0f + 1e-4f rounds to the same float as
    // 1.0001f, so comparing in float would accept an out-of-range action. Phrased as "must be
    // inside the band" so a NaN action (for which every comparison is false) is rejected too.
    // Host boundary (GPU-native-kernels Mission 7): the action may live on any device and is
    // read back with one device->host copy; the physics runs on the host in double.
    const double requested = static_cast<double>(action.to_host_vector()[0]);
    if (!(requested >= -1.0 - kActionRangeTolerance && requested <= 1.0 + kActionRangeTolerance)) {
        throw std::invalid_argument(
            "ContinuousCartPoleEnv::step: action must be a force fraction in [-1, 1] (within 1e-4)");
    }
    // Clamp to exactly [-1, 1] so a tolerated float-rounding excess yields exactly
    // +/-kForceMag, not kForceMag scaled by 1.00001.
    const double fraction = requested < -1.0 ? -1.0 : (requested > 1.0 ? 1.0 : requested);

    // The only difference from CartPoleEnv::step(): a scaled force instead of a sign choice.
    const double force = fraction * kForceMag;
    const double costheta = std::cos(theta_);
    const double sintheta = std::sin(theta_);

    const double temp = (force + kPoleMassLength * theta_dot_ * theta_dot_ * sintheta) / kTotalMass;
    const double thetaacc = (kGravity * sintheta - costheta * temp) /
                            (kLength * (4.0 / 3.0 - kMassPole * costheta * costheta / kTotalMass));
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
