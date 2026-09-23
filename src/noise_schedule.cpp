#include "exai/noise_schedule.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "exai/assert.hpp"

namespace exai {

NoiseSchedule::NoiseSchedule(int64_t num_timesteps, float beta_start, float beta_end)
    : num_timesteps_(num_timesteps) {
    if (num_timesteps <= 0) {
        throw std::invalid_argument("NoiseSchedule: num_timesteps must be positive");
    }
    if (!(beta_start < beta_end)) {
        throw std::invalid_argument("NoiseSchedule: beta_start must be strictly less than beta_end");
    }

    const size_t count = static_cast<size_t>(num_timesteps);
    beta_.reserve(count);
    alpha_.reserve(count);
    alpha_bar_.reserve(count);

    float running_product = 1.0f;
    for (int64_t t = 1; t <= num_timesteps; ++t) {
        // Linear spacing with both endpoints hit exactly: beta_1 == beta_start,
        // beta_T == beta_end. T == 1 has no interval to interpolate across (the divisor
        // would be zero), and collapses onto the t = 1 endpoint.
        const float beta = (num_timesteps == 1)
                               ? beta_start
                               : beta_start + (beta_end - beta_start) * static_cast<float>(t - 1) /
                                                  static_cast<float>(num_timesteps - 1);
        const float alpha = 1.0f - beta;
        // alpha_bar is a *cumulative* product, so it is built incrementally here rather than
        // recomputed per accessor call -- recomputing would make a full T-step sampling loop
        // quadratic in T for no reason.
        running_product *= alpha;

        beta_.push_back(beta);
        alpha_.push_back(alpha);
        alpha_bar_.push_back(running_product);
    }
}

size_t NoiseSchedule::index_of(int64_t t, const char* context) const {
    if (t < 1 || t > num_timesteps_) {
        throw std::out_of_range(std::string(context) + ": timestep t must be in [1, " +
                                std::to_string(num_timesteps_) + "], got " + std::to_string(t));
    }
    return static_cast<size_t>(t - 1);
}

float NoiseSchedule::beta(int64_t t) const {
    return beta_[index_of(t, "NoiseSchedule::beta")];
}

float NoiseSchedule::alpha(int64_t t) const {
    return alpha_[index_of(t, "NoiseSchedule::alpha")];
}

float NoiseSchedule::alpha_bar(int64_t t) const {
    return alpha_bar_[index_of(t, "NoiseSchedule::alpha_bar")];
}

Tensor NoiseSchedule::add_noise(const Tensor& x0, const Tensor& epsilon, int64_t t) const {
    // Raw host loop dereferencing Tensor::data() directly -- see the header's note and
    // mission_host_loop_guards.md.
    EXAI_ASSERT(x0.device() == DeviceType::Cpu);
    EXAI_ASSERT(epsilon.device() == DeviceType::Cpu);

    if (!(x0.shape() == epsilon.shape())) {
        throw std::invalid_argument("NoiseSchedule::add_noise: x0 and epsilon must have the same shape");
    }
    const size_t idx = index_of(t, "NoiseSchedule::add_noise");

    const float alpha_bar_t = alpha_bar_[idx];
    const float signal_scale = std::sqrt(alpha_bar_t);
    // 1 - alpha_bar_t is clamped at zero before the sqrt: alpha_bar is a long product of
    // floats slightly below 1, so it can only drift *below* 1 here, but the clamp keeps a
    // pathological schedule from producing a NaN silently rather than a wrong-but-finite value.
    const float noise_scale = std::sqrt(std::fmax(0.0f, 1.0f - alpha_bar_t));

    // Copy-construct from x0 rather than allocating through a stored DeviceBackend*: this
    // class deliberately holds no backend (its constructor takes only schedule parameters),
    // and a Tensor's copy constructor already carries the right backend and shape.
    Tensor x_t(x0);
    for (int64_t i = 0; i < x_t.numel(); ++i) {
        x_t.data()[i] = signal_scale * x0.data()[i] + noise_scale * epsilon.data()[i];
    }
    return x_t;
}

Tensor NoiseSchedule::denoise_step(const Tensor& x_t, const Tensor& predicted_epsilon, const Tensor& z,
                                   int64_t t) const {
    EXAI_ASSERT(x_t.device() == DeviceType::Cpu);
    EXAI_ASSERT(predicted_epsilon.device() == DeviceType::Cpu);
    EXAI_ASSERT(z.device() == DeviceType::Cpu);

    if (!(x_t.shape() == predicted_epsilon.shape()) || !(x_t.shape() == z.shape())) {
        throw std::invalid_argument(
            "NoiseSchedule::denoise_step: x_t, predicted_epsilon and z must all have the same shape");
    }
    const size_t idx = index_of(t, "NoiseSchedule::denoise_step");

    const float beta_t = beta_[idx];
    const float alpha_t = alpha_[idx];
    const float alpha_bar_t = alpha_bar_[idx];

    const float inv_sqrt_alpha = 1.0f / std::sqrt(alpha_t);
    const float eps_coefficient = beta_t / std::sqrt(std::fmax(0.0f, 1.0f - alpha_bar_t));
    const float z_scale = std::sqrt(beta_t);

    Tensor x_prev(x_t);
    for (int64_t i = 0; i < x_prev.numel(); ++i) {
        x_prev.data()[i] = inv_sqrt_alpha * (x_t.data()[i] - eps_coefficient * predicted_epsilon.data()[i]) +
                           z_scale * z.data()[i];
    }
    return x_prev;
}

}  // namespace exai
