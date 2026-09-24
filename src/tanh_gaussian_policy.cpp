#include "exai/tanh_gaussian_policy.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "exai/assert.hpp"
#include "exai/shape.hpp"

namespace exai {
namespace {

// 0.5 * log(2*pi), the Gaussian log-density's normalizing constant. Spelled as a literal
// (rather than 0.5*std::log(2*M_PI)) because M_PI is not standard C++ and this value is
// otherwise recomputed identically on every element.
constexpr double kHalfLogTwoPi = 0.91893853320467274178;

}  // namespace

TanhGaussianPolicy::TanhGaussianPolicy(DeviceBackend* backend)
    : backend_(backend),
      last_action_(Shape({0}), backend),
      last_std_(Shape({0}), backend),
      last_epsilon_(Shape({0}), backend) {}

TanhGaussianSample TanhGaussianPolicy::forward(const Tensor& mean, const Tensor& log_std, const Tensor& epsilon) {
    // Dereferences Tensor::data() directly in a raw host loop (exp/tanh/log have no backend
    // primitive) -- not yet backend-generic. See
    // campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision and
    // mission_host_loop_guards.md.
    EXAI_ASSERT(mean.device() == DeviceType::Cpu);
    EXAI_ASSERT(log_std.device() == DeviceType::Cpu);
    EXAI_ASSERT(epsilon.device() == DeviceType::Cpu);

    if (mean.rank() != 2 || mean.shape().dim(0) < 1 || mean.shape().dim(1) < 1) {
        throw std::invalid_argument(
            "TanhGaussianPolicy::forward: mean must have shape (N, action_dim) with N >= 1 and action_dim >= 1");
    }
    if (!(mean.shape() == log_std.shape()) || !(mean.shape() == epsilon.shape())) {
        throw std::invalid_argument(
            "TanhGaussianPolicy::forward: mean, log_std and epsilon must all have the same shape");
    }

    const int64_t batch_size = mean.shape().dim(0);
    const int64_t action_dim = mean.shape().dim(1);

    Tensor action(mean.shape(), backend_);
    Tensor std_cache(mean.shape(), backend_);
    std::vector<float> log_probs(static_cast<size_t>(batch_size));

    for (int64_t n = 0; n < batch_size; ++n) {
        // Accumulated in double: log_prob is a sum over action_dim of terms that can each
        // reach ~14 in magnitude (the stabilizer's bound) while the per-element gradient
        // information lives in far smaller differences. Same disposition as CartPoleEnv's
        // double-precision integration -- the stored Tensor stays float.
        double row_log_prob = 0.0;
        for (int64_t d = 0; d < action_dim; ++d) {
            const int64_t i = n * action_dim + d;
            const float std_value = std::exp(log_std.data()[i]);
            // u is exactly Reparameterize's own formula; the tanh is what SAC adds on top.
            const float u = mean.data()[i] + std_value * epsilon.data()[i];
            const float a = std::tanh(u);
            std_cache.data()[i] = std_value;
            action.data()[i] = a;

            const double eps_value = static_cast<double>(epsilon.data()[i]);
            // The quadratic term is -0.5*epsilon^2, *not* -0.5*((u-mean)/std)^2: those are
            // equal by construction (the reparameterization identity) but only the former
            // makes visible that this term has no live dependency on mean/log_std, which
            // backward()'s derivation relies on.
            const double squash_correction =
                std::log(1.0 - static_cast<double>(a) * static_cast<double>(a) +
                         static_cast<double>(kLogProbStabilizer));
            row_log_prob += -0.5 * eps_value * eps_value - static_cast<double>(log_std.data()[i]) - kHalfLogTwoPi -
                            squash_correction;
        }
        log_probs[static_cast<size_t>(n)] = static_cast<float>(row_log_prob);
    }

    last_action_ = action;
    last_std_ = std::move(std_cache);
    last_epsilon_ = epsilon;
    has_forwarded_ = true;

    return TanhGaussianSample{std::move(action), Tensor(Shape({batch_size, 1}), backend_, log_probs)};
}

TanhGaussianGrad TanhGaussianPolicy::backward(const Tensor& grad_action, const Tensor& grad_log_prob) const {
    if (!has_forwarded_) {
        throw std::logic_error("TanhGaussianPolicy::backward called before forward");
    }
    if (!(grad_action.shape() == last_action_.shape())) {
        throw std::invalid_argument(
            "TanhGaussianPolicy::backward: grad_action shape must match the cached forward shape");
    }
    if (!(grad_log_prob.shape() == last_action_.shape())) {
        throw std::invalid_argument(
            "TanhGaussianPolicy::backward: grad_log_prob shape must match the cached forward shape "
            "(N, action_dim) -- log_prob's own (N, 1) gradient broadcast across the row");
    }

    Tensor grad_mean(last_action_.shape(), backend_);
    Tensor grad_log_std(last_action_.shape(), backend_);

    for (int64_t i = 0; i < last_action_.numel(); ++i) {
        const float a = last_action_.data()[i];
        const float one_minus_a_sq = 1.0f - a * a;  // tanh's derivative, d(action)/d(u)
        // d(log_prob)/d(u), from the -log(1 - a^2 + c) term alone. Note the stabilizer is in
        // the denominator only: it comes from inside the log, not from the chain rule's
        // numerator, so the two are genuinely different expressions and must not be collapsed.
        const float dlogprob_du = 2.0f * a * one_minus_a_sq / (one_minus_a_sq + kLogProbStabilizer);

        // The whole point of this class's novel shape: both incoming gradients meet at u.
        const float grad_u = grad_action.data()[i] * one_minus_a_sq + grad_log_prob.data()[i] * dlogprob_du;

        grad_mean.data()[i] = grad_u;  // d(u)/d(mean) == 1
        // d(u)/d(log_std) = std*epsilon (chain rule through exp), plus log_prob's own explicit
        // -log_std term contributing -1 * grad_log_prob directly, bypassing u entirely.
        grad_log_std.data()[i] = grad_u * last_std_.data()[i] * last_epsilon_.data()[i] - grad_log_prob.data()[i];
    }

    return TanhGaussianGrad{std::move(grad_mean), std::move(grad_log_std)};
}

}  // namespace exai
