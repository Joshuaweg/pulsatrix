#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/kl_divergence_loss.hpp"

namespace pulsatrix {
namespace {

class KLDivergenceLossTest : public ::testing::Test {
protected:
    CPUBackend backend;
    KLDivergenceLoss loss{&backend};
};

// KL(N(0, I) || N(0, I)) == 0 exactly: mu = 0 and log_sigma = 0 makes every per-element term
// 0 + 1 - 0 - 1. This is the one value the closed form must get exactly right.
TEST_F(KLDivergenceLossTest, ForwardIsZeroForStandardNormalPosterior) {
    Tensor mu(Shape({3, 4}), &backend);
    Tensor log_sigma(Shape({3, 4}), &backend);

    EXPECT_NEAR(loss.forward(mu, log_sigma), 0.0f, 1e-6f);
}

TEST_F(KLDivergenceLossTest, ForwardComputesHandVerifiedValue) {
    // N = 1, latent = 2. mu = [1, 0], log_sigma = [0, ln(2)].
    // d=0: 1 + exp(0) - 0 - 1 = 1
    // d=1: 0 + exp(2*ln2) - 2*ln2 - 1 = 4 - 1.3862944 - 1 = 1.6137056
    // loss = 0.5 * (1 + 1.6137056) / 1 = 1.3068528
    Tensor mu(Shape({1, 2}), &backend, {1.0f, 0.0f});
    Tensor log_sigma(Shape({1, 2}), &backend, {0.0f, std::log(2.0f)});

    EXPECT_NEAR(loss.forward(mu, log_sigma), 1.3068528f, 1e-5f);
}

// The batch dimension is a *mean*, not a sum: duplicating an example across the batch must
// leave the loss unchanged. This is what pins the ELBO normalization (sum over latent_dim,
// mean over batch) rather than any other combination.
TEST_F(KLDivergenceLossTest, ForwardAveragesOverBatchNotSums) {
    Tensor mu_one(Shape({1, 2}), &backend, {1.0f, -0.5f});
    Tensor log_sigma_one(Shape({1, 2}), &backend, {0.3f, -0.4f});
    Tensor mu_two(Shape({2, 2}), &backend, {1.0f, -0.5f, 1.0f, -0.5f});
    Tensor log_sigma_two(Shape({2, 2}), &backend, {0.3f, -0.4f, 0.3f, -0.4f});

    KLDivergenceLoss loss_one(&backend);
    KLDivergenceLoss loss_two(&backend);
    const float value_one = loss_one.forward(mu_one, log_sigma_one);
    const float value_two = loss_two.forward(mu_two, log_sigma_two);

    EXPECT_NEAR(value_one, value_two, 1e-5f);
    // ...and that shared value is genuinely non-zero, so the comparison above is not
    // satisfiable by a loss that always returns 0.
    EXPECT_GT(value_one, 1e-2f);
}

TEST_F(KLDivergenceLossTest, ForwardIsNonNegativeForAssortedPosteriors) {
    // KL divergence is non-negative by definition; the closed form must inherit that.
    Tensor mu(Shape({2, 3}), &backend, {0.8f, -0.6f, 1.2f, 0.5f, -0.9f, 1.1f});
    Tensor log_sigma(Shape({2, 3}), &backend, {0.35f, -0.85f, 1.0f, 0.15f, -0.45f, 0.7f});

    EXPECT_GT(loss.forward(mu, log_sigma), 0.0f);
}

TEST_F(KLDivergenceLossTest, ForwardThrowsOnMismatchedShapes) {
    Tensor mu(Shape({2, 3}), &backend);
    Tensor log_sigma(Shape({2, 4}), &backend);

    EXPECT_THROW({ (void)loss.forward(mu, log_sigma); }, std::invalid_argument);
}

TEST_F(KLDivergenceLossTest, ForwardThrowsOnNonRank2Input) {
    Tensor mu(Shape({6}), &backend);
    Tensor log_sigma(Shape({6}), &backend);

    EXPECT_THROW({ (void)loss.forward(mu, log_sigma); }, std::invalid_argument);
}

TEST_F(KLDivergenceLossTest, BackwardThrowsBeforeForward) {
    EXPECT_THROW({ (void)loss.backward(); }, std::logic_error);
}

TEST_F(KLDivergenceLossTest, BackwardIsZeroAtTheStandardNormalMinimum) {
    // mu = 0, log_sigma = 0 is the global minimum of this loss, so both gradients vanish
    // there: mu/N = 0 and (exp(0) - 1)/N = 0.
    Tensor mu(Shape({2, 3}), &backend);
    Tensor log_sigma(Shape({2, 3}), &backend);
    (void)loss.forward(mu, log_sigma);

    ReparamGrad grads = loss.backward();

    for (int64_t i = 0; i < grads.grad_mu.numel(); ++i) {
        EXPECT_NEAR(grads.grad_mu.data()[i], 0.0f, 1e-6f) << "index " << i;
        EXPECT_NEAR(grads.grad_log_sigma.data()[i], 0.0f, 1e-6f) << "index " << i;
    }
}

TEST_F(KLDivergenceLossTest, BackwardComputesHandVerifiedGradients) {
    // N = 2. grad_mu = mu/2; grad_log_sigma = (exp(2*log_sigma) - 1)/2.
    // log_sigma = ln(2) -> (4 - 1)/2 = 1.5; log_sigma = 0 -> 0.
    Tensor mu(Shape({2, 1}), &backend, {3.0f, -1.0f});
    Tensor log_sigma(Shape({2, 1}), &backend, {std::log(2.0f), 0.0f});
    (void)loss.forward(mu, log_sigma);

    ReparamGrad grads = loss.backward();

    EXPECT_NEAR(grads.grad_mu.data()[0], 1.5f, 1e-5f);
    EXPECT_NEAR(grads.grad_mu.data()[1], -0.5f, 1e-5f);
    EXPECT_NEAR(grads.grad_log_sigma.data()[0], 1.5f, 1e-5f);
    EXPECT_NEAR(grads.grad_log_sigma.data()[1], 0.0f, 1e-5f);
}

// ---------------------------------------------------------------------------------------
// Central finite-difference gradient checks.
//
// The loss is itself the scalar objective, so no artificial grad seed is needed -- the
// numeric derivative is taken directly on forward()'s return value. The fixture is
// deliberately non-uniform (N != latent_dim, mu spanning both signs over ~0.5-1.2, log_sigma
// spanning both signs so exp(2*log_sigma) ranges ~0.18-7.4) rather than small-and-uniform:
// RWKVModule's mission found the hard way that a uniform small-scale fixture can make a
// check pass *vacuously* against the tolerance floor. Tolerance is relative
// (1e-3 + 2e-2*|numeric|) for the same established reason. Non-vacuity is additionally
// asserted inline (every numeric gradient is well clear of the floor) and was verified by
// mutation probe (see the mission's close-out evidence).
// ---------------------------------------------------------------------------------------
namespace {

constexpr int64_t kN = 2;
constexpr int64_t kLatent = 3;

const std::vector<float>& fd_mu_values() {
    static const std::vector<float> v{0.80f, -0.60f, 1.20f, 0.50f, -0.90f, 1.10f};
    return v;
}
const std::vector<float>& fd_log_sigma_values() {
    static const std::vector<float> v{0.35f, -0.85f, 1.00f, 0.15f, -0.45f, 0.70f};
    return v;
}

float scalar_loss(CPUBackend& backend, const std::vector<float>& mu_values,
                  const std::vector<float>& log_sigma_values) {
    KLDivergenceLoss l(&backend);
    Tensor mu(Shape({kN, kLatent}), &backend, mu_values);
    Tensor log_sigma(Shape({kN, kLatent}), &backend, log_sigma_values);
    return l.forward(mu, log_sigma);
}

float relative_tolerance(float numeric) {
    return 1e-3f + 2e-2f * std::fabs(numeric);
}

}  // namespace

TEST_F(KLDivergenceLossTest, BackwardGradMuMatchesFiniteDifference) {
    const float h = 1e-3f;
    Tensor mu(Shape({kN, kLatent}), &backend, fd_mu_values());
    Tensor log_sigma(Shape({kN, kLatent}), &backend, fd_log_sigma_values());
    (void)loss.forward(mu, log_sigma);
    ReparamGrad grads = loss.backward();

    ASSERT_EQ(grads.grad_mu.numel(), kN * kLatent);
    for (size_t idx = 0; idx < fd_mu_values().size(); ++idx) {
        std::vector<float> plus = fd_mu_values();
        std::vector<float> minus = fd_mu_values();
        plus[idx] += h;
        minus[idx] -= h;
        const float numeric = (scalar_loss(backend, plus, fd_log_sigma_values()) -
                               scalar_loss(backend, minus, fd_log_sigma_values())) /
                              (2.0f * h);
        EXPECT_NEAR(grads.grad_mu.data()[static_cast<int64_t>(idx)], numeric, relative_tolerance(numeric))
            << "mu index " << idx;
        EXPECT_GT(std::fabs(numeric), 1e-2f) << "mu index " << idx;
    }
}

TEST_F(KLDivergenceLossTest, BackwardGradLogSigmaMatchesFiniteDifference) {
    const float h = 1e-3f;
    Tensor mu(Shape({kN, kLatent}), &backend, fd_mu_values());
    Tensor log_sigma(Shape({kN, kLatent}), &backend, fd_log_sigma_values());
    (void)loss.forward(mu, log_sigma);
    ReparamGrad grads = loss.backward();

    ASSERT_EQ(grads.grad_log_sigma.numel(), kN * kLatent);
    for (size_t idx = 0; idx < fd_log_sigma_values().size(); ++idx) {
        std::vector<float> plus = fd_log_sigma_values();
        std::vector<float> minus = fd_log_sigma_values();
        plus[idx] += h;
        minus[idx] -= h;
        const float numeric =
            (scalar_loss(backend, fd_mu_values(), plus) - scalar_loss(backend, fd_mu_values(), minus)) / (2.0f * h);
        EXPECT_NEAR(grads.grad_log_sigma.data()[static_cast<int64_t>(idx)], numeric, relative_tolerance(numeric))
            << "log_sigma index " << idx;
        EXPECT_GT(std::fabs(numeric), 1e-2f) << "log_sigma index " << idx;
    }
}

using KLDivergenceLossDeathTest = KLDivergenceLossTest;

// GPU-native-kernels Mission 1b: the computation is device-generic, so mixed-device inputs
// are a reachable caller error -- rejected before any kernel sees two devices' pointers.
TEST_F(KLDivergenceLossTest, ForwardThrowsOnMixedDevices) {
    Tensor mu(Shape({1, 2}), &backend, {0.5f, -0.5f});
    Tensor log_sigma(Shape({1, 2}), &backend, {0.1f, 0.2f}, DeviceType::Hip);
    EXPECT_THROW((void)loss.forward(mu, log_sigma), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
