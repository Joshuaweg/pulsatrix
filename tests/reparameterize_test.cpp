#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/reparameterize.hpp"

namespace pulsatrix {
namespace {

class ReparameterizeTest : public ::testing::Test {
protected:
    CPUBackend backend;
    Reparameterize reparam{&backend};
};

TEST_F(ReparameterizeTest, ForwardComputesHandVerifiedValue) {
    // z = mu + exp(log_sigma)*eps.
    // log_sigma = 0 -> exp = 1; log_sigma = ln(2) -> exp = 2.
    Tensor mu(Shape({1, 3}), &backend, {1.0f, -2.0f, 0.5f});
    Tensor log_sigma(Shape({1, 3}), &backend, {0.0f, std::log(2.0f), 0.0f});
    Tensor eps(Shape({1, 3}), &backend, {0.25f, 3.0f, -1.5f});

    Tensor z = reparam.forward(mu, log_sigma, eps);

    ASSERT_EQ(z.numel(), 3);
    EXPECT_NEAR(z.data()[0], 1.0f + 1.0f * 0.25f, 1e-5f);
    EXPECT_NEAR(z.data()[1], -2.0f + 2.0f * 3.0f, 1e-5f);
    EXPECT_NEAR(z.data()[2], 0.5f + 1.0f * -1.5f, 1e-5f);
}

TEST_F(ReparameterizeTest, ForwardWithZeroEpsilonReturnsMuExactly) {
    // The deterministic/"mean" path: epsilon = 0 collapses z onto mu regardless of sigma.
    Tensor mu(Shape({2, 2}), &backend, {1.0f, -2.0f, 0.5f, 7.0f});
    Tensor log_sigma(Shape({2, 2}), &backend, {2.0f, -3.0f, 0.0f, 1.0f});
    Tensor eps(Shape({2, 2}), &backend, {0.0f, 0.0f, 0.0f, 0.0f});

    Tensor z = reparam.forward(mu, log_sigma, eps);

    for (int64_t i = 0; i < z.numel(); ++i) {
        EXPECT_FLOAT_EQ(z.data()[i], mu.data()[i]) << "index " << i;
    }
}

TEST_F(ReparameterizeTest, ForwardPreservesShape) {
    Tensor mu(Shape({4, 5}), &backend);
    Tensor log_sigma(Shape({4, 5}), &backend);
    Tensor eps(Shape({4, 5}), &backend);

    Tensor z = reparam.forward(mu, log_sigma, eps);

    EXPECT_EQ(z.shape().rank(), 2);
    EXPECT_EQ(z.shape().dim(0), 4);
    EXPECT_EQ(z.shape().dim(1), 5);
}

TEST_F(ReparameterizeTest, ForwardThrowsOnMismatchedLogSigmaShape) {
    Tensor mu(Shape({2, 3}), &backend);
    Tensor log_sigma(Shape({2, 4}), &backend);
    Tensor eps(Shape({2, 3}), &backend);

    EXPECT_THROW({ (void)reparam.forward(mu, log_sigma, eps); }, std::invalid_argument);
}

TEST_F(ReparameterizeTest, ForwardThrowsOnMismatchedEpsilonShape) {
    Tensor mu(Shape({2, 3}), &backend);
    Tensor log_sigma(Shape({2, 3}), &backend);
    Tensor eps(Shape({3, 3}), &backend);

    EXPECT_THROW({ (void)reparam.forward(mu, log_sigma, eps); }, std::invalid_argument);
}

TEST_F(ReparameterizeTest, BackwardThrowsBeforeForward) {
    Tensor grad_z(Shape({1, 3}), &backend, {1.0f, 1.0f, 1.0f});

    EXPECT_THROW({ (void)reparam.backward(grad_z); }, std::logic_error);
}

TEST_F(ReparameterizeTest, BackwardThrowsOnMismatchedGradShape) {
    Tensor mu(Shape({1, 3}), &backend);
    Tensor log_sigma(Shape({1, 3}), &backend);
    Tensor eps(Shape({1, 3}), &backend);
    (void)reparam.forward(mu, log_sigma, eps);

    Tensor grad_z(Shape({1, 4}), &backend);
    EXPECT_THROW({ (void)reparam.backward(grad_z); }, std::invalid_argument);
}

// grad_mu is the trivial identity dz/dmu = 1, so it gets a direct-value check rather than a
// finite-difference one (an FD check of an exact pass-through proves little the hand value
// doesn't). grad_log_sigma is the real chain rule and gets the FD treatment below.
TEST_F(ReparameterizeTest, BackwardGradMuIsGradZPassedThrough) {
    Tensor mu(Shape({2, 2}), &backend, {0.3f, -1.1f, 2.0f, 0.7f});
    Tensor log_sigma(Shape({2, 2}), &backend, {0.4f, -0.9f, 1.2f, 0.1f});
    Tensor eps(Shape({2, 2}), &backend, {1.3f, -0.6f, 0.8f, -1.7f});
    Tensor grad_z(Shape({2, 2}), &backend, {1.0f, -0.5f, 0.3f, 0.8f});
    (void)reparam.forward(mu, log_sigma, eps);

    ReparamGrad grads = reparam.backward(grad_z);

    ASSERT_EQ(grads.grad_mu.numel(), 4);
    for (int64_t i = 0; i < grad_z.numel(); ++i) {
        EXPECT_FLOAT_EQ(grads.grad_mu.data()[i], grad_z.data()[i]) << "index " << i;
    }
    // ...and grad_mu is not trivially zero, so the check above is not vacuously satisfiable.
    EXPECT_GT(std::fabs(grads.grad_mu.data()[0]), 1e-4f);
}

TEST_F(ReparameterizeTest, BackwardGradLogSigmaMatchesHandDerivedValue) {
    // grad_log_sigma = grad_z * exp(log_sigma) * eps. With log_sigma = ln(2): 2*eps*grad_z.
    Tensor mu(Shape({1, 2}), &backend, {0.0f, 0.0f});
    Tensor log_sigma(Shape({1, 2}), &backend, {std::log(2.0f), 0.0f});
    Tensor eps(Shape({1, 2}), &backend, {3.0f, -1.5f});
    Tensor grad_z(Shape({1, 2}), &backend, {0.5f, 2.0f});
    (void)reparam.forward(mu, log_sigma, eps);

    ReparamGrad grads = reparam.backward(grad_z);

    EXPECT_NEAR(grads.grad_log_sigma.data()[0], 0.5f * 2.0f * 3.0f, 1e-5f);
    EXPECT_NEAR(grads.grad_log_sigma.data()[1], 2.0f * 1.0f * -1.5f, 1e-5f);
}

// ---------------------------------------------------------------------------------------
// Central finite-difference gradient check for grad_log_sigma.
//
// The fixture is deliberately non-uniform (N != latent_dim, log_sigma spanning both signs
// with |exp(log_sigma)| from ~0.4 to ~2.7, sign-alternating epsilon and a sign-alternating
// grad seed) rather than small-and-uniform: RWKVModule's mission found the hard way that a
// uniform small-scale fixture can make a check pass *vacuously* against the tolerance floor.
// Tolerance is relative (1e-3 + 2e-2*|numeric|) for the same established reason -- a fixed
// absolute tolerance is either vacuous for small gradients or spuriously tight for large
// ones. Non-vacuity verified by mutation probe (see the mission's close-out evidence).
// ---------------------------------------------------------------------------------------
namespace {

const std::vector<float>& fd_mu_values() {
    static const std::vector<float> v{0.80f, -0.60f, 1.20f, 0.50f, -0.90f, 1.10f};
    return v;
}
const std::vector<float>& fd_log_sigma_values() {
    static const std::vector<float> v{0.35f, -0.85f, 1.00f, 0.15f, -0.45f, 0.70f};
    return v;
}
const std::vector<float>& fd_epsilon_values() {
    static const std::vector<float> v{1.30f, -0.70f, 0.40f, -1.60f, 0.95f, -0.25f};
    return v;
}
const std::vector<float>& fd_grad_seed_values() {
    static const std::vector<float> v{1.00f, -0.50f, 0.30f, 0.80f, -0.20f, 0.90f};
    return v;
}

constexpr int64_t kN = 2;
constexpr int64_t kLatent = 3;

// L = sum_i z_i * seed_i, so dL/dz == seed and the analytic backward() can be seeded with it.
float scalar_loss(CPUBackend& backend, const std::vector<float>& log_sigma_values) {
    Reparameterize r(&backend);
    Tensor mu(Shape({kN, kLatent}), &backend, fd_mu_values());
    Tensor log_sigma(Shape({kN, kLatent}), &backend, log_sigma_values);
    Tensor eps(Shape({kN, kLatent}), &backend, fd_epsilon_values());
    Tensor z = r.forward(mu, log_sigma, eps);

    float total = 0.0f;
    for (int64_t i = 0; i < z.numel(); ++i) {
        total += z.data()[i] * fd_grad_seed_values()[static_cast<size_t>(i)];
    }
    return total;
}

float relative_tolerance(float numeric) {
    return 1e-3f + 2e-2f * std::fabs(numeric);
}

}  // namespace

TEST_F(ReparameterizeTest, BackwardGradLogSigmaMatchesFiniteDifference) {
    const float h = 1e-3f;
    Tensor mu(Shape({kN, kLatent}), &backend, fd_mu_values());
    Tensor log_sigma(Shape({kN, kLatent}), &backend, fd_log_sigma_values());
    Tensor eps(Shape({kN, kLatent}), &backend, fd_epsilon_values());
    Tensor grad_seed(Shape({kN, kLatent}), &backend, fd_grad_seed_values());

    (void)reparam.forward(mu, log_sigma, eps);
    ReparamGrad grads = reparam.backward(grad_seed);

    ASSERT_EQ(grads.grad_log_sigma.numel(), kN * kLatent);
    for (size_t idx = 0; idx < fd_log_sigma_values().size(); ++idx) {
        std::vector<float> plus = fd_log_sigma_values();
        std::vector<float> minus = fd_log_sigma_values();
        plus[idx] += h;
        minus[idx] -= h;
        const float numeric = (scalar_loss(backend, plus) - scalar_loss(backend, minus)) / (2.0f * h);
        EXPECT_NEAR(grads.grad_log_sigma.data()[static_cast<int64_t>(idx)], numeric, relative_tolerance(numeric))
            << "log_sigma index " << idx;
        // Every element's numeric gradient is well clear of the 1e-3 tolerance floor, so no
        // element of this check can pass vacuously.
        EXPECT_GT(std::fabs(numeric), 1e-2f) << "log_sigma index " << idx;
    }
}

using ReparameterizeDeathTest = ReparameterizeTest;

// GPU-native-kernels Mission 1b: the computation is device-generic, so mixed-device inputs
// are a reachable caller error -- rejected before any kernel sees two devices' pointers.
TEST_F(ReparameterizeTest, ForwardThrowsOnMixedDevices) {
    Tensor mu(Shape({2}), &backend, {0.5f, -0.5f});
    Tensor log_sigma(Shape({2}), &backend, {0.1f, 0.2f});
    Tensor epsilon(Shape({2}), &backend, {1.0f, -1.0f}, DeviceType::Cuda);
    EXPECT_THROW((void)reparam.forward(mu, log_sigma, epsilon), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
