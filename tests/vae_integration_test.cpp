/** @file vae_integration_test.cpp
 *  @brief End-to-end proof that Reparameterize + KLDivergenceLoss compose into a trainable
 *         VAE: encoder -> reparameterize -> decoder, trained with SGD on toy data, asserting
 *         the combined (reconstruction + KL) objective decreases.
 *
 *  The per-class unit tests prove each piece is independently correct. They cannot prove the
 *  pieces *compose* -- that the two gradient pairs add up on the right tensors, that the
 *  reparameterization chain rule lines up with what the decoder hands back, that the whole
 *  thing is a descent direction. That is this file's job, and it replaces a standalone
 *  examples/ demo binary for this mission's scope (mission_vae_module.md).
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/kl_divergence_loss.hpp"
#include "exai/linear_module.hpp"
#include "exai/mse_loss.hpp"
#include "exai/relu_module.hpp"
#include "exai/reparameterize.hpp"
#include "exai/sgd_optimizer.hpp"

namespace exai {
namespace {

constexpr int64_t kBatch = 4;
constexpr int64_t kInput = 4;
constexpr int64_t kHidden = 6;
constexpr int64_t kLatent = 2;
constexpr float kKLWeight = 1.0f;

// Deterministic pseudo-random initialization. LinearModule zero-initializes, and a VAE built
// entirely from zero weights is a fixed point of its own gradient (every parameter gradient
// is a product containing a zero activation or a zero weight), so training would provably
// never move. A fixed LCG keeps the test reproducible without pulling in <random>'s
// implementation-defined engines -- the same "deliberately chosen fixed weights" disposition
// XorNetwork's constructor uses, just generated rather than hand-listed.
std::vector<float> lcg_values(size_t count, uint32_t seed) {
    std::vector<float> values;
    values.reserve(count);
    uint32_t state = seed;
    for (size_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        const float unit = static_cast<float>((state >> 8) & 0xFFFFu) / 65535.0f;  // [0, 1]
        values.push_back((unit - 0.5f) * 1.2f);                                    // [-0.6, 0.6]
    }
    return values;
}

/** @brief Encoder trunk + separate mu/log_sigma heads + reparameterization + decoder. */
class ToyVAE {
public:
    explicit ToyVAE(DeviceBackend* backend)
        : backend_(backend),
          enc_(kInput, kHidden, backend),
          enc_relu_(backend),
          mu_head_(kHidden, kLatent, backend),
          log_sigma_head_(kHidden, kLatent, backend),
          reparam_(backend),
          dec_(kLatent, kHidden, backend),
          dec_relu_(backend),
          dec_out_(kHidden, kInput, backend),
          recon_loss_(backend),
          kl_loss_(backend) {
        enc_.set_weight(lcg_values(kInput * kHidden, 11u));
        mu_head_.set_weight(lcg_values(kHidden * kLatent, 22u));
        log_sigma_head_.set_weight(lcg_values(kHidden * kLatent, 33u));
        dec_.set_weight(lcg_values(kLatent * kHidden, 44u));
        dec_out_.set_weight(lcg_values(kHidden * kInput, 55u));
    }

    /** @brief Runs the full pipeline and returns the combined reconstruction + KL objective. */
    float forward(const Tensor& x, const Tensor& epsilon) {
        Tensor h = enc_relu_.forward(enc_.forward(x));
        Tensor mu = mu_head_.forward(h);
        Tensor log_sigma = log_sigma_head_.forward(h);
        Tensor z = reparam_.forward(mu, log_sigma, epsilon);
        Tensor recon = dec_out_.forward(dec_relu_.forward(dec_.forward(z)));

        const float recon_value = recon_loss_.forward(recon, x);
        const float kl_value = kl_loss_.forward(mu, log_sigma);
        return recon_value + kKLWeight * kl_value;
    }

    /**
     * @brief The KL term alone, for the current parameters -- encoder heads only, no decoder.
     * @note Uses its own probe loss object so it cannot disturb the training-path caches.
     */
    float kl_term(const Tensor& x) {
        Tensor h = enc_relu_.forward(enc_.forward(x));
        Tensor mu = mu_head_.forward(h);
        Tensor log_sigma = log_sigma_head_.forward(h);
        KLDivergenceLoss probe(backend_);
        return probe.forward(mu, log_sigma);
    }

    /** @brief One full training step; returns the combined loss *before* the update. */
    float train_step(const Tensor& x, const Tensor& epsilon, SGDOptimizer& optimizer) {
        // Gradients accumulate across backward() calls by design (Tensor::accumulate), so a
        // step must zero them first or every step silently adds onto the previous one --
        // exactly the bug XorNetwork::train_step documents finding the hard way.
        for (LinearModule* layer : layers()) {
            optimizer.zero_grad(*layer);
        }

        const float loss_value = forward(x, epsilon);

        // Reconstruction branch: MSE -> decoder -> grad w.r.t. z.
        Tensor grad_recon = recon_loss_.backward();
        Tensor grad_dec_relu_out = dec_out_.backward(grad_recon);
        Tensor grad_dec_out = dec_relu_.backward(grad_dec_relu_out);
        Tensor grad_z = dec_.backward(grad_dec_out);

        // The two gradient pairs meet here: the reconstruction path arrives through the
        // reparameterization chain rule, the KL path arrives directly on (mu, log_sigma).
        // This addition is the thing the unit tests structurally cannot check.
        ReparamGrad from_recon = reparam_.backward(grad_z);
        ReparamGrad from_kl = kl_loss_.backward();
        Tensor grad_mu = from_recon.grad_mu;
        Tensor grad_log_sigma = from_recon.grad_log_sigma;
        for (int64_t i = 0; i < grad_mu.numel(); ++i) {
            grad_mu.data()[i] += kKLWeight * from_kl.grad_mu.data()[i];
            grad_log_sigma.data()[i] += kKLWeight * from_kl.grad_log_sigma.data()[i];
        }

        // Both heads hang off the same encoder trunk, so their input gradients sum.
        Tensor grad_h = mu_head_.backward(grad_mu);
        grad_h.accumulate(log_sigma_head_.backward(grad_log_sigma));
        (void)enc_.backward(enc_relu_.backward(grad_h));

        for (LinearModule* layer : layers()) {
            optimizer.step(*layer);
        }
        return loss_value;
    }

private:
    std::vector<LinearModule*> layers() {
        return {&enc_, &mu_head_, &log_sigma_head_, &dec_, &dec_out_};
    }

    DeviceBackend* backend_;
    LinearModule enc_;
    ReluModule enc_relu_;
    LinearModule mu_head_;
    LinearModule log_sigma_head_;
    Reparameterize reparam_;
    LinearModule dec_;
    ReluModule dec_relu_;
    LinearModule dec_out_;
    MSELoss recon_loss_;
    KLDivergenceLoss kl_loss_;
};

class VAEIntegrationTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // Four fixed toy vectors to reconstruct, and a fixed epsilon (this codebase's
    // reparameterization takes epsilon as a caller-supplied deterministic input by design,
    // which is exactly what makes this training run reproducible).
    Tensor toy_data() {
        return Tensor(Shape({kBatch, kInput}), &backend,
                      {0.90f, 0.10f, -0.40f, 0.60f,   //
                       -0.70f, 0.50f, 0.80f, -0.20f,  //
                       0.30f, -0.90f, 0.20f, 1.00f,   //
                       -0.50f, 0.70f, -0.10f, 0.40f});
    }

    Tensor fixed_epsilon() {
        return Tensor(Shape({kBatch, kLatent}), &backend,
                      {0.50f, -0.30f, 0.80f, 0.20f, -0.60f, 0.40f, 0.10f, -0.90f});
    }
};

TEST_F(VAEIntegrationTest, CombinedLossDecreasesOverTraining) {
    constexpr int kSteps = 1000;
    constexpr float kLearningRate = 0.02f;

    ToyVAE vae(&backend);
    SGDOptimizer optimizer(kLearningRate);
    Tensor x = toy_data();
    Tensor epsilon = fixed_epsilon();

    const float initial_loss = vae.forward(x, epsilon);
    float mid_loss = 0.0f;
    float last_loss = initial_loss;
    for (int step = 0; step < kSteps; ++step) {
        last_loss = vae.train_step(x, epsilon, optimizer);
        if (step == kSteps / 2) {
            mid_loss = last_loss;
        }
    }
    const float final_loss = vae.forward(x, epsilon);

    std::cout << "[VAE] initial combined loss " << initial_loss << " | mid (step " << kSteps / 2 << ") " << mid_loss
              << " | final " << final_loss << std::endl;

    EXPECT_GT(initial_loss, 0.0f);
    EXPECT_TRUE(std::isfinite(final_loss));
    EXPECT_LT(mid_loss, initial_loss);
    EXPECT_LT(final_loss, mid_loss);
    // A meaningful decrease, not merely a non-increase that a no-op update would also satisfy.
    EXPECT_LT(final_loss, 0.5f * initial_loss);
}

// The decrease above must come from the *gradients*, not from any descent an untrained
// forward pass would show anyway. Re-running forward() repeatedly without optimizer steps
// must leave the loss bit-identical -- if it drifts, the "training" result above proves
// nothing.
TEST_F(VAEIntegrationTest, ForwardWithoutOptimizerStepsIsStationary) {
    ToyVAE vae(&backend);
    Tensor x = toy_data();
    Tensor epsilon = fixed_epsilon();

    const float first = vae.forward(x, epsilon);
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(vae.forward(x, epsilon), first);
    }
}

// Both terms of the objective must actually be exercised: a pipeline that silently dropped
// the KL branch would still reconstruct, and the loss would still fall. Pin that the KL term
// is non-zero at initialization (so it genuinely contributes to the combined objective) and
// that the trained posterior has been pulled toward the N(0, I) prior.
TEST_F(VAEIntegrationTest, KLTermIsNonZeroAtInitAndShrinksWithTraining) {
    constexpr int kSteps = 1000;
    ToyVAE vae(&backend);
    SGDOptimizer optimizer(0.02f);
    Tensor x = toy_data();
    Tensor epsilon = fixed_epsilon();

    const float initial_kl = vae.kl_term(x);
    for (int step = 0; step < kSteps; ++step) {
        (void)vae.train_step(x, epsilon, optimizer);
    }
    const float final_kl = vae.kl_term(x);

    std::cout << "[VAE] initial KL term " << initial_kl << " | final " << final_kl << std::endl;

    // Non-trivially present at initialization -- otherwise "the KL branch trains" would be
    // satisfiable by a pipeline that never wired the KL gradients in at all.
    EXPECT_GT(initial_kl, 1e-2f);
    // ...and the posterior has been pulled toward the N(0, I) prior, which only happens if
    // KLDivergenceLoss::backward()'s gradients actually reached mu_head_/log_sigma_head_.
    EXPECT_LT(final_kl, initial_kl);
    EXPECT_GE(final_kl, 0.0f);
    EXPECT_TRUE(std::isfinite(final_kl));
}

}  // namespace
}  // namespace exai
