/** @file gan_integration_test.cpp
 *  @brief End-to-end proof that BCEWithLogitsLoss + two ordinary MLPs + two independent SGD
 *         optimizers compose into a trainable GAN, and a regression test pinning this
 *         mission's one genuine resolved-under-ambiguity design point: the *extra*
 *         `D_optimizer.zero_grad()` that must follow the generator step.
 *
 *  A GAN needs no new Module type in this codebase -- generator and discriminator are both
 *  ordinary Linear/Relu stacks. What the per-class unit test structurally cannot prove is
 *  that the *training procedure* composes: that the discriminator's two-term loss really does
 *  sum in its gradient buffers (this codebase's accumulate-until-zero_grad semantics used as a
 *  feature), that the generator's gradient routes correctly *through* the discriminator, and
 *  that the gradient the generator step leaves behind inside the discriminator is discarded
 *  rather than silently fed into the discriminator's next update. That is this file's job.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "exai/bce_with_logits_loss.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/linear_module.hpp"
#include "exai/relu_module.hpp"
#include "exai/sequential_module.hpp"
#include "exai/sgd_optimizer.hpp"

namespace exai {
namespace {

constexpr int64_t kBatch = 4;
constexpr int64_t kNoise = 3;
constexpr int64_t kDataDim = 2;
constexpr int64_t kHidden = 6;

// Deterministic pseudo-random initialization. LinearModule zero-initializes, and a network
// built entirely from zero weights is a fixed point of its own gradient, so training would
// provably never move -- the same reason ToyVAE seeds itself this way (vae_integration_test).
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

/**
 * @brief Generator (noise -> 2D point) and discriminator (2D point -> 1 logit), both plain
 *        Linear->Relu->Linear stacks wrapped in SequentialModule so one optimizer call
 *        covers every parameter of one network and nothing of the other.
 */
class ToyGAN {
public:
    explicit ToyGAN(DeviceBackend* backend)
        : backend_(backend),
          g_in_(kNoise, kHidden, backend),
          g_relu_(backend),
          g_out_(kHidden, kDataDim, backend),
          d_in_(kDataDim, kHidden, backend),
          d_relu_(backend),
          d_out_(kHidden, 1, backend),
          generator_({&g_in_, &g_relu_, &g_out_}),
          discriminator_({&d_in_, &d_relu_, &d_out_}),
          bce_(backend) {
        g_in_.set_weight(lcg_values(kNoise * kHidden, 101u));
        g_out_.set_weight(lcg_values(kHidden * kDataDim, 202u));
        d_in_.set_weight(lcg_values(kDataDim * kHidden, 303u));
        d_out_.set_weight(lcg_values(kHidden * 1, 404u));
    }

    Module& generator() { return generator_; }
    Module& discriminator() { return discriminator_; }

    /**
     * @brief One discriminator step: loss_D = BCE(D(x_real), 1) + BCE(D(G(z)), 0).
     * @return loss_D before the update.
     * @note The two terms are never added by hand. D.backward(grad_real) accumulates into
     *       D's parameter-gradient buffers, then D.backward(grad_fake) accumulates further --
     *       this codebase's accumulate-until-zero_grad semantics is exactly the sum the
     *       objective calls for. That is why zero_grad() comes *after* step(), not before
     *       the first backward(): the accumulation between the two backward() calls is
     *       load-bearing here, unlike in a single-loss training loop.
     */
    float discriminator_step(const Tensor& x_real, const Tensor& z, SGDOptimizer& d_optimizer) {
        const float loss_value = accumulate_discriminator_grads(x_real, z);
        d_optimizer.step(discriminator_);
        d_optimizer.zero_grad(discriminator_);
        return loss_value;
    }

    /**
     * @brief One generator step, Goodfellow's non-saturating form: loss_G = BCE(D(G(z)), 1).
     * @param apply_discriminator_zero_grad_fix Whether to discard the gradient this step
     *        accumulated inside D. Always true in real training; the false branch exists
     *        only so the regression test below can exhibit the bug this fix prevents.
     * @return loss_G before the update.
     * @note `discriminator_.backward(grad_bce)` is required to route the gradient *through*
     *       D back to G's output -- but it also, unavoidably, accumulates into D's own
     *       parameter gradients. Those gradients belong to G's objective and must not feed
     *       D's next update, so D's optimizer is zeroed here too. This is the mission's
     *       documented gotcha; see BCEWithLogitsLoss's own @warning.
     */
    float generator_step(const Tensor& z, SGDOptimizer& g_optimizer, SGDOptimizer& d_optimizer,
                         bool apply_discriminator_zero_grad_fix) {
        Tensor fake = generator_.forward(z);
        Tensor logits = discriminator_.forward(fake);
        const float loss_value = bce_.forward(logits, labels(1.0f));

        Tensor grad_fake = discriminator_.backward(bce_.backward());
        (void)generator_.backward(grad_fake);

        g_optimizer.step(generator_);
        g_optimizer.zero_grad(generator_);

        if (apply_discriminator_zero_grad_fix) {
            d_optimizer.zero_grad(discriminator_);
        }
        return loss_value;
    }

    /**
     * @brief Runs the discriminator's forward/backward accumulation *only* -- no optimizer
     *        step, no zero_grad -- and returns D's resulting parameter gradients.
     * @note Deliberately shares discriminator_step()'s body so the snapshot is exactly the
     *       gradient a real next discriminator step would apply, not a re-derivation of it.
     */
    std::vector<float> next_discriminator_grads(const Tensor& x_real, const Tensor& z) {
        (void)accumulate_discriminator_grads(x_real, z);
        return discriminator_grads();
    }

    /** @brief Flattened copy of every discriminator parameter gradient, in parameters() order. */
    std::vector<float> discriminator_grads() {
        std::vector<float> flat;
        for (ParamRef p : discriminator_.parameters()) {
            for (int64_t i = 0; i < p.grad->numel(); ++i) {
                flat.push_back(p.grad->data()[i]);
            }
        }
        return flat;
    }

    /** @brief Flattened copy of every discriminator parameter value, in parameters() order. */
    std::vector<float> discriminator_params() {
        std::vector<float> flat;
        for (ParamRef p : discriminator_.parameters()) {
            for (int64_t i = 0; i < p.value->numel(); ++i) {
                flat.push_back(p.value->data()[i]);
            }
        }
        return flat;
    }

    /** @brief A (kBatch, 1) tensor of a constant label, for the BCE target argument. */
    Tensor labels(float value) {
        Tensor t(Shape({kBatch, 1}), backend_);
        t.fill(value);
        return t;
    }

private:
    float accumulate_discriminator_grads(const Tensor& x_real, const Tensor& z) {
        // Real branch first, fully (forward *and* backward), before the fake branch touches
        // D's forward cache -- D.backward() consumes the most recent forward()'s cache.
        Tensor logits_real = discriminator_.forward(x_real);
        const float loss_real = bce_.forward(logits_real, labels(1.0f));
        (void)discriminator_.backward(bce_.backward());

        Tensor fake = generator_.forward(z);
        Tensor logits_fake = discriminator_.forward(fake);
        const float loss_fake = bce_.forward(logits_fake, labels(0.0f));
        (void)discriminator_.backward(bce_.backward());

        return loss_real + loss_fake;
    }

    DeviceBackend* backend_;
    LinearModule g_in_;
    ReluModule g_relu_;
    LinearModule g_out_;
    LinearModule d_in_;
    ReluModule d_relu_;
    LinearModule d_out_;
    SequentialModule generator_;
    SequentialModule discriminator_;
    BCEWithLogitsLoss bce_;
};

class GANIntegrationTest : public ::testing::Test {
protected:
    CPUBackend backend;

    /** @brief A fixed tight cluster of 2D points near (1, 1) -- the "real" data distribution. */
    Tensor real_data() {
        return Tensor(Shape({kBatch, kDataDim}), &backend,
                      {1.00f, 1.00f,   //
                       0.90f, 1.10f,   //
                       1.05f, 0.95f,   //
                       0.95f, 1.05f});
    }

    /**
     * @brief Mean squared distance from the generator's current output to the real cluster's
     *        centroid (1, 1) -- a direct, adversarial-training-appropriate progress measure.
     */
    float generator_distance_to_real_cluster(ToyGAN& gan) {
        Tensor fake = gan.generator().forward(noise());
        float total = 0.0f;
        for (int64_t b = 0; b < kBatch; ++b) {
            for (int64_t d = 0; d < kDataDim; ++d) {
                const float diff = fake.data()[b * kDataDim + d] - 1.0f;
                total += diff * diff;
            }
        }
        return total / static_cast<float>(kBatch * kDataDim);
    }

    /** @brief Fixed noise vectors. Deterministic by design, exactly as ToyVAE's epsilon is. */
    Tensor noise() {
        return Tensor(Shape({kBatch, kNoise}), &backend,
                      {0.50f, -0.30f, 0.80f,   //
                       0.20f, -0.60f, 0.40f,   //
                       0.10f, -0.90f, 0.70f,   //
                       -0.40f, 0.30f, -0.20f});
    }
};

// The composition proof: several alternating D/G steps, two independent optimizers, both
// losses finite and non-NaN throughout. Adversarial training is not a descent problem -- the
// two objectives oppose each other, so (unlike the VAE's combined objective) there is no
// monotone "loss decreases" invariant to assert. Finiteness plus the discriminator actually
// learning to separate real from fake is what is genuinely true here.
TEST_F(GANIntegrationTest, AlternatingTrainingKeepsBothLossesFinite) {
    constexpr int kSteps = 2000;
    ToyGAN gan(&backend);
    SGDOptimizer d_optimizer(0.05f);
    SGDOptimizer g_optimizer(0.05f);
    Tensor x_real = real_data();
    Tensor z = noise();

    float first_d_loss = 0.0f;
    float first_g_loss = 0.0f;
    float last_d_loss = 0.0f;
    float last_g_loss = 0.0f;
    const float initial_distance = generator_distance_to_real_cluster(gan);

    for (int step = 0; step < kSteps; ++step) {
        const float d_loss = gan.discriminator_step(x_real, z, d_optimizer);
        const float g_loss = gan.generator_step(z, g_optimizer, d_optimizer, /*apply_fix=*/true);

        ASSERT_TRUE(std::isfinite(d_loss)) << "discriminator loss non-finite at step " << step;
        ASSERT_TRUE(std::isfinite(g_loss)) << "generator loss non-finite at step " << step;
        ASSERT_FALSE(std::isnan(d_loss)) << "discriminator loss NaN at step " << step;
        ASSERT_FALSE(std::isnan(g_loss)) << "generator loss NaN at step " << step;
        // BCE is non-negative by construction; a negative value means the stable rewrite is
        // wrong, which finiteness alone would not catch.
        ASSERT_GE(d_loss, 0.0f) << "step " << step;
        ASSERT_GE(g_loss, 0.0f) << "step " << step;

        if (step == 0) {
            first_d_loss = d_loss;
            first_g_loss = g_loss;
        }
        last_d_loss = d_loss;
        last_g_loss = g_loss;
    }

    const float final_distance = generator_distance_to_real_cluster(gan);
    std::cout << "[GAN] D loss " << first_d_loss << " -> " << last_d_loss << " | G loss " << first_g_loss << " -> "
              << last_g_loss << " | G distance to real cluster " << initial_distance << " -> " << final_distance
              << std::endl;

    // Non-vacuity. Adversarial training has no monotone loss invariant to assert -- the two
    // objectives oppose each other, and a healthy run converges toward the 2*ln(2) ~ 1.386
    // equilibrium where D cannot separate, so "loss_D decreases" is simply not a true
    // property of a working GAN. What *is* true, and what a no-op training loop could never
    // satisfy, is that the generator's samples move toward the real data cluster.
    EXPECT_LT(final_distance, initial_distance);
    EXPECT_LT(final_distance, 0.5f * initial_distance);
}

// Sanity counterpart: forward passes alone must not move anything, so any difference observed
// above is attributable to the optimizer steps.
TEST_F(GANIntegrationTest, RepeatedGeneratorForwardWithoutStepsIsStationary) {
    ToyGAN gan(&backend);
    Tensor z = noise();

    Tensor first = gan.generator().forward(z);
    for (int i = 0; i < 5; ++i) {
        Tensor again = gan.generator().forward(z);
        for (int64_t k = 0; k < first.numel(); ++k) {
            EXPECT_FLOAT_EQ(again.data()[k], first.data()[k]) << "element " << k;
        }
    }
}

// The discriminator's two-term loss relies on gradient accumulation being a *feature*:
// D.backward(grad_real) then D.backward(grad_fake) must sum in D's buffers. Pin that directly
// -- if some future change made backward() overwrite rather than accumulate, the training
// test above would still pass (it would just be optimizing the fake term alone).
TEST_F(GANIntegrationTest, DiscriminatorGradientIsTheSumOfBothLossTerms) {
    ToyGAN gan(&backend);
    Tensor x_real = real_data();
    Tensor z = noise();
    SGDOptimizer d_optimizer(0.05f);

    // Real term alone.
    Tensor logits_real = gan.discriminator().forward(x_real);
    BCEWithLogitsLoss bce_real(&backend);
    (void)bce_real.forward(logits_real, gan.labels(1.0f));
    (void)gan.discriminator().backward(bce_real.backward());
    const std::vector<float> real_only = gan.discriminator_grads();
    d_optimizer.zero_grad(gan.discriminator());

    // Fake term alone.
    Tensor fake = gan.generator().forward(z);
    Tensor logits_fake = gan.discriminator().forward(fake);
    BCEWithLogitsLoss bce_fake(&backend);
    (void)bce_fake.forward(logits_fake, gan.labels(0.0f));
    (void)gan.discriminator().backward(bce_fake.backward());
    const std::vector<float> fake_only = gan.discriminator_grads();
    d_optimizer.zero_grad(gan.discriminator());

    // Both terms, accumulated the way discriminator_step() does it.
    const std::vector<float> combined = gan.next_discriminator_grads(x_real, z);

    ASSERT_EQ(combined.size(), real_only.size());
    ASSERT_EQ(combined.size(), fake_only.size());
    float max_term = 0.0f;
    for (size_t i = 0; i < combined.size(); ++i) {
        EXPECT_NEAR(combined[i], real_only[i] + fake_only[i], 1e-5f) << "param element " << i;
        max_term = std::max(max_term, std::fabs(real_only[i]));
    }
    // Non-vacuity: the real term is genuinely non-zero, so "sum" is not trivially "fake only".
    EXPECT_GT(max_term, 1e-3f);
}

// ---------------------------------------------------------------------------------------
// THE GOTCHA, as a regression test rather than prose.
//
// The generator step must call discriminator_.backward() to route its gradient through D --
// which also accumulates into D's *own* parameter gradients. If D's optimizer is not zeroed
// after that step, those generator-objective gradients survive into D's next step and are
// applied to D's weights. Both GANs below hold bit-identical parameters at the comparison
// point (a generator step never updates D's *values*, and zero_grad touches only gradient
// buffers), so the only possible source of any difference is the leftover contamination.
// ---------------------------------------------------------------------------------------
TEST_F(GANIntegrationTest, SkippingDiscriminatorZeroGradAfterGeneratorStepCorruptsIt) {
    Tensor x_real = real_data();
    Tensor z = noise();

    ToyGAN fixed(&backend);
    ToyGAN buggy(&backend);
    SGDOptimizer fixed_d(0.05f), fixed_g(0.05f);
    SGDOptimizer buggy_d(0.05f), buggy_g(0.05f);

    (void)fixed.discriminator_step(x_real, z, fixed_d);
    (void)buggy.discriminator_step(x_real, z, buggy_d);
    (void)fixed.generator_step(z, fixed_g, fixed_d, /*apply_fix=*/true);
    (void)buggy.generator_step(z, buggy_g, buggy_d, /*apply_fix=*/false);

    // Precondition of the whole comparison: the two discriminators are numerically identical.
    const std::vector<float> fixed_params = fixed.discriminator_params();
    const std::vector<float> buggy_params = buggy.discriminator_params();
    ASSERT_EQ(fixed_params.size(), buggy_params.size());
    for (size_t i = 0; i < fixed_params.size(); ++i) {
        ASSERT_FLOAT_EQ(fixed_params[i], buggy_params[i]) << "discriminator param " << i;
    }

    // The fixed run starts its next step from a zeroed gradient buffer...
    const std::vector<float> fixed_start = fixed.discriminator_grads();
    for (size_t i = 0; i < fixed_start.size(); ++i) {
        EXPECT_FLOAT_EQ(fixed_start[i], 0.0f) << "fixed run leftover at " << i;
    }
    // ...while the buggy run is carrying the generator step's gradient. That leftover must be
    // genuinely non-zero, or this whole test would be vacuous -- it is what proves
    // D.backward() during the generator step really does write into D's parameter gradients.
    const std::vector<float> contamination = buggy.discriminator_grads();
    float max_contamination = 0.0f;
    for (float v : contamination) {
        max_contamination = std::max(max_contamination, std::fabs(v));
    }
    std::cout << "[GAN] leftover contamination in D after an unfixed generator step: " << max_contamination
              << std::endl;
    ASSERT_GT(max_contamination, 1e-3f);

    // Now run each one's next discriminator gradient accumulation. Same inputs, same
    // parameters -- the gradients must nevertheless differ, by exactly the contamination.
    const std::vector<float> fixed_next = fixed.next_discriminator_grads(x_real, z);
    const std::vector<float> buggy_next = buggy.next_discriminator_grads(x_real, z);

    ASSERT_EQ(fixed_next.size(), buggy_next.size());
    float max_difference = 0.0f;
    for (size_t i = 0; i < fixed_next.size(); ++i) {
        EXPECT_NEAR(buggy_next[i], fixed_next[i] + contamination[i], 1e-5f)
            << "param element " << i << ": the difference is exactly the leftover gradient";
        max_difference = std::max(max_difference, std::fabs(buggy_next[i] - fixed_next[i]));
    }
    std::cout << "[GAN] max |buggy next-step grad - clean next-step grad| = " << max_difference << std::endl;

    // The headline assertion: skipping D_optimizer.zero_grad() after the generator step makes
    // D's next gradient differ from the clean-start case. The fix is load-bearing.
    EXPECT_GT(max_difference, 1e-3f);
}

}  // namespace
}  // namespace exai
