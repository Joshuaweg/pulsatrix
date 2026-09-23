/** @file diffusion_integration_test.cpp
 *  @brief End-to-end proof that NoiseSchedule + SinusoidalTimestepEmbedding + the existing
 *         LinearModule/ReluModule/MSELoss/SGDOptimizer stack compose into a real, working
 *         DDPM: an epsilon_theta denoiser trained on toy noised data with the published
 *         L_simple objective, followed by a full T-step reverse sampling loop.
 *
 *  Both halves are required and neither substitutes for the other. The training half proves
 *  the *objective* is a descent direction through a network that consumes x_t and a timestep
 *  embedding. The sampling half proves the *process* -- T chained denoise_step calls, each
 *  feeding the previous step's output back through the network -- actually runs to completion
 *  and stays bounded, which no per-step unit test can establish: the per-step tests check one
 *  application of the formula in isolation, and only the chained loop exercises the
 *  step-to-step composition (including the 1/sqrt(alpha_t) amplification compounding T times
 *  and the deterministic t == 1 final step) at all.
 *
 *  Non-vacuity of the *value* checks lives in noise_schedule_test.cpp, which is where a
 *  mutation probe (non-cumulative alpha_bar; flipped sign on denoise_step's eps term) was
 *  confirmed to fail 6 tests. This file's assertions are boundedness/shape/dependence
 *  assertions, deliberately weaker -- a sampling loop is not expected to have a closed-form
 *  expected output.
 *
 *  Backbone scope: a plain Linear->Relu->Linear MLP on small flat vectors, with the timestep
 *  embedding *concatenated* onto the input, not a convolutional U-Net with FiLM conditioning.
 *  That is a deliberate, pre-resolved scope cut (mission_diffusion_module.md's "Scope cut"
 *  section) overriding the research pass's own recommendation: this codebase has no
 *  up/downsampling or skip-connection primitives to build a real U-Net from, and the
 *  noise-schedule math, training objective and sampling process being proved here are all
 *  architecture-agnostic. Do not "improve" this back toward a U-Net.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/linear_module.hpp"
#include "exai/mse_loss.hpp"
#include "exai/noise_schedule.hpp"
#include "exai/relu_module.hpp"
#include "exai/sgd_optimizer.hpp"
#include "exai/sinusoidal_timestep_embedding.hpp"

namespace exai {
namespace {

constexpr int64_t kBatch = 4;
constexpr int64_t kDim = 4;      ///< Data dimension -- the toy "image" is a 4-vector.
constexpr int64_t kEmbDim = 4;   ///< Timestep embedding width.
constexpr int64_t kHidden = 16;  ///< Denoiser hidden width.
constexpr int64_t kTimesteps = 10;
constexpr float kBetaStart = 1e-4f;
// Steeper than the paper's 0.02 end point, deliberately: with only T = 10 steps a 0.02 end
// would leave alpha_bar_T ~ 0.94, i.e. x_T barely noised at all, and the "reverse process"
// would be a near no-op that proves nothing. 0.2 over 10 steps gives alpha_bar_T ~ 0.35 --
// a genuinely destroyed signal to sample back from.
constexpr float kBetaEnd = 0.2f;

/** @brief Deterministic pseudo-random values -- see vae_integration_test.cpp's identical LCG. */
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
 * @brief The epsilon_theta denoiser: concat(x_t, timestep_embedding(t)) -> Linear -> Relu ->
 *        Linear -> predicted noise, same shape as x_t.
 */
class ToyDenoiser {
public:
    explicit ToyDenoiser(DeviceBackend* backend)
        : backend_(backend),
          in_(kDim + kEmbDim, kHidden, backend),
          relu_(backend),
          out_(kHidden, kDim, backend),
          loss_(backend) {
        // LinearModule zero-initializes, and a zero-weight MLP is a fixed point of its own
        // gradient (every weight gradient is a product containing a zero activation or a zero
        // weight), so training would provably never move without this.
        in_.set_weight(lcg_values(static_cast<size_t>((kDim + kEmbDim) * kHidden), 101u));
        out_.set_weight(lcg_values(static_cast<size_t>(kHidden * kDim), 202u));
    }

    /** @brief Predicts the noise in x_t at timestep t. */
    Tensor predict(const Tensor& x_t, int64_t t) {
        return out_.forward(relu_.forward(in_.forward(conditioned_input(x_t, t))));
    }

    /**
     * @brief One L_simple training step on a single (x0, epsilon, t) triple.
     *        loss = MSE(epsilon, epsilon_theta(add_noise(x0, epsilon, t), t)).
     * @return The loss value *before* this step's parameter update.
     */
    float train_step(const NoiseSchedule& schedule, const Tensor& x0, const Tensor& epsilon, int64_t t,
                     SGDOptimizer& optimizer) {
        // Gradients accumulate across backward() calls by design (Tensor::accumulate), so a
        // step must zero them first -- the bug XorNetwork::train_step documents finding the
        // hard way.
        optimizer.zero_grad(in_);
        optimizer.zero_grad(out_);

        const Tensor x_t = schedule.add_noise(x0, epsilon, t);
        const Tensor predicted = predict(x_t, t);
        // The DDPM training objective is literally an MSE between true and predicted noise --
        // no diffusion-specific loss class exists or is needed.
        const float loss_value = loss_.forward(predicted, epsilon);

        (void)in_.backward(relu_.backward(out_.backward(loss_.backward())));

        optimizer.step(in_);
        optimizer.step(out_);
        return loss_value;
    }

    /** @brief The objective on one triple, without touching any parameter (probe loss object). */
    float evaluate(const NoiseSchedule& schedule, const Tensor& x0, const Tensor& epsilon, int64_t t) {
        const Tensor x_t = schedule.add_noise(x0, epsilon, t);
        const Tensor predicted = predict(x_t, t);
        MSELoss probe(backend_);
        return probe.forward(predicted, epsilon);
    }

private:
    /** @brief Builds (N, kDim + kEmbDim) by appending the timestep embedding row to every example. */
    Tensor conditioned_input(const Tensor& x_t, int64_t t) {
        const Tensor embedding = SinusoidalTimestepEmbedding(t, kEmbDim, backend_);
        const int64_t batch = x_t.shape().dim(0);
        Tensor conditioned(Shape({batch, kDim + kEmbDim}), backend_);
        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t d = 0; d < kDim; ++d) {
                conditioned.data()[n * (kDim + kEmbDim) + d] = x_t.data()[n * kDim + d];
            }
            for (int64_t e = 0; e < kEmbDim; ++e) {
                // The same embedding row is shared by every example in the batch: one
                // timestep conditions the whole batch, which is exactly how a DDPM training
                // step over a single t works.
                conditioned.data()[n * (kDim + kEmbDim) + kDim + e] = embedding.data()[e];
            }
        }
        return conditioned;
    }

    DeviceBackend* backend_;
    LinearModule in_;
    ReluModule relu_;
    LinearModule out_;
    MSELoss loss_;
};

class DiffusionIntegrationTest : public ::testing::Test {
protected:
    CPUBackend backend;
    NoiseSchedule schedule{kTimesteps, kBetaStart, kBetaEnd};

    /** @brief The fixed toy dataset the denoiser learns to clean. */
    Tensor toy_data() {
        return Tensor(Shape({kBatch, kDim}), &backend,
                      {0.90f, 0.10f, -0.40f, 0.60f,   //
                       -0.70f, 0.50f, 0.80f, -0.20f,  //
                       0.30f, -0.90f, 0.20f, 1.00f,   //
                       -0.50f, 0.70f, -0.10f, 0.40f});
    }

    /**
     * @brief The fixed (epsilon, t) half of each training triple.
     * @note epsilon is caller-supplied and deterministic -- the same convention
     *       Reparameterize established, which is what makes this whole training run
     *       reproducible rather than merely "usually passes".
     */
    std::vector<std::pair<Tensor, int64_t>> training_triples() {
        std::vector<std::pair<Tensor, int64_t>> triples;
        const int64_t timesteps[] = {1, 3, 6, 10};
        uint32_t seed = 7u;
        for (int64_t t : timesteps) {
            triples.emplace_back(Tensor(Shape({kBatch, kDim}), &backend,
                                        lcg_values(static_cast<size_t>(kBatch * kDim), seed)),
                                 t);
            seed += 13u;
        }
        return triples;
    }

    /** @brief Mean objective across every training triple -- the reported "training loss". */
    float epoch_loss(ToyDenoiser& denoiser, const Tensor& x0,
                     const std::vector<std::pair<Tensor, int64_t>>& triples) {
        float total = 0.0f;
        for (const auto& triple : triples) {
            total += denoiser.evaluate(schedule, x0, triple.first, triple.second);
        }
        return total / static_cast<float>(triples.size());
    }
};

// ---------------------------------------------------------------------------------------
// Half 1 of the mission's end-to-end proof: the L_simple objective trains.
// ---------------------------------------------------------------------------------------

TEST_F(DiffusionIntegrationTest, NoisePredictionLossDecreasesOverTraining) {
    constexpr int kEpochs = 3000;
    constexpr float kLearningRate = 0.02f;

    ToyDenoiser denoiser(&backend);
    SGDOptimizer optimizer(kLearningRate);
    Tensor x0 = toy_data();
    auto triples = training_triples();

    const float initial_loss = epoch_loss(denoiser, x0, triples);
    float mid_loss = 0.0f;
    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        for (const auto& triple : triples) {
            (void)denoiser.train_step(schedule, x0, triple.first, triple.second, optimizer);
        }
        if (epoch == kEpochs / 2) {
            mid_loss = epoch_loss(denoiser, x0, triples);
        }
    }
    const float final_loss = epoch_loss(denoiser, x0, triples);

    std::cout << "[Diffusion] initial L_simple " << initial_loss << " | mid (epoch " << kEpochs / 2 << ") "
              << mid_loss << " | final " << final_loss << std::endl;

    EXPECT_GT(initial_loss, 0.0f);
    EXPECT_TRUE(std::isfinite(final_loss));
    EXPECT_LT(mid_loss, initial_loss);
    EXPECT_LT(final_loss, mid_loss);
    // A meaningful decrease, not merely a non-increase a no-op update would also satisfy.
    EXPECT_LT(final_loss, 0.5f * initial_loss);
}

// The decrease above must come from the gradients, not from drift a repeated forward pass
// would show anyway.
TEST_F(DiffusionIntegrationTest, EvaluationWithoutOptimizerStepsIsStationary) {
    ToyDenoiser denoiser(&backend);
    Tensor x0 = toy_data();
    auto triples = training_triples();

    const float first = epoch_loss(denoiser, x0, triples);
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(epoch_loss(denoiser, x0, triples), first);
    }
}

// The timestep conditioning must actually be *used*: a denoiser that ignored its embedding
// slots would still train (it would learn the t-averaged noise predictor) and the loss would
// still fall, so the training test alone cannot catch a dropped embedding.
TEST_F(DiffusionIntegrationTest, PredictionDependsOnTheTimestep) {
    ToyDenoiser denoiser(&backend);
    Tensor x_t = toy_data();

    Tensor early = denoiser.predict(x_t, 1);
    Tensor late = denoiser.predict(x_t, kTimesteps);

    ASSERT_EQ(early.numel(), late.numel());
    float max_difference = 0.0f;
    for (int64_t i = 0; i < early.numel(); ++i) {
        max_difference = std::fmax(max_difference, std::fabs(early.data()[i] - late.data()[i]));
    }
    EXPECT_GT(max_difference, 1e-3f);
}

// ---------------------------------------------------------------------------------------
// Half 2 of the mission's end-to-end proof: the full reverse sampling process runs.
// ---------------------------------------------------------------------------------------

TEST_F(DiffusionIntegrationTest, FullReverseSamplingLoopProducesFiniteSample) {
    constexpr int kEpochs = 3000;
    ToyDenoiser denoiser(&backend);
    SGDOptimizer optimizer(0.02f);
    Tensor x0 = toy_data();
    auto triples = training_triples();

    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        for (const auto& triple : triples) {
            (void)denoiser.train_step(schedule, x0, triple.first, triple.second, optimizer);
        }
    }

    // A fixed, fully-noised starting point x_T -- caller-supplied rather than sampled, same
    // determinism convention as epsilon everywhere else in this mission.
    Tensor x(Shape({kBatch, kDim}), &backend, lcg_values(static_cast<size_t>(kBatch * kDim), 909u));
    Tensor zeros(Shape({kBatch, kDim}), &backend);
    zeros.fill(0.0f);

    for (int64_t t = kTimesteps; t >= 1; --t) {
        const Tensor predicted = denoiser.predict(x, t);
        // z is zero at the final step by the published algorithm -- that step is deterministic.
        const Tensor z = (t > 1) ? Tensor(Shape({kBatch, kDim}), &backend,
                                          lcg_values(static_cast<size_t>(kBatch * kDim),
                                                     static_cast<uint32_t>(1000 + t)))
                                 : zeros;
        Tensor next = schedule.denoise_step(x, predicted, z, t);
        for (int64_t i = 0; i < next.numel(); ++i) {
            ASSERT_TRUE(std::isfinite(next.data()[i])) << "t = " << t << ", index " << i;
        }
        x = next;
    }

    // Shape is preserved end to end -- x_0 has exactly the shape of the data the model was
    // trained on.
    EXPECT_EQ(x.shape().rank(), 2);
    EXPECT_EQ(x.shape().dim(0), kBatch);
    EXPECT_EQ(x.shape().dim(1), kDim);

    float max_magnitude = 0.0f;
    for (int64_t i = 0; i < x.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(x.data()[i])) << "index " << i;
        EXPECT_FALSE(std::isnan(x.data()[i])) << "index " << i;
        max_magnitude = std::fmax(max_magnitude, std::fabs(x.data()[i]));
    }
    std::cout << "[Diffusion] reverse sampling over " << kTimesteps << " steps -> max |x_0| = " << max_magnitude
              << std::endl;
    // Not merely finite: a sampler that diverged geometrically over T steps would still be
    // "finite" for small T. Pin it to the same order of magnitude as the training data.
    EXPECT_LT(max_magnitude, 50.0f);
    EXPECT_GT(max_magnitude, 0.0f);
}

// The sampling loop must be driven by the schedule, not by the starting point alone: two
// different x_T values must lead to two different x_0 values (a reverse process that
// collapsed to a constant regardless of input would still pass the finiteness assertions).
TEST_F(DiffusionIntegrationTest, ReverseSamplingIsDrivenByItsStartingPoint) {
    ToyDenoiser denoiser(&backend);
    Tensor zeros(Shape({kBatch, kDim}), &backend);
    zeros.fill(0.0f);

    auto sample_from = [&](uint32_t seed) {
        Tensor x(Shape({kBatch, kDim}), &backend, lcg_values(static_cast<size_t>(kBatch * kDim), seed));
        for (int64_t t = kTimesteps; t >= 1; --t) {
            const Tensor predicted = denoiser.predict(x, t);
            x = schedule.denoise_step(x, predicted, zeros, t);
        }
        return x;
    };

    Tensor a = sample_from(31u);
    Tensor b = sample_from(97u);

    float max_difference = 0.0f;
    for (int64_t i = 0; i < a.numel(); ++i) {
        max_difference = std::fmax(max_difference, std::fabs(a.data()[i] - b.data()[i]));
    }
    EXPECT_GT(max_difference, 1e-3f);
}

}  // namespace
}  // namespace exai
