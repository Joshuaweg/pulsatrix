/** @file generator_population_gan_integration_test.cpp
 *  @brief End-to-end proof that GeneratorPopulation + GeneratePooledFakeSamples +
 *         BackwardThroughPopulation compose into a working multi-generator GAN: one shared
 *         discriminator trained against fakes pooled from several independent generators, with
 *         each generator's own gradient routed back correctly, and each generator
 *         independently, individually trainable against that shared discriminator (the same
 *         non-saturating per-generator step gan_integration_test.cpp's own ToyGAN uses).
 *
 *  Reuses this codebase's own established GAN training-loop pattern (BCEWithLogitsLoss +
 *  ordinary Linear/Relu SequentialModule stacks) rather than inventing a new one -- scaling it
 *  from one generator to a population is exactly this mission's own scope.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "pulsatrix/bce_with_logits_loss.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/generator_population.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/sgd_optimizer.hpp"

namespace pulsatrix {
namespace {

constexpr int64_t kNoise = 3;
constexpr int64_t kDataDim = 2;
constexpr int64_t kHidden = 6;
constexpr int64_t kPerGeneratorBatch = 4;

// Same deterministic pseudo-random initialization gan_integration_test.cpp's own ToyGAN uses.
std::vector<float> lcg_values(size_t count, uint32_t seed) {
    std::vector<float> values;
    values.reserve(count);
    uint32_t state = seed;
    for (size_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        const float unit = static_cast<float>((state >> 8) & 0xFFFFu) / 65535.0f;
        values.push_back((unit - 0.5f) * 1.2f);
    }
    return values;
}

// SequentialModule's exact ownership-taking constructor shape differs from a simple
// pointer-list (see gan_integration_test.cpp, which owns each layer as a named member and
// passes raw pointers). A population's generators must each independently own their own
// layers with no shared members, so this file owns each generator's layers via a small
// wrapper class instead of gan_integration_test.cpp's own named-member style.

class OwnedGenerator : public Module {
public:
    OwnedGenerator(DeviceBackend* backend, uint32_t seed)
        : in_(kNoise, kHidden, backend), relu_(backend), out_(kHidden, kDataDim, backend), seq_({&in_, &relu_, &out_}) {
        in_.set_weight(lcg_values(static_cast<size_t>(kNoise * kHidden), seed));
        out_.set_weight(lcg_values(static_cast<size_t>(kHidden * kDataDim), seed + 1));
    }

    [[nodiscard]] Tensor forward_impl(const Tensor& input) override { return seq_.forward(input); }
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override { return seq_.backward(grad_output); }
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<ParamRef> parameters() override { return seq_.parameters(); }
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override {
        return seq_.propagate_relevance(relevance_out, config);
    }

private:
    LinearModule in_;
    ReluModule relu_;
    LinearModule out_;
    SequentialModule seq_;
};

class OwnedDiscriminator : public Module {
public:
    explicit OwnedDiscriminator(DeviceBackend* backend, uint32_t seed)
        : in_(kDataDim, kHidden, backend), relu_(backend), out_(kHidden, 1, backend), seq_({&in_, &relu_, &out_}) {
        in_.set_weight(lcg_values(static_cast<size_t>(kDataDim * kHidden), seed));
        out_.set_weight(lcg_values(static_cast<size_t>(kHidden * 1), seed + 1));
    }

    [[nodiscard]] Tensor forward_impl(const Tensor& input) override { return seq_.forward(input); }
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override { return seq_.backward(grad_output); }
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<ParamRef> parameters() override { return seq_.parameters(); }
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override {
        return seq_.propagate_relevance(relevance_out, config);
    }

private:
    LinearModule in_;
    ReluModule relu_;
    LinearModule out_;
    SequentialModule seq_;
};

class GeneratorPopulationGANIntegrationTest : public ::testing::Test {
protected:
    CPUBackend backend;

    static constexpr size_t kPopulationSize = 3;

    Tensor real_data(int64_t batch) {
        std::vector<float> values;
        values.reserve(static_cast<size_t>(batch * kDataDim));
        for (int64_t b = 0; b < batch; ++b) {
            values.push_back(1.0f + 0.05f * static_cast<float>(b % 3 - 1));
            values.push_back(1.0f - 0.05f * static_cast<float>(b % 2));
        }
        return Tensor(Shape({batch, kDataDim}), &backend, values);
    }

    Tensor noise_for(uint32_t seed) {
        return Tensor(Shape({kPerGeneratorBatch, kNoise}), &backend,
                       lcg_values(static_cast<size_t>(kPerGeneratorBatch * kNoise), seed));
    }

    Tensor labels(int64_t batch, float value) {
        Tensor t(Shape({batch, 1}), &backend);
        t.fill(value);
        return t;
    }

    float distance_to_real_cluster(Module& generator, const Tensor& z) {
        Tensor fake = generator.forward(z);
        float total = 0.0f;
        for (int64_t b = 0; b < kPerGeneratorBatch; ++b) {
            for (int64_t d = 0; d < kDataDim; ++d) {
                const float diff = fake.data()[b * kDataDim + d] - 1.0f;
                total += diff * diff;
            }
        }
        return total / static_cast<float>(kPerGeneratorBatch * kDataDim);
    }
};

// The composition proof: a population of 3 independently-seeded generators, one shared
// discriminator trained against their *pooled* fake output, and each generator individually
// trained (non-saturating loss) against that same discriminator. Every population member's
// own samples must move toward the real cluster -- proving the pooling/gradient-routing
// harness composes correctly at population scale, not just for one generator.
TEST_F(GeneratorPopulationGANIntegrationTest, EveryPopulationMemberMovesTowardRealClusterAfterTraining) {
    std::vector<std::unique_ptr<Module>> generators;
    for (size_t i = 0; i < kPopulationSize; ++i) {
        generators.push_back(std::make_unique<OwnedGenerator>(&backend, static_cast<uint32_t>(101 + i * 10)));
    }
    GeneratorPopulation population(std::move(generators));
    OwnedDiscriminator discriminator(&backend, /*seed=*/303u);
    BCEWithLogitsLoss bce(&backend);
    SGDOptimizer d_optimizer(0.05f);
    std::vector<SGDOptimizer> g_optimizers(kPopulationSize, SGDOptimizer(0.05f));

    std::vector<Tensor> noise_per_generator;
    std::vector<int64_t> batch_sizes;
    for (size_t i = 0; i < kPopulationSize; ++i) {
        noise_per_generator.push_back(noise_for(static_cast<uint32_t>(1000 + i * 7)));
        batch_sizes.push_back(kPerGeneratorBatch);
    }

    std::vector<float> initial_distances;
    for (size_t i = 0; i < kPopulationSize; ++i) {
        initial_distances.push_back(distance_to_real_cluster(population.generator(i), noise_per_generator[i]));
    }

    constexpr int kSteps = 2000;
    const int64_t total_batch = kPerGeneratorBatch * static_cast<int64_t>(kPopulationSize);
    for (int step = 0; step < kSteps; ++step) {
        // Discriminator step: real term, plus the fake term over the whole pooled population.
        Tensor x_real = real_data(total_batch);
        Tensor logits_real = discriminator.forward(x_real);
        (void)bce.forward(logits_real, labels(total_batch, 1.0f));
        (void)discriminator.backward(bce.backward());

        Tensor pooled_fake = GeneratePooledFakeSamples(population, noise_per_generator, &backend);
        Tensor logits_fake = discriminator.forward(pooled_fake);
        (void)bce.forward(logits_fake, labels(total_batch, 0.0f));
        Tensor grad_pooled_fake = discriminator.backward(bce.backward());
        BackwardThroughPopulation(population, grad_pooled_fake, batch_sizes, &backend);

        d_optimizer.step(discriminator);
        d_optimizer.zero_grad(discriminator);
        for (size_t i = 0; i < kPopulationSize; ++i) {
            g_optimizers[i].zero_grad(population.generator(i));
        }

        // Per-generator step: each population member trains individually against D
        // (Goodfellow's non-saturating form), exactly gan_integration_test.cpp's own
        // ToyGAN::generator_step pattern, just looped once per population member.
        for (size_t i = 0; i < kPopulationSize; ++i) {
            Tensor fake = population.generator(i).forward(noise_per_generator[i]);
            Tensor logits = discriminator.forward(fake);
            (void)bce.forward(logits, labels(kPerGeneratorBatch, 1.0f));
            Tensor grad_fake = discriminator.backward(bce.backward());
            (void)population.generator(i).backward(grad_fake);

            g_optimizers[i].step(population.generator(i));
            g_optimizers[i].zero_grad(population.generator(i));
            d_optimizer.zero_grad(discriminator);
        }
    }

    for (size_t i = 0; i < kPopulationSize; ++i) {
        float final_distance = distance_to_real_cluster(population.generator(i), noise_per_generator[i]);
        EXPECT_LT(final_distance, initial_distances[i]) << "population member " << i;
        EXPECT_LT(final_distance, 0.5f * initial_distances[i]) << "population member " << i;
    }
}

}  // namespace
}  // namespace pulsatrix
