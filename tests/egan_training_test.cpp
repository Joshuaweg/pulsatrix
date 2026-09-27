#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/egan_training.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/sgd_optimizer.hpp"

namespace pulsatrix {
namespace {

TEST(RunEGANGenerationTest, RunsOneGenerationAndUpdatesBothPopulationAndDiscriminator) {
    CPUBackend backend;
    std::vector<std::unique_ptr<Module>> generators;
    generators.push_back(std::make_unique<LinearModule>(2, 2, &backend));
    generators.back()->parameters()[0].value->fill(0.1f);
    generators.push_back(std::make_unique<LinearModule>(2, 2, &backend));
    generators.back()->parameters()[0].value->fill(0.2f);
    GeneratorPopulation population(std::move(generators));

    LinearModule discriminator(2, 1, &backend);
    discriminator.set_weight({0.3f, -0.2f});
    discriminator.set_bias({0.0f});
    SGDOptimizer d_optimizer(0.05f);
    auto make_generator = [&backend]() -> std::unique_ptr<Module> { return std::make_unique<LinearModule>(2, 2, &backend); };

    std::vector<float> initial_gen0_weight = FlattenParameters(population.generator(0));
    std::vector<float> initial_gen1_weight = FlattenParameters(population.generator(1));
    Tensor initial_discriminator_weight = discriminator.weight();
    std::vector<float> initial_discriminator_weight_values(initial_discriminator_weight.data(),
                                                             initial_discriminator_weight.data() +
                                                                 initial_discriminator_weight.numel());

    Tensor real_batch(Shape({4, 2}), &backend, {1.0f, 1.0f, 0.9f, 1.1f, 1.05f, 0.95f, 0.95f, 1.05f});
    std::vector<Tensor> noise{Tensor(Shape({3, 2}), &backend, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}),
                               Tensor(Shape({3, 2}), &backend, {0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f})};

    std::vector<float> fitness = RunEGANGeneration(population, discriminator, real_batch, noise,
                                                     {MutationObjective::Heuristic}, make_generator,
                                                     /*g_learning_rate=*/0.1f, d_optimizer, &backend);

    ASSERT_EQ(fitness.size(), 2u);
    EXPECT_TRUE(std::isfinite(fitness[0]));
    EXPECT_TRUE(std::isfinite(fitness[1]));

    EXPECT_NE(FlattenParameters(population.generator(0)), initial_gen0_weight);
    EXPECT_NE(FlattenParameters(population.generator(1)), initial_gen1_weight);

    std::vector<float> final_discriminator_weight_values(discriminator.weight().data(),
                                                           discriminator.weight().data() + discriminator.weight().numel());
    EXPECT_NE(final_discriminator_weight_values, initial_discriminator_weight_values);
}

TEST(RunEGANGenerationTest, ThrowsOnNoiseCountMismatch) {
    CPUBackend backend;
    std::vector<std::unique_ptr<Module>> generators;
    generators.push_back(std::make_unique<LinearModule>(2, 2, &backend));
    GeneratorPopulation population(std::move(generators));
    LinearModule discriminator(2, 1, &backend);
    SGDOptimizer d_optimizer(0.05f);
    Tensor real_batch(Shape({2, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f});
    auto make_generator = [&backend]() -> std::unique_ptr<Module> { return std::make_unique<LinearModule>(2, 2, &backend); };

    EXPECT_THROW((void)RunEGANGeneration(population, discriminator, real_batch, {}, {MutationObjective::Heuristic},
                                          make_generator, 0.1f, d_optimizer, &backend),
                 std::invalid_argument);
}

TEST(RunEGANGenerationTest, ThrowsOnEmptyObjectives) {
    CPUBackend backend;
    std::vector<std::unique_ptr<Module>> generators;
    generators.push_back(std::make_unique<LinearModule>(2, 2, &backend));
    GeneratorPopulation population(std::move(generators));
    LinearModule discriminator(2, 1, &backend);
    SGDOptimizer d_optimizer(0.05f);
    Tensor real_batch(Shape({2, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f});
    std::vector<Tensor> noise{Tensor(Shape({2, 2}), &backend, {0.1f, 0.2f, 0.3f, 0.4f})};
    auto make_generator = [&backend]() -> std::unique_ptr<Module> { return std::make_unique<LinearModule>(2, 2, &backend); };

    EXPECT_THROW((void)RunEGANGeneration(population, discriminator, real_batch, noise, {}, make_generator, 0.1f,
                                          d_optimizer, &backend),
                 std::invalid_argument);
}

TEST(RunEGANTrainingTest, ThrowsOnNonPositiveGenerations) {
    CPUBackend backend;
    std::vector<std::unique_ptr<Module>> generators;
    generators.push_back(std::make_unique<LinearModule>(2, 2, &backend));
    GeneratorPopulation population(std::move(generators));
    LinearModule discriminator(2, 1, &backend);
    SGDOptimizer d_optimizer(0.05f);
    Tensor real_batch(Shape({2, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f});
    std::mt19937 rng(1);
    auto make_generator = [&backend]() -> std::unique_ptr<Module> { return std::make_unique<LinearModule>(2, 2, &backend); };

    EXPECT_THROW(RunEGANTraining(population, discriminator, real_batch, /*num_generations=*/0, /*noise_dim=*/2,
                                  /*per_generator_batch=*/3, {MutationObjective::Heuristic}, make_generator, 0.1f,
                                  d_optimizer, &backend, rng),
                 std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
