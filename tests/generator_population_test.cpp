#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/generator_population.hpp"
#include "pulsatrix/linear_module.hpp"

namespace pulsatrix {
namespace {

TEST(SliceBatchTest, ExtractsExactRowRangeAsHandDerived) {
    CPUBackend backend;
    Tensor t(Shape({5, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});

    Tensor first_two = SliceBatch(t, /*start=*/0, /*count=*/2, &backend);
    Tensor last_three = SliceBatch(t, /*start=*/2, /*count=*/3, &backend);

    EXPECT_EQ(first_two.shape(), Shape({2, 2}));
    EXPECT_FLOAT_EQ(first_two.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(first_two.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(first_two.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(first_two.data()[3], 4.0f);

    EXPECT_EQ(last_three.shape(), Shape({3, 2}));
    EXPECT_FLOAT_EQ(last_three.data()[0], 5.0f);
    EXPECT_FLOAT_EQ(last_three.data()[1], 6.0f);
    EXPECT_FLOAT_EQ(last_three.data()[2], 7.0f);
    EXPECT_FLOAT_EQ(last_three.data()[3], 8.0f);
    EXPECT_FLOAT_EQ(last_three.data()[4], 9.0f);
    EXPECT_FLOAT_EQ(last_three.data()[5], 10.0f);
}

TEST(SliceBatchTest, ThrowsWhenRangeExceedsLeadingDimension) {
    CPUBackend backend;
    Tensor t(Shape({5, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
    EXPECT_THROW((void)SliceBatch(t, 4, 2, &backend), std::invalid_argument);
}

TEST(SliceBatchTest, ThrowsOnNonPositiveCount) {
    CPUBackend backend;
    Tensor t(Shape({5, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
    EXPECT_THROW((void)SliceBatch(t, 0, 0, &backend), std::invalid_argument);
}

TEST(GeneratorPopulationTest, ThrowsOnEmptyGenerators) {
    EXPECT_THROW(GeneratorPopulation(std::vector<std::unique_ptr<Module>>{}), std::invalid_argument);
}

TEST(GeneratorPopulationTest, ThrowsOnNullGenerator) {
    std::vector<std::unique_ptr<Module>> generators;
    generators.push_back(nullptr);
    EXPECT_THROW(GeneratorPopulation(std::move(generators)), std::invalid_argument);
}

TEST(GeneratorPopulationTest, ReplaceGeneratorThrowsOnNull) {
    CPUBackend backend;
    std::vector<std::unique_ptr<Module>> generators;
    generators.push_back(std::make_unique<LinearModule>(1, 1, &backend));
    GeneratorPopulation population(std::move(generators));
    EXPECT_THROW(population.ReplaceGenerator(0, nullptr), std::invalid_argument);
}

TEST(GeneratePooledFakeSamplesTest, PoolsEachGeneratorsForwardOutputInOrder) {
    CPUBackend backend;
    auto gen0 = std::make_unique<LinearModule>(1, 1, &backend);
    gen0->set_weight({2.0f});
    gen0->set_bias({1.0f});
    auto gen1 = std::make_unique<LinearModule>(1, 1, &backend);
    gen1->set_weight({3.0f});
    gen1->set_bias({0.0f});

    std::vector<std::unique_ptr<Module>> generators;
    generators.push_back(std::move(gen0));
    generators.push_back(std::move(gen1));
    GeneratorPopulation population(std::move(generators));

    std::vector<Tensor> noise;
    noise.emplace_back(Shape({2, 1}), &backend, std::vector<float>{1.0f, 2.0f});
    noise.emplace_back(Shape({3, 1}), &backend, std::vector<float>{1.0f, 2.0f, 3.0f});

    Tensor pooled = GeneratePooledFakeSamples(population, noise, &backend);

    // gen0(x) = 2x+1 -> [3, 5]; gen1(x) = 3x -> [3, 6, 9]; pooled = [3, 5, 3, 6, 9].
    EXPECT_EQ(pooled.shape(), Shape({5, 1}));
    EXPECT_FLOAT_EQ(pooled.data()[0], 3.0f);
    EXPECT_FLOAT_EQ(pooled.data()[1], 5.0f);
    EXPECT_FLOAT_EQ(pooled.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(pooled.data()[3], 6.0f);
    EXPECT_FLOAT_EQ(pooled.data()[4], 9.0f);
}

TEST(GeneratePooledFakeSamplesTest, ThrowsOnNoiseCountMismatch) {
    CPUBackend backend;
    std::vector<std::unique_ptr<Module>> generators;
    generators.push_back(std::make_unique<LinearModule>(1, 1, &backend));
    GeneratorPopulation population(std::move(generators));
    std::vector<Tensor> noise;
    EXPECT_THROW((void)GeneratePooledFakeSamples(population, noise, &backend), std::invalid_argument);
}

TEST(BackwardThroughPopulationTest, RoutesEachSliceToTheCorrectGeneratorsGradient) {
    CPUBackend backend;
    auto gen0_owned = std::make_unique<LinearModule>(1, 1, &backend);
    gen0_owned->set_weight({5.0f});
    gen0_owned->set_bias({0.0f});
    LinearModule* gen0 = gen0_owned.get();
    (void)gen0->forward(Tensor(Shape({2, 1}), &backend, {2.0f, 3.0f}));

    auto gen1_owned = std::make_unique<LinearModule>(1, 1, &backend);
    gen1_owned->set_weight({7.0f});
    gen1_owned->set_bias({0.0f});
    LinearModule* gen1 = gen1_owned.get();
    (void)gen1->forward(Tensor(Shape({3, 1}), &backend, {1.0f, 1.0f, 1.0f}));

    std::vector<std::unique_ptr<Module>> generators;
    generators.push_back(std::move(gen0_owned));
    generators.push_back(std::move(gen1_owned));
    GeneratorPopulation population(std::move(generators));

    Tensor pooled_grad(Shape({5, 1}), &backend, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f});
    BackwardThroughPopulation(population, pooled_grad, {2, 3}, &backend);

    // gen0's slice is [10, 20] against forward input [2, 3]:
    //   weight_grad = 2*10 + 3*20 = 80, bias_grad = 10+20 = 30.
    EXPECT_FLOAT_EQ(gen0->weight_grad().data()[0], 80.0f);
    EXPECT_FLOAT_EQ(gen0->bias_grad().data()[0], 30.0f);
    // gen1's slice is [30, 40, 50] against forward input [1, 1, 1]:
    //   weight_grad = 1*30 + 1*40 + 1*50 = 120, bias_grad = 30+40+50 = 120.
    EXPECT_FLOAT_EQ(gen1->weight_grad().data()[0], 120.0f);
    EXPECT_FLOAT_EQ(gen1->bias_grad().data()[0], 120.0f);
}

TEST(BackwardThroughPopulationTest, ThrowsOnBatchSizesCountMismatch) {
    CPUBackend backend;
    std::vector<std::unique_ptr<Module>> generators;
    generators.push_back(std::make_unique<LinearModule>(1, 1, &backend));
    GeneratorPopulation population(std::move(generators));
    Tensor pooled_grad(Shape({2, 1}), &backend, {1.0f, 2.0f});
    EXPECT_THROW(BackwardThroughPopulation(population, pooled_grad, {}, &backend), std::invalid_argument);
}

TEST(BackwardThroughPopulationTest, ThrowsWhenBatchSizesDoNotSumToPooledLeadingDimension) {
    CPUBackend backend;
    std::vector<std::unique_ptr<Module>> generators;
    generators.push_back(std::make_unique<LinearModule>(1, 1, &backend));
    (void)generators[0]->forward(Tensor(Shape({2, 1}), &backend, {1.0f, 2.0f}));
    GeneratorPopulation population(std::move(generators));
    Tensor pooled_grad(Shape({2, 1}), &backend, {1.0f, 2.0f});
    EXPECT_THROW(BackwardThroughPopulation(population, pooled_grad, {3}, &backend), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
