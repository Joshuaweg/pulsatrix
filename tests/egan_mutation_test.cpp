#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/egan_mutation.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sgd_optimizer.hpp"

namespace pulsatrix {
namespace {

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double Softplus(double x) { return std::log(1.0 + std::exp(x)); }

TEST(MutationLossTest, HeuristicMatchesHandDerivedBCEValueAndGradient) {
    CPUBackend backend;
    MutationLoss loss(&backend);
    Tensor logits(Shape({1, 1}), &backend, {2.0f});

    float value = loss.Forward(MutationObjective::Heuristic, logits);
    Tensor grad = loss.Backward();

    // BCE(x=2, target=1) = softplus(-x) = log(1+exp(-2)).
    EXPECT_NEAR(value, static_cast<float>(Softplus(-2.0)), 1e-5f);
    // backward = sigmoid(x) - 1.
    EXPECT_NEAR(grad.data()[0], static_cast<float>(Sigmoid(2.0) - 1.0), 1e-5f);
}

TEST(MutationLossTest, LeastSquaresMatchesHandDerivedMSEValueAndGradient) {
    CPUBackend backend;
    MutationLoss loss(&backend);
    Tensor logits(Shape({1, 1}), &backend, {2.0f});

    float value = loss.Forward(MutationObjective::LeastSquares, logits);
    Tensor grad = loss.Backward();

    // MSE((x=2) vs target=1) = (2-1)^2 = 1.0; backward = 2*(x-1)/numel = 2.0.
    EXPECT_NEAR(value, 1.0f, 1e-5f);
    EXPECT_NEAR(grad.data()[0], 2.0f, 1e-5f);
}

TEST(MutationLossTest, MinimaxMatchesHandDerivedNegatedBCEValueAndGradient) {
    CPUBackend backend;
    MutationLoss loss(&backend);
    Tensor logits(Shape({1, 1}), &backend, {2.0f});

    float value = loss.Forward(MutationObjective::Minimax, logits);
    Tensor grad = loss.Backward();

    // Minimax(x=2) = -BCE(x, target=0) = -softplus(x) = -log(1+exp(2)).
    EXPECT_NEAR(value, static_cast<float>(-Softplus(2.0)), 1e-5f);
    // backward = -sigmoid(x).
    EXPECT_NEAR(grad.data()[0], static_cast<float>(-Sigmoid(2.0)), 1e-5f);
}

TEST(QualityFitnessTest, ReturnsMeanSigmoidOfLogits) {
    CPUBackend backend;
    Tensor logits(Shape({2, 1}), &backend, {0.0f, 2.0f});

    float fitness = QualityFitness(logits);

    double expected = (Sigmoid(0.0) + Sigmoid(2.0)) / 2.0;
    EXPECT_NEAR(fitness, static_cast<float>(expected), 1e-5f);
}

TEST(DiversityFitnessTest, MatchesHandDerivedGradientNormForKnownLinearDiscriminator) {
    CPUBackend backend;
    LinearModule discriminator(1, 1, &backend);
    discriminator.set_weight({1.0f});
    discriminator.set_bias({0.0f});
    Tensor fake(Shape({2, 1}), &backend, {1.0f, 2.0f});

    float fitness = DiversityFitness(discriminator, fake, &backend);

    // logits = [1, 2] (weight=1, bias=0). BCE(x,0) backward: grad[i] = sigmoid(x_i)/numel.
    double grad1 = Sigmoid(1.0) / 2.0;
    double grad2 = Sigmoid(2.0) / 2.0;
    // weight_grad = x1*grad1 + x2*grad2; bias_grad = grad1 + grad2.
    double weight_grad = 1.0 * grad1 + 2.0 * grad2;
    double bias_grad = grad1 + grad2;
    double norm = std::sqrt(weight_grad * weight_grad + bias_grad * bias_grad);
    double expected = -std::log(norm);

    EXPECT_NEAR(fitness, static_cast<float>(expected), 1e-4f);
}

TEST(DiversityFitnessTest, ThrowsWhenDiscriminatorHasNoParameters) {
    CPUBackend backend;
    ReluModule discriminator(&backend);
    Tensor fake(Shape({2, 1}), &backend, {1.0f, 2.0f});
    EXPECT_THROW((void)DiversityFitness(discriminator, fake, &backend), std::invalid_argument);
}

TEST(CombinedFitnessTest, SumsQualityAndGammaScaledDiversity) {
    EXPECT_NEAR(CombinedFitness(2.0f, 3.0f, 0.5f), 3.5f, 1e-6f);
}

TEST(RunMutationStepTest, ReturnsFiniteFitnessAndUpdatesOffspringWeights) {
    CPUBackend backend;
    LinearModule offspring(1, 1, &backend);
    offspring.set_weight({0.5f});
    offspring.set_bias({0.1f});
    LinearModule discriminator(1, 1, &backend);
    discriminator.set_weight({1.0f});
    discriminator.set_bias({0.0f});
    Tensor noise(Shape({4, 1}), &backend, {0.1f, 0.2f, 0.3f, 0.4f});
    SGDOptimizer g_optimizer(0.1f);
    SGDOptimizer d_zero_grad_helper(0.0f);

    float initial_weight = offspring.weight().data()[0];
    float fitness = RunMutationStep(offspring, discriminator, noise, MutationObjective::Heuristic, g_optimizer,
                                     &backend);
    d_zero_grad_helper.zero_grad(discriminator);

    EXPECT_TRUE(std::isfinite(fitness));
    EXPECT_NE(offspring.weight().data()[0], initial_weight);
}

// A real, meaningful property of E-GAN's own design: the three named objectives must produce
// genuinely different training signals, not just call the same code path three times under
// different names.
TEST(RunMutationStepTest, DifferentObjectivesProduceDifferentUpdatesFromTheSameStartingPoint) {
    CPUBackend backend;
    Tensor noise(Shape({4, 1}), &backend, {0.1f, 0.2f, 0.3f, 0.4f});

    auto run_with = [&](MutationObjective objective) {
        LinearModule offspring(1, 1, &backend);
        offspring.set_weight({0.5f});
        offspring.set_bias({0.1f});
        LinearModule discriminator(1, 1, &backend);
        discriminator.set_weight({1.0f});
        discriminator.set_bias({0.0f});
        SGDOptimizer g_optimizer(0.1f);
        (void)RunMutationStep(offspring, discriminator, noise, objective, g_optimizer, &backend);
        return offspring.weight().data()[0];
    };

    float heuristic_weight = run_with(MutationObjective::Heuristic);
    float least_squares_weight = run_with(MutationObjective::LeastSquares);
    float minimax_weight = run_with(MutationObjective::Minimax);

    EXPECT_NE(heuristic_weight, least_squares_weight);
    EXPECT_NE(heuristic_weight, minimax_weight);
    EXPECT_NE(least_squares_weight, minimax_weight);
}

}  // namespace
}  // namespace pulsatrix
