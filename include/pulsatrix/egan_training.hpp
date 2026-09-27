/** @file egan_training.hpp
 *  @brief E-GAN's own full generational training loop (Wang et al. 2019): each generation,
 *         every population member independently attempts every given mutation objective
 *         (Phase 5 Mission 1) via a fresh weight-copy offspring, keeps whichever mutation
 *         scored the best combined fitness, then the shared discriminator trains on a real
 *         batch plus the (now-mutated) population's own pooled fake output (Phase 5 Mission 0).
 *  @ingroup evolutionary
 *  @note Passing a single-element objectives list (repeated N times, e.g. {Heuristic,
 *        Heuristic, Heuristic}) turns this same loop into a compute-matched single-objective
 *        GAN baseline -- deliberately not a separate code path, so any measured difference
 *        between the two arms is attributable to the objective-diversity/selection mechanism
 *        itself, not to a structurally different training loop.
 */
#pragma once

#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/bce_with_logits_loss.hpp"
#include "pulsatrix/egan_mutation.hpp"
#include "pulsatrix/generator_population.hpp"
#include "pulsatrix/sgd_optimizer.hpp"

namespace pulsatrix {

/**
 * @brief Runs one E-GAN generation: for every population member, attempts every objective in
 *        objectives (each against a fresh weight-copy offspring built by make_generator +
 *        RestoreParameters), keeps the best-combined-fitness offspring, replaces that
 *        population slot with it, then trains discriminator on real_batch plus the
 *        now-mutated population's own pooled fake output.
 * @param make_generator Constructs a fresh generator instance with the same architecture as
 *        every population member.
 * @param g_learning_rate Learning rate for the SGDOptimizer each mutation attempt trains with.
 * @return Each population member's own winning objective's fitness, in population order.
 * @throws std::invalid_argument if noise_per_generator.size() != population.size(), or
 *         objectives is empty.
 */
template <typename MakeGenerator, typename DOptimizerT>
std::vector<float> RunEGANGeneration(GeneratorPopulation& population, Module& discriminator,
                                      const Tensor& real_batch, const std::vector<Tensor>& noise_per_generator,
                                      const std::vector<MutationObjective>& objectives, MakeGenerator make_generator,
                                      float g_learning_rate, DOptimizerT& d_optimizer, DeviceBackend* backend,
                                      float gamma = 0.05f) {
    if (noise_per_generator.size() != population.size()) {
        throw std::invalid_argument("RunEGANGeneration: noise_per_generator.size() must equal population.size()");
    }
    if (objectives.empty()) {
        throw std::invalid_argument("RunEGANGeneration: objectives must not be empty");
    }

    std::vector<float> winning_fitness(population.size());
    std::vector<int64_t> batch_sizes(population.size());

    for (size_t i = 0; i < population.size(); ++i) {
        std::vector<float> parent_weights = FlattenParameters(population.generator(i));
        std::unique_ptr<Module> best_offspring;
        float best_fitness = -std::numeric_limits<float>::infinity();

        for (MutationObjective objective : objectives) {
            std::unique_ptr<Module> offspring = make_generator();
            RestoreParameters(*offspring, parent_weights);
            SGDOptimizer g_optimizer(g_learning_rate);
            float fitness =
                RunMutationStep(*offspring, discriminator, noise_per_generator[i], objective, g_optimizer, backend, gamma);
            d_optimizer.zero_grad(discriminator);  // RunMutationStep's own documented contract

            if (fitness > best_fitness) {
                best_fitness = fitness;
                best_offspring = std::move(offspring);
            }
        }

        winning_fitness[i] = best_fitness;
        batch_sizes[i] = noise_per_generator[i].shape().dim(0);
        population.ReplaceGenerator(i, std::move(best_offspring));
    }

    // Discriminator step: real term + pooled-fake term from the now-mutated population.
    BCEWithLogitsLoss bce(backend);
    Tensor logits_real = discriminator.forward(real_batch);
    Tensor real_labels(logits_real.shape(), backend);
    real_labels.fill(1.0f);
    (void)bce.forward(logits_real, real_labels);
    (void)discriminator.backward(bce.backward());

    Tensor pooled_fake = GeneratePooledFakeSamples(population, noise_per_generator, backend);
    Tensor logits_fake = discriminator.forward(pooled_fake);
    Tensor fake_labels(logits_fake.shape(), backend);
    fake_labels.fill(0.0f);
    (void)bce.forward(logits_fake, fake_labels);
    Tensor grad_pooled_fake = discriminator.backward(bce.backward());
    BackwardThroughPopulation(population, grad_pooled_fake, batch_sizes, backend);

    d_optimizer.step(discriminator);
    d_optimizer.zero_grad(discriminator);
    // BackwardThroughPopulation just accumulated the discriminator step's own gradient
    // contribution into every population member's parameter-gradient buffer (an unavoidable
    // side effect of routing a gradient through backward(), the same contamination hazard
    // BCEWithLogitsLoss's own @warning documents for the generator step) -- cleared here so
    // next generation's own mutation attempts start from a clean buffer, not a mix of this
    // generation's D-step contamination and their own fresh accumulation.
    for (size_t i = 0; i < population.size(); ++i) {
        ZeroModuleGradients(population.generator(i));
    }

    return winning_fitness;
}

/**
 * @brief RNG-driven wrapper: draws fresh standard-normal noise for every population member
 *        every generation, then runs RunEGANGeneration num_generations times.
 * @throws std::invalid_argument if num_generations is not positive (delegates to
 *         RunEGANGeneration for its own preconditions each generation).
 */
template <typename MakeGenerator, typename DOptimizerT, typename RNG>
void RunEGANTraining(GeneratorPopulation& population, Module& discriminator, const Tensor& real_batch,
                      int num_generations, int64_t noise_dim, int64_t per_generator_batch,
                      const std::vector<MutationObjective>& objectives, MakeGenerator make_generator,
                      float g_learning_rate, DOptimizerT& d_optimizer, DeviceBackend* backend, RNG& rng,
                      float gamma = 0.05f) {
    if (num_generations <= 0) {
        throw std::invalid_argument("RunEGANTraining: num_generations must be positive");
    }

    std::normal_distribution<float> noise_dist(0.0f, 1.0f);
    for (int gen = 0; gen < num_generations; ++gen) {
        std::vector<Tensor> noise_per_generator;
        noise_per_generator.reserve(population.size());
        for (size_t i = 0; i < population.size(); ++i) {
            std::vector<float> values(static_cast<size_t>(per_generator_batch * noise_dim));
            for (auto& v : values) {
                v = noise_dist(rng);
            }
            noise_per_generator.emplace_back(Shape({per_generator_batch, noise_dim}), backend, values);
        }
        (void)RunEGANGeneration(population, discriminator, real_batch, noise_per_generator, objectives,
                                 make_generator, g_learning_rate, d_optimizer, backend, gamma);
    }
}

}  // namespace pulsatrix
