/** @file generator_population.hpp
 *  @brief E-GAN's own population-of-generators infrastructure (Wang et al. 2019, "Evolutionary
 *         Generative Adversarial Networks"): N independently-parameterized generator Modules,
 *         and the concrete mechanism for training one shared discriminator against fake
 *         samples pooled from the *whole* population -- in E-GAN, the discriminator's own
 *         training distribution is the union of every population member's output, not one
 *         generator's alone, which is what gives the discriminator (and, in turn, each
 *         generator's own training signal) a harder, more diverse target than a single-
 *         generator GAN ever sees.
 *  @ingroup evolutionary
 *  @note Mutation objectives (minimax/heuristic/least-squares) and the fitness-based
 *        survivor-selection step that actually make this "evolutionary" are deliberately NOT
 *        this file's job -- this mission's own scope is purely the population-management
 *        infrastructure: hold a population of generators, and prove the shared discriminator
 *        can be trained against their pooled output with gradients routed back correctly to
 *        each individual generator. See Phase 5 Mission 1 for the mutation/fitness machinery.
 *  @note Reuses this codebase's own established GAN training-loop pattern (gan_integration_test.cpp's
 *        ToyGAN: BCEWithLogitsLoss + ordinary Linear/Relu SequentialModule stacks, no new
 *        Module type needed) rather than inventing a new one -- a population of generators is
 *        still just N ordinary Modules, ganged together by this file's own orchestration code.
 */
#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include "pulsatrix/module.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief Owns N independently-parameterized generator Modules. */
class GeneratorPopulation {
public:
    /** @throws std::invalid_argument if generators is empty or any entry is null. */
    explicit GeneratorPopulation(std::vector<std::unique_ptr<Module>> generators)
        : generators_(std::move(generators)) {
        if (generators_.empty()) {
            throw std::invalid_argument("GeneratorPopulation: must have at least one generator");
        }
        for (const auto& g : generators_) {
            if (g == nullptr) {
                throw std::invalid_argument("GeneratorPopulation: generator entries must not be null");
            }
        }
    }

    [[nodiscard]] size_t size() const { return generators_.size(); }

    /** @throws std::out_of_range if index >= size(). */
    [[nodiscard]] Module& generator(size_t index) { return *generators_.at(index); }
    [[nodiscard]] const Module& generator(size_t index) const { return *generators_.at(index); }

    /**
     * @brief Replaces population member index with new_generator -- the concrete mechanism a
     *        future survivor-selection step (Phase 5 Mission 1) will use to install a mutated
     *        offspring in place of its parent.
     * @throws std::invalid_argument if new_generator is null.
     * @throws std::out_of_range if index >= size().
     */
    void ReplaceGenerator(size_t index, std::unique_ptr<Module> new_generator) {
        if (new_generator == nullptr) {
            throw std::invalid_argument("GeneratorPopulation::ReplaceGenerator: new_generator must not be null");
        }
        generators_.at(index) = std::move(new_generator);
    }

private:
    std::vector<std::unique_ptr<Module>> generators_;
};

/**
 * @brief Extracts rows [start, start+count) along t's leading dimension into a new Tensor --
 *        the inverse of Tensor::Stack, needed to split a pooled discriminator gradient back
 *        into each population member's own slice.
 * @throws std::invalid_argument if t has rank 0, count <= 0, start < 0, or start+count exceeds
 *         t's own leading dimension.
 */
inline Tensor SliceBatch(const Tensor& t, int64_t start, int64_t count, DeviceBackend* backend) {
    if (t.shape().rank() < 1) {
        throw std::invalid_argument("SliceBatch: t must have rank >= 1");
    }
    int64_t leading = t.shape().dim(0);
    if (count <= 0 || start < 0 || start + count > leading) {
        throw std::invalid_argument("SliceBatch: [start, start+count) must be a valid sub-range of t's leading dimension");
    }

    std::vector<int64_t> out_dims{count};
    for (int64_t i = 1; i < t.shape().rank(); ++i) {
        out_dims.push_back(t.shape().dim(i));
    }
    Shape out_shape(out_dims);

    int64_t row_size = (leading > 0) ? (t.numel() / leading) : 0;
    std::vector<float> values(static_cast<size_t>(row_size * count));
    for (int64_t i = 0; i < row_size * count; ++i) {
        values[static_cast<size_t>(i)] = t.data()[start * row_size + i];
    }
    return Tensor(out_shape, backend, values, t.device());
}

/**
 * @brief Flattens every parameter tensor module.parameters() reports (in that order) into one
 *        vector -- the concrete mechanism Phase 5 Mission 2's mutation-offspring construction
 *        uses to copy a parent generator's current weights into a freshly-constructed
 *        (architecturally identical) offspring instance before mutating the copy.
 */
inline std::vector<float> FlattenParameters(Module& module) {
    std::vector<float> flat;
    for (const auto& p : module.parameters()) {
        for (int64_t i = 0; i < p.value->numel(); ++i) {
            flat.push_back(p.value->data()[i]);
        }
    }
    return flat;
}

/**
 * @brief Overwrites every parameter tensor module.parameters() reports (in that order) from
 *        flat -- the inverse of FlattenParameters.
 * @throws std::invalid_argument if flat's size doesn't exactly match the total parameter count
 *         module.parameters() reports.
 */
inline void RestoreParameters(Module& module, const std::vector<float>& flat) {
    size_t offset = 0;
    for (const auto& p : module.parameters()) {
        int64_t n = p.value->numel();
        for (int64_t i = 0; i < n; ++i) {
            if (offset >= flat.size()) {
                throw std::invalid_argument("RestoreParameters: flat has fewer entries than module.parameters() needs");
            }
            p.value->data()[i] = flat[offset];
            ++offset;
        }
    }
    if (offset != flat.size()) {
        throw std::invalid_argument("RestoreParameters: flat has more entries than module.parameters() needs");
    }
}

/**
 * @brief Zeros every gradient tensor module.parameters() reports -- a standalone alternative
 *        to calling some optimizer's own zero_grad(module) when no persistent per-module
 *        optimizer instance is being kept around (Phase 5 Mission 2's own mutation-attempt
 *        loop constructs a fresh optimizer per attempt, so there is no single optimizer
 *        instance left to call zero_grad on between generations).
 */
inline void ZeroModuleGradients(Module& module) {
    for (const auto& p : module.parameters()) {
        for (int64_t i = 0; i < p.grad->numel(); ++i) {
            p.grad->data()[i] = 0.0f;
        }
    }
}

/**
 * @brief Runs every population member's generator forward on its own noise batch
 *        (noise_per_generator[i] for population member i), then pools the results
 *        (Tensor::Stack, concatenated along the batch dimension, in population order) into
 *        one combined fake-sample batch -- the discriminator's own training input in E-GAN.
 * @throws std::invalid_argument if noise_per_generator.size() != population.size() (delegates
 *         to each generator's own forward() and to Tensor::Stack for their own preconditions).
 */
inline Tensor GeneratePooledFakeSamples(GeneratorPopulation& population,
                                         const std::vector<Tensor>& noise_per_generator, DeviceBackend* backend) {
    if (noise_per_generator.size() != population.size()) {
        throw std::invalid_argument("GeneratePooledFakeSamples: noise_per_generator.size() must equal population.size()");
    }
    std::vector<Tensor> fakes;
    fakes.reserve(population.size());
    for (size_t i = 0; i < population.size(); ++i) {
        fakes.push_back(population.generator(i).forward(noise_per_generator[i]));
    }
    return Tensor::Stack(fakes, backend);
}

/**
 * @brief Given pooled_grad (the gradient w.r.t. the pooled fake batch GeneratePooledFakeSamples
 *        produced -- e.g. from discriminator.backward() called after a forward on that pooled
 *        batch), routes each population member's own slice back through that member's own
 *        backward() -- the concrete mechanism that lets one shared discriminator train against
 *        every generator's own output while each generator's own parameters still receive
 *        exactly its own correct gradient.
 * @param batch_sizes Each population member's own noise batch's leading-dimension size, in the
 *        same order GeneratePooledFakeSamples pooled them (not assumed uniform across members).
 * @throws std::invalid_argument if batch_sizes.size() != population.size(), or the batch sizes
 *         don't sum to pooled_grad's own leading dimension (delegates to SliceBatch/each
 *         generator's own backward() for their own further preconditions).
 */
inline void BackwardThroughPopulation(GeneratorPopulation& population, const Tensor& pooled_grad,
                                       const std::vector<int64_t>& batch_sizes, DeviceBackend* backend) {
    if (batch_sizes.size() != population.size()) {
        throw std::invalid_argument("BackwardThroughPopulation: batch_sizes.size() must equal population.size()");
    }
    int64_t total = 0;
    for (int64_t b : batch_sizes) {
        total += b;
    }
    if (pooled_grad.shape().rank() < 1 || total != pooled_grad.shape().dim(0)) {
        throw std::invalid_argument("BackwardThroughPopulation: batch_sizes must sum to pooled_grad's leading dimension");
    }

    int64_t offset = 0;
    for (size_t i = 0; i < population.size(); ++i) {
        int64_t count = batch_sizes[i];
        Tensor grad_slice = SliceBatch(pooled_grad, offset, count, backend);
        (void)population.generator(i).backward(grad_slice);
        offset += count;
    }
}

}  // namespace pulsatrix
