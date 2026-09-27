/** @file individual.hpp
 *  @brief A genetic-algorithm candidate solution: a genotype paired with its fitness.
 *  @ingroup evolutionary
 */
#pragma once

namespace pulsatrix {

/**
 * @brief A single candidate solution in a genetic algorithm population.
 * @tparam Genotype The candidate solution's own representation (e.g. std::vector<double>).
 * @tparam FitnessT The fitness value's type (default: double, single-objective, maximized).
 * @note Compile-time template design (Decision Point 4,
 *       campaign_exai_dl_library_evolutionary_deep_learning, resolved 2026-09-27): DEAP's
 *       `creator.create` dynamically synthesizes an Individual class at runtime to work
 *       around Python's lack of compile-time generics -- a plain templated aggregate
 *       expresses the same concept at compile time, with zero runtime class-synthesis cost.
 *       GA operators (selection.hpp) sit in this campaign's hottest inner loop
 *       (per-individual, per-generation), so static polymorphism is the deliberate default
 *       here, unlike DeviceBackend's runtime dispatch (which is unavoidable there -- a
 *       Tensor doesn't know its device until runtime; a GA operator's identity is fixed at
 *       algorithm-design time).
 */
template <typename Genotype, typename FitnessT = double>
struct Individual {
    Genotype genes;
    FitnessT fitness{};
};

}  // namespace pulsatrix
