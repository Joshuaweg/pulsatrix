/** @file nsga2.hpp
 *  @brief NSGA-II multi-objective survivor selection (Deb, Pratap, Agarwal, Meyarivan,
 *         "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II," IEEE TEVC 6(2),
 *         2002): fast non-dominated sorting, crowding distance, and truncation selection.
 *  @ingroup evolutionary
 *  @note Fortin & Parizeau, "Revisiting the NSGA-II Crowding-Distance Computation," GECCO
 *        2013, is this campaign's own standing citation (research doc
 *        research_2026_evolutionary_deep_learning.md §1, §5) for verifying a from-scratch
 *        crowding-distance implementation against the corrected algorithm rather than a
 *        naive textbook description. This implementation's own specific correctness
 *        precautions (documented on CrowdingDistance below): a stable per-objective sort for
 *        deterministic tie-breaking, and skipping (not dividing by zero on) any objective
 *        that is uniform across an entire front.
 *  @note Multi-objective fitness uses FitnessT = std::vector<double> (one entry per
 *        objective) -- Individual<Genotype, FitnessT> (individual.hpp) already supports this
 *        directly, no new genotype/fitness container needed. Convention: maximization in
 *        every objective, consistent with this campaign's scalar-fitness operators
 *        (selection.hpp).
 */
#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "pulsatrix/individual.hpp"

namespace pulsatrix {

using Objectives = std::vector<double>;

/**
 * @brief Pareto dominance (maximization convention): a dominates b iff a[i] >= b[i] for
 *        every objective i, and a[i] > b[i] for at least one objective.
 * @throws std::invalid_argument if a.size() != b.size().
 */
inline bool Dominates(const Objectives& a, const Objectives& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Dominates: objective vectors must have equal size");
    }
    bool strictly_better_in_one = false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] < b[i]) {
            return false;
        }
        if (a[i] > b[i]) {
            strictly_better_in_one = true;
        }
    }
    return strictly_better_in_one;
}

/**
 * @brief Fast non-dominated sort (Deb et al. 2002, Algorithm: fast-non-dominated-sort):
 *        partitions [0, objectives.size()) into fronts -- front 0 is the non-dominated set,
 *        front 1 is non-dominated after removing front 0, and so on.
 * @throws std::invalid_argument -- propagated from Dominates if objective vectors have
 *         inconsistent sizes.
 */
inline std::vector<std::vector<size_t>> FastNonDominatedSort(
    const std::vector<Objectives>& objectives) {
    const size_t n = objectives.size();
    std::vector<size_t> domination_count(n, 0);
    std::vector<std::vector<size_t>> dominates_set(n);
    std::vector<std::vector<size_t>> fronts;
    std::vector<size_t> current_front;

    for (size_t p = 0; p < n; ++p) {
        for (size_t q = 0; q < n; ++q) {
            if (p == q) {
                continue;
            }
            if (Dominates(objectives[p], objectives[q])) {
                dominates_set[p].push_back(q);
            } else if (Dominates(objectives[q], objectives[p])) {
                domination_count[p]++;
            }
        }
        if (domination_count[p] == 0) {
            current_front.push_back(p);
        }
    }

    while (!current_front.empty()) {
        fronts.push_back(current_front);
        std::vector<size_t> next_front;
        for (size_t p : current_front) {
            for (size_t q : dominates_set[p]) {
                if (--domination_count[q] == 0) {
                    next_front.push_back(q);
                }
            }
        }
        current_front = std::move(next_front);
    }
    return fronts;
}

/**
 * @brief Crowding distance within a single front.
 * @param front_objectives Objective vectors for exactly the individuals in one front (any
 *        order); the returned distances are indexed identically (distances[i] corresponds to
 *        front_objectives[i]), not by any global population index.
 * @throws std::invalid_argument if the objective vectors are not all the same size.
 * @note Fronts of size 1 or 2: every individual is a boundary individual for every objective
 *       by construction -- assigned infinity directly, no neighbor-gap computation.
 * @note Uses a stable sort per objective so that individuals sharing an identical value for
 *       that objective break ties in a fixed, reproducible order rather than an
 *       implementation- or input-order-dependent one.
 * @note If a front is uniform in some objective (max == min across the whole front), that
 *       objective contributes nothing to any individual's distance in this front (skipped
 *       outright) rather than dividing by a zero range.
 */
inline std::vector<double> CrowdingDistance(const std::vector<Objectives>& front_objectives) {
    const size_t n = front_objectives.size();
    std::vector<double> distances(n, 0.0);
    if (n == 0) {
        return distances;
    }
    const size_t num_objectives = front_objectives[0].size();
    for (const auto& obj : front_objectives) {
        if (obj.size() != num_objectives) {
            throw std::invalid_argument(
                "CrowdingDistance: every objective vector in a front must have equal size");
        }
    }
    if (n <= 2) {
        std::fill(distances.begin(), distances.end(), std::numeric_limits<double>::infinity());
        return distances;
    }

    for (size_t m = 0; m < num_objectives; ++m) {
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return front_objectives[a][m] < front_objectives[b][m];
        });

        const double min_val = front_objectives[order.front()][m];
        const double max_val = front_objectives[order.back()][m];
        const double range = max_val - min_val;

        distances[order.front()] = std::numeric_limits<double>::infinity();
        distances[order.back()] = std::numeric_limits<double>::infinity();

        if (range <= 0.0) {
            continue;  // objective m is uniform across this front -- no information to add
        }
        for (size_t i = 1; i + 1 < n; ++i) {
            if (distances[order[i]] == std::numeric_limits<double>::infinity()) {
                continue;  // already a boundary individual under a different objective
            }
            distances[order[i]] +=
                (front_objectives[order[i + 1]][m] - front_objectives[order[i - 1]][m]) / range;
        }
    }
    return distances;
}

/**
 * @brief NSGA-II survivor selection: combines population and offspring, fast-non-dominated-
 *        sorts the pool, includes whole fronts (best first) until the next front would
 *        overflow mu, then fills the remainder from that final front by crowding distance
 *        (largest first -- more diverse/isolated solutions preferred).
 * @throws std::invalid_argument if population.size() + offspring.size() < mu.
 * @note Same (population, offspring, mu) -> next-generation-population signature as
 *       GenerationalReplacement/MuPlusLambdaReplacement/MuCommaLambdaReplacement
 *       (survivor_selection.hpp) -- directly usable as RunEvolutionaryLoop's SurvivorFn.
 */
template <typename Genotype>
std::vector<Individual<Genotype, Objectives>> NSGA2Replacement(
    const std::vector<Individual<Genotype, Objectives>>& population,
    std::vector<Individual<Genotype, Objectives>> offspring, size_t mu) {
    if (population.size() + offspring.size() < mu) {
        throw std::invalid_argument(
            "NSGA2Replacement: population.size() + offspring.size() must be >= mu");
    }

    std::vector<Individual<Genotype, Objectives>> combined = population;
    combined.insert(combined.end(), std::make_move_iterator(offspring.begin()),
                     std::make_move_iterator(offspring.end()));

    std::vector<Objectives> objectives;
    objectives.reserve(combined.size());
    for (const auto& ind : combined) {
        objectives.push_back(ind.fitness);
    }
    auto fronts = FastNonDominatedSort(objectives);

    std::vector<Individual<Genotype, Objectives>> next_generation;
    next_generation.reserve(mu);
    for (const auto& front : fronts) {
        if (next_generation.size() + front.size() <= mu) {
            for (size_t idx : front) {
                next_generation.push_back(combined[idx]);
            }
            if (next_generation.size() == mu) {
                break;
            }
        } else {
            std::vector<Objectives> front_objectives;
            front_objectives.reserve(front.size());
            for (size_t idx : front) {
                front_objectives.push_back(combined[idx].fitness);
            }
            auto distances = CrowdingDistance(front_objectives);

            std::vector<size_t> order(front.size());
            std::iota(order.begin(), order.end(), size_t{0});
            std::stable_sort(order.begin(), order.end(),
                              [&](size_t a, size_t b) { return distances[a] > distances[b]; });

            const size_t remaining = mu - next_generation.size();
            for (size_t i = 0; i < remaining; ++i) {
                next_generation.push_back(combined[front[order[i]]]);
            }
            break;
        }
    }
    return next_generation;
}

}  // namespace pulsatrix
