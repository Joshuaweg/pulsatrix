/** @file neat_speciation.hpp
 *  @brief NEAT speciation (Stanley & Miikkulainen 2002): a compatibility-distance metric over
 *         two genomes' gene lists, population grouping by that metric, and fitness sharing --
 *         the mechanism that protects a structurally novel but not-yet-optimized genome from
 *         being immediately out-competed before its innovation has a chance to be refined.
 *  @ingroup evolutionary
 */
#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "pulsatrix/neat_genome.hpp"

namespace pulsatrix {

/**
 * @brief Compatibility distance: delta = c1*E/N + c2*D/N + c3*W_bar, where E is the count of
 *        excess genes (innovation numbers beyond the other genome's own highest), D is the
 *        count of disjoint genes (innovation numbers within the overlapping range but present
 *        in only one genome), N is the larger genome's gene count (or 1 if both genomes have
 *        fewer than 20 connection genes -- the original paper's own small-genome exception),
 *        and W_bar is the average weight difference over genes with matching innovation
 *        numbers (present in both genomes, regardless of enabled/disabled status).
 * @throws std::invalid_argument if either genome has zero connection genes.
 */
inline double CompatibilityDistance(const NEATGenome& a, const NEATGenome& b, double c1, double c2, double c3) {
    const auto& conn_a = a.connections();
    const auto& conn_b = b.connections();
    if (conn_a.empty() || conn_b.empty()) {
        throw std::invalid_argument("CompatibilityDistance: both genomes must have at least one connection gene");
    }

    int max_innovation_a = 0;
    for (const auto& c : conn_a) {
        max_innovation_a = std::max(max_innovation_a, c.innovation);
    }
    int max_innovation_b = 0;
    for (const auto& c : conn_b) {
        max_innovation_b = std::max(max_innovation_b, c.innovation);
    }
    int overlap_bound = std::min(max_innovation_a, max_innovation_b);

    int excess = 0;
    int disjoint = 0;
    int matching = 0;
    double weight_diff_sum = 0.0;

    for (const auto& ca : conn_a) {
        const auto it = std::find_if(conn_b.begin(), conn_b.end(),
                                      [&](const ConnectionGene& cb) { return cb.innovation == ca.innovation; });
        if (it != conn_b.end()) {
            ++matching;
            weight_diff_sum += std::fabs(ca.weight - it->weight);
        } else if (ca.innovation > overlap_bound) {
            ++excess;
        } else {
            ++disjoint;
        }
    }
    for (const auto& cb : conn_b) {
        const bool in_a = std::any_of(conn_a.begin(), conn_a.end(),
                                       [&](const ConnectionGene& ca) { return ca.innovation == cb.innovation; });
        if (!in_a) {
            if (cb.innovation > overlap_bound) {
                ++excess;
            } else {
                ++disjoint;
            }
        }
    }

    size_t size_a = conn_a.size();
    size_t size_b = conn_b.size();
    size_t larger = std::max(size_a, size_b);
    double n = (larger < 20) ? 1.0 : static_cast<double>(larger);
    double w_bar = (matching > 0) ? weight_diff_sum / static_cast<double>(matching) : 0.0;

    return c1 * static_cast<double>(excess) / n + c2 * static_cast<double>(disjoint) / n + c3 * w_bar;
}

/** @brief Population grouping into species: each inner vector is a list of indices into the
 *         population vector that were passed to SpeciatePopulation. */
struct SpeciesAssignment {
    std::vector<std::vector<size_t>> species;
};

/**
 * @brief Groups population into species: each genome joins the first existing species whose
 *        representative (that species' own first member) it is compatible with (distance <
 *        compatibility_threshold); otherwise it founds a new species with itself as
 *        representative.
 * @throws std::invalid_argument if population is empty or compatibility_threshold <= 0.
 */
inline SpeciesAssignment SpeciatePopulation(const std::vector<NEATGenome>& population,
                                            double compatibility_threshold, double c1, double c2, double c3) {
    if (population.empty()) {
        throw std::invalid_argument("SpeciatePopulation: population must not be empty");
    }
    if (compatibility_threshold <= 0.0) {
        throw std::invalid_argument("SpeciatePopulation: compatibility_threshold must be positive");
    }

    std::vector<std::vector<size_t>> species;
    std::vector<size_t> representatives;

    for (size_t i = 0; i < population.size(); ++i) {
        bool placed = false;
        for (size_t s = 0; s < species.size(); ++s) {
            double distance = CompatibilityDistance(population[i], population[representatives[s]], c1, c2, c3);
            if (distance < compatibility_threshold) {
                species[s].push_back(i);
                placed = true;
                break;
            }
        }
        if (!placed) {
            species.push_back({i});
            representatives.push_back(i);
        }
    }
    return SpeciesAssignment{species};
}

/**
 * @brief Fitness sharing: each individual's adjusted fitness is its own raw fitness divided
 *        by the size of its species -- protects small, structurally novel species from being
 *        immediately out-competed by a large, already-optimized one.
 * @throws std::invalid_argument if raw_fitness's size doesn't match the total number of
 *         individuals named across every species (i.e. every population index must appear in
 *         exactly one species).
 */
inline std::vector<double> ComputeAdjustedFitness(const std::vector<double>& raw_fitness,
                                                   const std::vector<std::vector<size_t>>& species) {
    size_t total_members = 0;
    for (const auto& s : species) {
        total_members += s.size();
    }
    if (total_members != raw_fitness.size()) {
        throw std::invalid_argument(
            "ComputeAdjustedFitness: species must partition exactly raw_fitness.size() individuals");
    }

    std::vector<double> adjusted(raw_fitness.size(), 0.0);
    for (const auto& s : species) {
        for (size_t index : s) {
            adjusted[index] = raw_fitness[index] / static_cast<double>(s.size());
        }
    }
    return adjusted;
}

}  // namespace pulsatrix
