/** @file persistence_diagram.hpp
 *  @brief A persistence diagram: the features of one homology dimension, each born at one scale
 *         and dying at a larger one.
 *  @ingroup tda
 */
#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace pulsatrix {

/** @brief One feature of a persistence diagram. */
struct PersistencePair {
    /** @brief The scale at which the feature appears. */
    double birth = 0;
    /** @brief The scale at which it disappears; infinity for a feature that never does, such as
     *         the one component left at the end. */
    double death = std::numeric_limits<double>::infinity();

    /** @brief death - birth: how long it lasts; infinity for an essential feature. */
    [[nodiscard]] double persistence() const { return death - birth; }
    /** @brief Whether it never dies. */
    [[nodiscard]] bool essential() const { return death == std::numeric_limits<double>::infinity(); }
};

/**
 * @brief The features of one homology dimension: H0 for components, H1 for loops, H2 for voids.
 * @note Pairs with zero persistence (birth == death) are left out, as Ripser leaves them out.
 */
struct PersistenceDiagram {
    /** @brief The homology dimension. */
    int dimension = 0;
    /** @brief The features, finite ones by increasing death, then by birth; essential ones last. */
    std::vector<PersistencePair> pairs;
    /**
     * @brief A bound on how far this diagram can be, in bottleneck distance, from the one the
     *        exact distances would give. Stability (Cohen-Steiner, Edelsbrunner and Harer, 2007)
     *        makes the distances' DistanceMatrix::error_bound such a bound.
     */
    double error_bound = 0;

    /** @brief The number of essential features: the Betti number at the largest scale. */
    [[nodiscard]] int64_t essential_count() const {
        int64_t count = 0;
        for (const PersistencePair& p : pairs) count += p.essential() ? 1 : 0;
        return count;
    }
};

}  // namespace pulsatrix
