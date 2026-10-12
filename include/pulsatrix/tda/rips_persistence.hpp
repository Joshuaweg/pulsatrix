/** @file rips_persistence.hpp
 *  @brief Persistent homology of the Vietoris-Rips filtration in every dimension up to a maximum:
 *         components (H0), loops (H1), voids (H2) and beyond, with each feature's critical edges
 *         and a representative cocycle.
 *  @ingroup tda
 *
 * The Vietoris-Rips complex at scale t holds every set of points whose pairwise distances are all
 * at most t; a simplex appears at its diameter, the largest distance among its vertices. As t
 * grows, loops and voids form and fill in. Each one is a pair (birth, death) of a persistence
 * diagram.
 *
 * The computation follows Bauer's Ripser (J. Appl. Comput. Topol. 2021): persistent cohomology,
 * which pairs the same intervals as homology; simplices indexed by the combinatorial number system
 * and their coboundaries enumerated on demand, never stored; columns reduced from the largest
 * scale down, with clearing (a simplex that kills a class one dimension down can't start one); and
 * columns already reduced when first seen, the common case, settled without a reduction. It is
 * written for pulsatrix rather than vendored, over any prime field Z/p.
 *
 * Cost grows quickly with the dimension. H1 is practical up to a few thousand points and H2 up
 * to about a thousand; beyond that, choose landmarks first. Memory holds every simplex of the
 * highest dimension asked for below the threshold that a lower one doesn't already account for.
 */
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "pulsatrix/tda/distance_matrix.hpp"
#include "pulsatrix/tda/persistence_diagram.hpp"

namespace pulsatrix {

/** @brief ComputeRipsPersistence()'s settings. */
struct RipsOptions {
    /** @brief The highest homology dimension to compute: 1 for loops, 2 for voids. At most 8. */
    int max_dimension = 1;
    /**
     * @brief The largest scale; simplices with a larger diameter are left out. A negative value
     *        (the default) uses the enclosing radius, min over points of the largest distance from
     *        that point. At that scale the complex is a cone, which has no loops or voids, so no
     *        feature in dimension 1 or above is cut short. Below it, a feature still alive at the
     *        threshold is reported as essential (death infinity).
     */
    double threshold = -1;
    /** @brief The prime p of the coefficient field Z/p. 2 is the usual choice; a larger prime is
     *         what circular coordinates need, to lift a cocycle to the integers. */
    int coefficient = 2;
    /** @brief Return each feature's representative cocycle. They can be large: one entry per
     *         simplex its reduction touched. */
    bool cocycles = false;
};

/** @brief One simplex of a cocycle with its coefficient. */
struct CocycleEntry {
    /** @brief The simplex's vertices, in increasing order. Its orientation is that order. */
    std::vector<int64_t> vertices;
    /** @brief Its value in Z/p, in [0, p). */
    int coefficient = 0;
};

/** @brief One feature of the diagram, with where it comes from. */
struct RipsFeature {
    /** @brief Its birth and death. */
    PersistencePair pair;
    /** @brief The edge whose length is the birth: the longest edge of the simplex that creates the
     *         feature, endpoints in increasing order. {-1, -1} in H0, where a point creates it. */
    std::array<int64_t, 2> birth_edge{{-1, -1}};
    /** @brief The edge whose length is the death: the longest edge of the simplex that kills it.
     *         {-1, -1} for an essential feature. In H0, the edge that merges two components. */
    std::array<int64_t, 2> death_edge{{-1, -1}};
    /**
     * @brief With RipsOptions::cocycles, for dimension 1 and above: a cocycle that represents the
     *        feature. Evaluated on the simplices of the complex at any scale in [birth, death), its
     *        coboundary is zero; it may also list simplices that appear later. Empty otherwise.
     */
    std::vector<CocycleEntry> cocycle;
};

/** @brief ComputeRipsPersistence()'s result. */
struct RipsResult {
    /** @brief One diagram per dimension, 0 to max_dimension; each carries the distances'
     *         error_bound. */
    std::vector<PersistenceDiagram> diagrams;
    /** @brief features[d][k] describes diagrams[d].pairs[k]. */
    std::vector<std::vector<RipsFeature>> features;
    /** @brief The threshold used. */
    double threshold = 0;
};

/**
 * @brief Persistent homology of the Vietoris-Rips filtration on the given distances, in dimensions
 *        0 to options.max_dimension.
 * @note Diagrams list finite pairs by death, then birth, then essential ones, as
 *       PersistenceDiagram does; pairs with zero persistence are left out. The diagrams don't
 *       depend on how equal distances are ordered; critical edges and cocycles do, and are the
 *       same on every platform.
 * @throws std::invalid_argument if distances has fewer than 2 points, max_dimension is outside
 *         [0, 8], coefficient isn't a prime below 2^15, or the simplices of dimension
 *         max_dimension + 1 can't be indexed in 63 bits.
 */
[[nodiscard]] RipsResult ComputeRipsPersistence(const DistanceMatrix& distances, const RipsOptions& options = {});

}  // namespace pulsatrix
