/** @file vectorization.hpp
 *  @brief Turning a persistence diagram into numbers for probes, statistics and plots: Betti
 *         curves, persistence landscapes, persistence images and persistent entropy.
 *  @ingroup tda
 *
 * A diagram is a multiset of points, which a classifier or a t-test can't take directly. These
 * map it to fixed-size vectors. They differ in what they keep and in how stable they are:
 *
 * | Vectorization | Stable under | Certified here |
 * |---|---|---|
 * | Persistence landscape | the bottleneck distance, 1-Lipschitz in the sup norm (Bubenik 2015) | yes |
 * | Persistence image | the 1-Wasserstein distance (Adams et al. 2017) | no |
 * | Betti curve | nothing useful: one point crossing a grid value changes it by 1 | no |
 * | Persistent entropy | not in general | no |
 *
 * Essential features (infinite death) are left out of every one of them except the Betti curve,
 * where they count as alive from their birth on. Report how many there were alongside.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/tda/diagram_distance.hpp"
#include "pulsatrix/tda/persistence_diagram.hpp"

namespace pulsatrix {

/** @brief n evenly spaced values from start to stop, both included (n >= 2), or {start} for n == 1. */
[[nodiscard]] std::vector<double> LinearGrid(double start, double stop, int64_t n);

/**
 * @brief The Betti curve: how many features are alive, birth <= t < death, at each scale t.
 * @throws std::invalid_argument if a grid value is NaN.
 */
[[nodiscard]] std::vector<int64_t> BettiCurve(const PersistenceDiagram& diagram, const std::vector<double>& grid);

/** @brief PersistenceLandscape()'s result: layers[k][i] is lambda_{k+1}(grid[i]). */
struct PersistenceLandscape {
    std::vector<double> grid;
    std::vector<std::vector<double>> layers;
};

/**
 * @brief The first `layers` persistence landscape functions (Bubenik, JMLR 2015) on a grid:
 *        lambda_k(t) is the k-th largest of max(0, min(t - b, d - t)) over the finite points.
 * @return Certified: the landscapes of the exact diagram are within bound() of these, in the sup
 *         norm, at every grid point, because landscapes are 1-Lipschitz in the bottleneck distance.
 * @throws std::invalid_argument if layers < 1 or a grid value is NaN.
 */
[[nodiscard]] Certified<PersistenceLandscape> ComputePersistenceLandscape(const PersistenceDiagram& diagram,
                                                                          const std::vector<double>& grid,
                                                                          int64_t layers = 5);

/** @brief ComputePersistenceImage()'s settings. */
struct PersistenceImageOptions {
    /** @brief Pixels along birth (columns) and persistence (rows). */
    int64_t width = 20;
    int64_t height = 20;
    /** @brief The birth range [birth_min, birth_max] and persistence range [0, persistence_max]
     *         the image covers. A max of 0 or less takes the diagram's own largest value (and the
     *         image is then not comparable across diagrams: set them when comparing). */
    double birth_min = 0;
    double birth_max = 0;
    double persistence_max = 0;
    /** @brief The Gaussian's standard deviation; 0 or less takes a tenth of the persistence
     *         range. */
    double sigma = 0;
};

/** @brief A persistence image: values[row * width + column], row 0 at persistence 0. */
struct PersistenceImage {
    int64_t width = 0;
    int64_t height = 0;
    double birth_min = 0;
    double birth_max = 0;
    double persistence_max = 0;
    double sigma = 0;
    std::vector<double> values;
};

/**
 * @brief The persistence image (Adams et al., JMLR 2017): each finite point (b, d) placed at
 *        (b, d - b), weighted by its persistence divided by persistence_max (so features near the
 *        diagonal count for little), and spread by a Gaussian; each pixel holds the integral over
 *        it, so a pixel's value doesn't depend on the resolution's sampling.
 * @throws std::invalid_argument if width or height is below 1, or the ranges are empty.
 */
[[nodiscard]] PersistenceImage ComputePersistenceImage(const PersistenceDiagram& diagram,
                                                       const PersistenceImageOptions& options = {});

/**
 * @brief Persistent entropy (Chintakunta et al. 2015; Rucco et al. 2016): the Shannon entropy, in
 *        nats, of the finite points' persistences divided by their sum. 0 for one point; log n
 *        for n points of equal persistence; 0 for an empty diagram.
 */
[[nodiscard]] double PersistentEntropy(const PersistenceDiagram& diagram);

}  // namespace pulsatrix
