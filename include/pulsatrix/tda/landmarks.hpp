/** @file landmarks.hpp
 *  @brief Farthest-point landmarks: a subset of a large point cloud that covers it, with a
 *         certified bound on what persistent homology loses by using the subset.
 *  @ingroup tda
 *
 * Persistent homology above H0 can't run on 100,000 activations. Farthest-point (maxmin) sampling
 * picks landmarks greedily, each the point farthest from those already picked, so they spread
 * over the whole cloud. Every point is then within the covering radius e of a landmark; the
 * landmarks are within Hausdorff, hence Gromov-Hausdorff, distance e of the cloud; and Rips
 * stability gives d_B(Dgm(landmarks), Dgm(cloud)) <= 2 e (the research notes' TDA-7 bound). The
 * greedy choice also gives a covering radius within a factor of 2 of the best possible for the
 * same number of landmarks (Gonzalez 1985).
 *
 * Distances to each new landmark come from one GEMM against all points on the backend, so the
 * full n x n matrix is never formed: memory is O(n + k d) beyond the points.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/tda/diagram_distance.hpp"
#include "pulsatrix/tda/persistence_diagram.hpp"
#include "pulsatrix/tda/point_cloud.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief FarthestPointLandmarks()'s settings. Selection stops at whichever limit comes first. */
struct LandmarkOptions {
    /** @brief The most landmarks to pick; 0 for no limit (then radius must be set). */
    int64_t count = 0;
    /** @brief Stop once every point is within this distance of a landmark; 0 for no limit. */
    double radius = 0;
    /** @brief The first landmark. */
    int64_t first = 0;
    /** @brief Subtract the mean point first: Euclidean distances are unchanged, and the float32
     *         error shrinks when the points share a large offset (see DistanceOptions::center). */
    bool center = false;
};

/** @brief FarthestPointLandmarks()'s result. */
struct Landmarks {
    /** @brief The landmarks' row indices, in the order they were picked. */
    std::vector<int64_t> indices;
    /** @brief For each point, the position in indices of its nearest landmark. */
    std::vector<int64_t> nearest;
    /** @brief For each point, its distance to that landmark. */
    std::vector<double> distance_to_nearest;
    /** @brief insertion_radii[k]: the covering radius before landmark k was added, which is its
     *         distance to the landmarks before it (infinity for the first). Non-increasing: plot it
     *         to choose a count. */
    std::vector<double> insertion_radii;
    /** @brief The covering radius, max over points of distance_to_nearest, with the float32 error
     *         of the distances behind it as its term: the exact covering radius is at most
     *         value + bound(). */
    Certified<double> covering_radius;

    /** @brief The error term subsampling adds to a diagram of the landmarks, as a bound on its
     *         bottleneck distance from the whole cloud's: twice the certified covering radius. */
    [[nodiscard]] ErrorTerm SubsamplingTerm() const {
        return {"landmark subsampling: twice the covering radius", 2 * (covering_radius.value + covering_radius.bound())};
    }
};

/**
 * @brief Picks farthest-point landmarks among the rows of points.
 * @param points shape (n, ...): a row per point, dimensions after the first flattened; distances
 *        are computed on its backend.
 * @note Ties go to the lowest index. Each step is one GEMM of the points against the new landmark
 *       and an O(n) update, so k landmarks cost O(n k d).
 * @throws std::invalid_argument if points has rank below 2, no rows, a NaN or infinity; if count
 *         and radius are both 0, either is negative, or first is out of range.
 */
[[nodiscard]] Landmarks FarthestPointLandmarks(const Tensor& points, const LandmarkOptions& options);

/** @brief Landmarks of a cloud's points, computed on the given backend. */
[[nodiscard]] Landmarks FarthestPointLandmarks(const PointCloud& cloud, DeviceBackend* backend,
                                               const LandmarkOptions& options);

/** @brief The landmark rows of points, as a (k, ...) tensor on points' backend. */
[[nodiscard]] Tensor LandmarkPoints(const Tensor& points, const Landmarks& landmarks);

/**
 * @brief A diagram of the landmarks re-certified against the whole cloud: its error_bound becomes
 *        its own plus the subsampling term, a bound on its bottleneck distance from the diagram of
 *        all the points (with exact distances).
 */
[[nodiscard]] PersistenceDiagram WithSubsamplingBound(PersistenceDiagram diagram, const Landmarks& landmarks);

}  // namespace pulsatrix
