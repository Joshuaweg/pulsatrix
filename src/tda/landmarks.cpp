#include "pulsatrix/tda/landmarks.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include "float_error.hpp"
#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {
namespace {

constexpr const char* kFunction = "FarthestPointLandmarks";

[[noreturn]] void Fail(const std::string& message) {
    throw std::invalid_argument(std::string(kFunction) + ": " + message);
}

}  // namespace

Landmarks FarthestPointLandmarks(const Tensor& points, const LandmarkOptions& options) {
    if (options.count < 0 || options.radius < 0 || std::isnan(options.radius)) {
        Fail("count and radius must not be negative");
    }
    if (options.count == 0 && options.radius == 0) Fail("set count, radius or both, to say when to stop");
    const tda_detail::PreparedPoints p = tda_detail::PreparePoints(points, options.center, false, 1, kFunction);
    if (options.first < 0 || options.first >= p.n) Fail("first must be a row of points");
    const std::vector<double> norm2 = tda_detail::SquaredNorms(p);
    const tda_detail::GemmSquaredError squared_error(p.d, kFunction);
    const std::size_t n = static_cast<std::size_t>(p.n), d = static_cast<std::size_t>(p.d);

    DeviceBackend* backend = points.backend();
    const Tensor rows(Shape({p.n, p.d}), backend, p.rows);
    Tensor dots(Shape({p.n, 1}), backend);

    Landmarks out;
    out.nearest.assign(n, -1);
    out.distance_to_nearest.assign(n, std::numeric_limits<double>::infinity());
    std::vector<double> error_to_nearest(n, 0.0);  // the float32 error of each distance_to_nearest
    int64_t current = options.first;
    while (true) {
        const int64_t k = static_cast<int64_t>(out.indices.size());
        const std::size_t c = static_cast<std::size_t>(current);
        out.insertion_radii.push_back(out.distance_to_nearest[c]);
        out.indices.push_back(current);

        // Distances from every point to the new landmark: one GEMM, (n x d) times (d x 1).
        backend->gemm_ex(rows.data(), false, rows.data() + c * d, true, dots.data(), n, d, 1, 0.0f);
        const std::vector<float> dot = dots.to_host_vector();
        for (std::size_t i = 0; i < n; ++i) {
            double distance = 0, error = 0;
            if (i != c) {
                const double squared = norm2[i] + norm2[c] - 2 * static_cast<double>(dot[i]);
                distance = std::sqrt(std::max(0.0, squared));
                error = tda_detail::DistanceError(squared, squared_error(norm2[i], norm2[c], dot[i]), distance) +
                        p.moved[i] + p.moved[c];
            }
            if (distance < out.distance_to_nearest[i]) {  // a tie keeps the earlier landmark
                out.distance_to_nearest[i] = distance;
                out.nearest[i] = k;
                error_to_nearest[i] = error;
            }
        }

        // The next landmark: the point farthest from all so far (the lowest index on a tie).
        std::size_t next = 0;
        for (std::size_t i = 1; i < n; ++i) {
            if (out.distance_to_nearest[i] > out.distance_to_nearest[next]) next = i;
        }
        const double radius = out.distance_to_nearest[next];
        if (radius == 0 || (options.count > 0 && k + 1 >= options.count) ||
            (options.radius > 0 && radius <= options.radius)) {
            break;
        }
        current = static_cast<int64_t>(next);
    }

    out.covering_radius.value = *std::max_element(out.distance_to_nearest.begin(), out.distance_to_nearest.end());
    // Each point's exact distance to its landmark is within error_to_nearest of the computed one,
    // so the exact covering radius, at most the largest exact such distance, is within the largest
    // error of the value.
    out.covering_radius.terms = {{"float32 distances to the landmarks",
                                  *std::max_element(error_to_nearest.begin(), error_to_nearest.end())}};
    return out;
}

Landmarks FarthestPointLandmarks(const PointCloud& cloud, DeviceBackend* backend, const LandmarkOptions& options) {
    return FarthestPointLandmarks(cloud.ToTensor(backend), options);
}

Tensor LandmarkPoints(const Tensor& points, const Landmarks& landmarks) {
    if (points.rank() < 2) throw std::invalid_argument("LandmarkPoints: points must have shape (n, ...)");
    const int64_t n = points.shape().dim(0);
    const int64_t width = points.numel() / n;
    const std::vector<float> host = points.to_host_vector();
    std::vector<float> rows;
    rows.reserve(landmarks.indices.size() * static_cast<std::size_t>(width));
    for (int64_t i : landmarks.indices) {
        if (i < 0 || i >= n) throw std::invalid_argument("LandmarkPoints: a landmark index is out of range");
        const auto begin = host.begin() + static_cast<std::ptrdiff_t>(i * width);
        rows.insert(rows.end(), begin, begin + static_cast<std::ptrdiff_t>(width));
    }
    std::vector<int64_t> dims{static_cast<int64_t>(landmarks.indices.size())};
    for (int64_t a = 1; a < points.rank(); ++a) dims.push_back(points.shape().dim(static_cast<std::size_t>(a)));
    return Tensor(Shape(dims), points.backend(), rows);
}

PersistenceDiagram WithSubsamplingBound(PersistenceDiagram diagram, const Landmarks& landmarks) {
    diagram.error_bound += landmarks.SubsamplingTerm().bound;
    return diagram;
}

}  // namespace pulsatrix
