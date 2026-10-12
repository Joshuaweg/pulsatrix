#include "pulsatrix/tda/vectorization.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>

namespace pulsatrix {
namespace {

void CheckGrid(const std::vector<double>& grid, const char* function) {
    for (double t : grid) {
        if (std::isnan(t)) throw std::invalid_argument(std::string(function) + ": the grid has a NaN");
    }
}

}  // namespace

std::vector<double> LinearGrid(double start, double stop, int64_t n) {
    if (n < 1) throw std::invalid_argument("LinearGrid: n must be at least 1");
    if (!std::isfinite(start) || !std::isfinite(stop)) throw std::invalid_argument("LinearGrid: start and stop must be finite");
    std::vector<double> grid(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        grid[static_cast<std::size_t>(i)] =
            n == 1 ? start : start + (stop - start) * static_cast<double>(i) / static_cast<double>(n - 1);
    }
    return grid;
}

std::vector<int64_t> BettiCurve(const PersistenceDiagram& diagram, const std::vector<double>& grid) {
    CheckGrid(grid, "BettiCurve");
    std::vector<int64_t> curve(grid.size(), 0);
    for (const PersistencePair& p : diagram.pairs) {
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (p.birth <= grid[i] && grid[i] < p.death) ++curve[i];
        }
    }
    return curve;
}

Certified<PersistenceLandscape> ComputePersistenceLandscape(const PersistenceDiagram& diagram,
                                                            const std::vector<double>& grid, int64_t layers) {
    if (layers < 1) throw std::invalid_argument("ComputePersistenceLandscape: layers must be at least 1");
    CheckGrid(grid, "ComputePersistenceLandscape");
    Certified<PersistenceLandscape> out;
    out.value.grid = grid;
    out.value.layers.assign(static_cast<std::size_t>(layers), std::vector<double>(grid.size(), 0.0));
    out.terms = {{"the diagram's error bound (landscapes are 1-Lipschitz in the bottleneck distance)",
                  diagram.error_bound}};
    std::vector<double> tents;
    for (std::size_t i = 0; i < grid.size(); ++i) {
        const double t = grid[i];
        tents.clear();
        for (const PersistencePair& p : diagram.pairs) {
            if (p.essential()) continue;
            const double tent = std::min(t - p.birth, p.death - t);
            if (tent > 0) tents.push_back(tent);
        }
        const std::size_t k = std::min(tents.size(), static_cast<std::size_t>(layers));
        std::partial_sort(tents.begin(), tents.begin() + static_cast<std::ptrdiff_t>(k), tents.end(),
                          std::greater<double>());
        for (std::size_t j = 0; j < k; ++j) out.value.layers[j][i] = tents[j];
    }
    return out;
}

PersistenceImage ComputePersistenceImage(const PersistenceDiagram& diagram, const PersistenceImageOptions& options) {
    if (options.width < 1 || options.height < 1) {
        throw std::invalid_argument("ComputePersistenceImage: width and height must be at least 1");
    }
    PersistenceImage image;
    image.width = options.width;
    image.height = options.height;
    image.birth_min = options.birth_min;
    image.birth_max = options.birth_max;
    image.persistence_max = options.persistence_max;
    double largest_birth = 0, largest_persistence = 0;
    for (const PersistencePair& p : diagram.pairs) {
        if (p.essential()) continue;
        largest_birth = std::max(largest_birth, p.birth);
        largest_persistence = std::max(largest_persistence, p.persistence());
    }
    if (image.birth_max <= 0) image.birth_max = largest_birth;
    if (image.persistence_max <= 0) image.persistence_max = largest_persistence;
    // An H0 diagram's points are all born at 0: give the birth axis some width anyway.
    if (image.birth_max <= image.birth_min) image.birth_max = image.birth_min + std::max(image.persistence_max, 1.0);
    if (image.persistence_max <= 0) image.persistence_max = 1;
    if (!(image.birth_max > image.birth_min) || !std::isfinite(image.birth_max) || !std::isfinite(image.persistence_max)) {
        throw std::invalid_argument("ComputePersistenceImage: the ranges are empty or not finite");
    }
    image.sigma = options.sigma > 0 ? options.sigma : image.persistence_max / 10;
    image.values.assign(static_cast<std::size_t>(image.width * image.height), 0.0);

    const double pixel_w = (image.birth_max - image.birth_min) / static_cast<double>(image.width);
    const double pixel_h = image.persistence_max / static_cast<double>(image.height);
    const double scale = 1.0 / (image.sigma * std::sqrt(2.0));
    // The mass of N(mean, sigma^2) in [lo, hi).
    auto mass = [&](double mean, double lo, double hi) {
        return 0.5 * (std::erf((hi - mean) * scale) - std::erf((lo - mean) * scale));
    };
    std::vector<double> column_mass(static_cast<std::size_t>(image.width)), row_mass(static_cast<std::size_t>(image.height));
    for (const PersistencePair& p : diagram.pairs) {
        if (p.essential()) continue;
        const double persistence = p.persistence();
        const double weight = std::min(1.0, persistence / image.persistence_max);
        if (weight <= 0) continue;
        for (int64_t c = 0; c < image.width; ++c) {
            const double lo = image.birth_min + pixel_w * static_cast<double>(c);
            column_mass[static_cast<std::size_t>(c)] = mass(p.birth, lo, lo + pixel_w);
        }
        for (int64_t r = 0; r < image.height; ++r) {
            const double lo = pixel_h * static_cast<double>(r);
            row_mass[static_cast<std::size_t>(r)] = mass(persistence, lo, lo + pixel_h);
        }
        for (int64_t r = 0; r < image.height; ++r) {
            for (int64_t c = 0; c < image.width; ++c) {
                image.values[static_cast<std::size_t>(r * image.width + c)] +=
                    weight * row_mass[static_cast<std::size_t>(r)] * column_mass[static_cast<std::size_t>(c)];
            }
        }
    }
    return image;
}

double PersistentEntropy(const PersistenceDiagram& diagram) {
    double total = 0;
    for (const PersistencePair& p : diagram.pairs) {
        if (!p.essential()) total += p.persistence();
    }
    if (total <= 0) return 0;
    double entropy = 0;
    for (const PersistencePair& p : diagram.pairs) {
        if (p.essential() || p.persistence() <= 0) continue;
        const double q = p.persistence() / total;
        entropy -= q * std::log(q);
    }
    return entropy;
}

}  // namespace pulsatrix
