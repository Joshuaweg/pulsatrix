/** @file linear_algebra.hpp
 *  @brief Small, dense linear-system solve -- shared by weighted_linear_regression.hpp
 *         (Phase 3's LIME/KernelSHAP) and gaussian_process.hpp (this campaign's GP-BO
 *         surrogate).
 *  @ingroup interpretability_agnostic
 *  @note Promoted out of weighted_linear_regression.hpp's private `detail` namespace
 *        (2026-09-27, campaign_exai_dl_library_hyperparameter_optimization Phase 2 Mission 0
 *        activation) once a second real consumer (the GP surrogate's posterior mean/variance)
 *        needed the identical solver -- not duplicated, promoted. Resolves that campaign's own
 *        Decision Point (c): GP posterior variance reuses this Gaussian-elimination solver via
 *        a second solve per query (K*v = k_star), rather than a new Cholesky-based
 *        factor-once/query-many utility -- simpler to implement correctly, and sufficient at
 *        the small trial counts (tens to low hundreds) this campaign's own Risk Register
 *        already scopes GP-BO to. A Cholesky path remains a legitimate future optimization if
 *        that scope ever changes, not a correctness requirement now.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsatrix {

/**
 * @brief Solves A*x = b via Gaussian elimination with partial pivoting.
 * @throws std::runtime_error if a pivot is too close to zero (a near-singular system) -- a
 *         legitimate, caller-triggerable condition (e.g. degenerate/duplicate sample points),
 *         not a programmer error, per this codebase's assert-vs-throw convention.
 * @note A, b are taken by value -- elimination is done in place on the local copies.
 * @note Scoped to the small, dense, well-conditioned systems this codebase's callers actually
 *       produce -- not a general-purpose numerics library.
 */
[[nodiscard]] inline std::vector<float> SolveLinearSystem(std::vector<std::vector<float>> a,
                                                           std::vector<float> b) {
    const auto n = static_cast<int64_t>(b.size());
    for (int64_t col = 0; col < n; ++col) {
        int64_t pivot_row = col;
        float pivot_magnitude = std::fabs(a[static_cast<size_t>(col)][static_cast<size_t>(col)]);
        for (int64_t row = col + 1; row < n; ++row) {
            float magnitude = std::fabs(a[static_cast<size_t>(row)][static_cast<size_t>(col)]);
            if (magnitude > pivot_magnitude) {
                pivot_magnitude = magnitude;
                pivot_row = row;
            }
        }
        if (pivot_magnitude < 1e-8f) {
            throw std::runtime_error("SolveLinearSystem: near-singular system");
        }
        std::swap(a[static_cast<size_t>(col)], a[static_cast<size_t>(pivot_row)]);
        std::swap(b[static_cast<size_t>(col)], b[static_cast<size_t>(pivot_row)]);

        for (int64_t row = col + 1; row < n; ++row) {
            float factor = a[static_cast<size_t>(row)][static_cast<size_t>(col)] /
                           a[static_cast<size_t>(col)][static_cast<size_t>(col)];
            for (int64_t c = col; c < n; ++c) {
                a[static_cast<size_t>(row)][static_cast<size_t>(c)] -=
                    factor * a[static_cast<size_t>(col)][static_cast<size_t>(c)];
            }
            b[static_cast<size_t>(row)] -= factor * b[static_cast<size_t>(col)];
        }
    }

    std::vector<float> x(static_cast<size_t>(n), 0.0f);
    for (int64_t row = n - 1; row >= 0; --row) {
        float sum = b[static_cast<size_t>(row)];
        for (int64_t c = row + 1; c < n; ++c) {
            sum -= a[static_cast<size_t>(row)][static_cast<size_t>(c)] * x[static_cast<size_t>(c)];
        }
        x[static_cast<size_t>(row)] = sum / a[static_cast<size_t>(row)][static_cast<size_t>(row)];
    }
    return x;
}

}  // namespace pulsatrix
