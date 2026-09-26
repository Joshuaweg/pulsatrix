/** @file weighted_linear_regression.hpp
 *  @brief Weighted least squares via normal equations -- the shared fitting primitive
 *         Phase 3's LIME and KernelSHAP explainers both reduce to.
 *  @ingroup interpretability_agnostic
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

namespace detail {

/**
 * @brief Solves A*x = b via Gaussian elimination with partial pivoting.
 * @throws std::runtime_error if a pivot is too close to zero (a near-singular system) --
 *         a legitimate, caller-triggerable condition (e.g. degenerate perturbation
 *         sampling), not a programmer error, per this codebase's assert-vs-throw
 *         convention.
 * @note A, b are taken by value -- elimination is done in place on the local copies.
 */
inline std::vector<float> SolveLinearSystem(std::vector<std::vector<float>> a, std::vector<float> b) {
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
            throw std::runtime_error("WeightedLinearRegression: near-singular system (degenerate samples?)");
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

}  // namespace detail

/**
 * @brief Fits w* = argmin_w sum_i weight_i*(target_i - w^T sample_i)^2 + l2_lambda*||w||^2
 *        via the normal equations (X^T W X + l2_lambda*I) w = X^T W y.
 * @param samples Each row is one sample's feature vector; every row must be the same length.
 * @param targets One target value per sample. Must match samples.size().
 * @param weights One non-negative weight per sample. Must match samples.size().
 * @param l2_lambda Ridge regularization strength. 0 = ordinary weighted least squares.
 * @return Coefficients, one per feature (no intercept -- callers that need one center
 *         their targets/samples around a reference point first, e.g. LIME's local-model
 *         convention).
 * @throws std::runtime_error if the normal-equations system is near-singular (e.g.
 *         degenerate/collinear samples).
 * @throws std::invalid_argument if samples is empty, or targets/weights/any sample row's
 *         length doesn't match -- external boundary (
 *         campaign_exai_dl_library_adversarial_hardening.md, Mission 2, finding 15
 *         systemic sweep). Escalated from PULSATRIX_ASSERT-only for consistency with this
 *         function's own near-singular-system check above, which already throws.
 * @note Gaussian elimination, scoped to the small, dense, well-conditioned systems this
 *       codebase's explainers actually produce -- not a general-purpose numerics library.
 */
[[nodiscard]] inline std::vector<float> fit_weighted_linear_regression(
    const std::vector<std::vector<float>>& samples, const std::vector<float>& targets,
    const std::vector<float>& weights, float l2_lambda) {
    if (samples.empty()) {
        throw std::invalid_argument("fit_weighted_linear_regression: samples must not be empty");
    }
    if (samples.size() != targets.size() || samples.size() != weights.size()) {
        throw std::invalid_argument(
            "fit_weighted_linear_regression: samples, targets, and weights must be the same size");
    }

    const auto n_features = static_cast<int64_t>(samples[0].size());
    for (const auto& row : samples) {
        if (static_cast<int64_t>(row.size()) != n_features) {
            throw std::invalid_argument("fit_weighted_linear_regression: every sample row must be the same length");
        }
    }

    std::vector<std::vector<float>> a(static_cast<size_t>(n_features),
                                       std::vector<float>(static_cast<size_t>(n_features), 0.0f));
    std::vector<float> b(static_cast<size_t>(n_features), 0.0f);

    for (size_t k = 0; k < samples.size(); ++k) {
        for (int64_t i = 0; i < n_features; ++i) {
            b[static_cast<size_t>(i)] += weights[k] * samples[k][static_cast<size_t>(i)] * targets[k];
            for (int64_t j = 0; j < n_features; ++j) {
                a[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
                    weights[k] * samples[k][static_cast<size_t>(i)] * samples[k][static_cast<size_t>(j)];
            }
        }
    }
    for (int64_t i = 0; i < n_features; ++i) {
        a[static_cast<size_t>(i)][static_cast<size_t>(i)] += l2_lambda;
    }

    return detail::SolveLinearSystem(std::move(a), std::move(b));
}

}  // namespace pulsatrix
