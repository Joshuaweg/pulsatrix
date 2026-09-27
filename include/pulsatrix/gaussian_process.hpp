/** @file gaussian_process.hpp
 *  @brief Gaussian-Process regression surrogate for Bayesian Optimization -- the "gaussian
 *         bands" technique named at this campaign's own drafting (Snoek, Larochelle, Adams,
 *         "Practical Bayesian Optimization of Machine Learning Algorithms," NeurIPS 2012).
 *  @ingroup hyperparameter_optimization
 *  @note Zero-mean prior, squared-exponential (RBF) kernel. Posterior mean/variance both
 *        reduce to solving a linear system against the training kernel matrix -- resolved
 *        via SolveLinearSystem (linear_algebra.hpp), reused rather than duplicated; see that
 *        file's own header comment for this campaign's Decision Point (c) resolution
 *        (Gaussian elimination per query, not a new Cholesky utility, at this campaign's
 *        stated small-trial-count scope).
 */
#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/linear_algebra.hpp"

namespace pulsatrix {

/**
 * @brief A fitted (or queryable-before-fitting-throws) Gaussian Process regressor with a
 *        squared-exponential kernel: k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 *
 *        length_scale^2)), plus additive observation noise (a small noise_variance is also
 *        standard GP practice purely as numerical "jitter" to keep the kernel matrix
 *        well-conditioned, independent of whether the underlying objective is actually
 *        noisy).
 */
class GaussianProcessRegressor {
public:
    /**
     * @param sigma_f Kernel signal variance (prior variance at any single point, before
     *        conditioning on data).
     * @param length_scale Kernel length scale -- how far apart two points must be before the
     *        kernel considers them roughly uncorrelated.
     * @param noise_variance Added to the kernel matrix's diagonal. Must be non-negative.
     * @throws std::invalid_argument if sigma_f <= 0, length_scale <= 0, or noise_variance < 0.
     */
    GaussianProcessRegressor(float sigma_f, float length_scale, float noise_variance)
        : sigma_f_(sigma_f), length_scale_(length_scale), noise_variance_(noise_variance) {
        if (sigma_f <= 0.0f) {
            throw std::invalid_argument("GaussianProcessRegressor: sigma_f must be positive");
        }
        if (length_scale <= 0.0f) {
            throw std::invalid_argument("GaussianProcessRegressor: length_scale must be positive");
        }
        if (noise_variance < 0.0f) {
            throw std::invalid_argument("GaussianProcessRegressor: noise_variance must be non-negative");
        }
    }

    /**
     * @brief Fits the GP to (inputs[i], targets[i]) observation pairs -- computes and solves
     *        the training kernel matrix once; Predict (below) reuses this fit.
     * @throws std::invalid_argument if inputs is empty, inputs.size() != targets.size(), or
     *         inputs' rows are not all the same dimension.
     * @throws std::runtime_error -- propagated from SolveLinearSystem if the kernel matrix is
     *         near-singular (e.g. duplicate input points with noise_variance == 0).
     */
    void Fit(std::vector<std::vector<float>> inputs, std::vector<float> targets) {
        if (inputs.empty()) {
            throw std::invalid_argument("GaussianProcessRegressor::Fit: inputs must not be empty");
        }
        if (inputs.size() != targets.size()) {
            throw std::invalid_argument("GaussianProcessRegressor::Fit: inputs and targets must be the same size");
        }
        const size_t dim = inputs[0].size();
        for (const auto& row : inputs) {
            if (row.size() != dim) {
                throw std::invalid_argument("GaussianProcessRegressor::Fit: every input row must be the same dimension");
            }
        }

        const size_t n = inputs.size();
        std::vector<std::vector<float>> k(n, std::vector<float>(n, 0.0f));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                k[i][j] = Kernel(inputs[i], inputs[j]);
            }
            k[i][i] += noise_variance_;
        }

        alpha_ = SolveLinearSystem(k, targets);
        training_kernel_ = std::move(k);
        inputs_ = std::move(inputs);
        targets_ = std::move(targets);
    }

    /** @brief A posterior prediction: mean and variance at one query point. */
    struct Posterior {
        double mean;
        double variance;
    };

    /**
     * @brief Predicts the posterior mean/variance at x, given the data passed to Fit.
     * @throws std::runtime_error if Fit has not been called yet.
     * @throws std::invalid_argument if x's dimension doesn't match the training inputs'.
     * @throws std::runtime_error -- propagated from SolveLinearSystem if the kernel matrix is
     *         near-singular.
     * @note variance is clamped to >= 0 -- floating-point rounding can otherwise make it a
     *       tiny negative number exactly at (or extremely near) a training point, which is
     *       not a meaningful negative variance, just accumulated rounding error.
     */
    [[nodiscard]] Posterior Predict(const std::vector<float>& x) const {
        if (inputs_.empty()) {
            throw std::runtime_error("GaussianProcessRegressor::Predict: called before Fit");
        }
        if (x.size() != inputs_[0].size()) {
            throw std::invalid_argument("GaussianProcessRegressor::Predict: x dimension must match training inputs");
        }

        const size_t n = inputs_.size();
        std::vector<float> k_star(n);
        for (size_t i = 0; i < n; ++i) {
            k_star[i] = Kernel(x, inputs_[i]);
        }

        double mean = 0.0;
        for (size_t i = 0; i < n; ++i) {
            mean += static_cast<double>(k_star[i]) * static_cast<double>(alpha_[i]);
        }

        auto v = SolveLinearSystem(training_kernel_, k_star);
        double explained_variance = 0.0;
        for (size_t i = 0; i < n; ++i) {
            explained_variance += static_cast<double>(k_star[i]) * static_cast<double>(v[i]);
        }
        double variance = static_cast<double>(Kernel(x, x)) - explained_variance;
        if (variance < 0.0) {
            variance = 0.0;
        }

        return Posterior{mean, variance};
    }

private:
    [[nodiscard]] float Kernel(const std::vector<float>& a, const std::vector<float>& b) const {
        float squared_distance = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float d = a[i] - b[i];
            squared_distance += d * d;
        }
        return sigma_f_ * sigma_f_ *
               std::exp(-squared_distance / (2.0f * length_scale_ * length_scale_));
    }

    float sigma_f_;
    float length_scale_;
    float noise_variance_;
    std::vector<std::vector<float>> inputs_;
    std::vector<float> targets_;
    std::vector<std::vector<float>> training_kernel_;
    std::vector<float> alpha_;
};

}  // namespace pulsatrix
