/** @file matrix_decompositions.hpp
 *  @brief Small dense decompositions on the host: symmetric eigensolver, power iteration, QR
 *         and SVD (roadmap FND-4).
 *  @ingroup interpretability_agnostic
 *  @note For the matrices interpretability works with -- covariances of activations, weight
 *        matrices, Gram matrices -- up to a few hundred rows and columns. Inputs are CPU
 *        tensors; arithmetic is in double; results are float tensors on the input's backend.
 *  @note Eigenvector and singular-vector signs are arbitrary mathematically, so every result
 *        here is made canonical: in each vector, the largest-magnitude entry (the first, on a
 *        tie) is positive. Values come in descending order, ties kept in a fixed order. The
 *        same input therefore always gives the same output, which makes vectors comparable
 *        across runs and models (PCA directions, intruder-dimension checks).
 */
#pragma once

#include <cstdint>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief SymmetricEigen()'s result. */
struct EigenResult {
    /** @brief Shape `(n)`: eigenvalues, largest first. */
    Tensor values;
    /** @brief Shape `(n, n)`: column j is the unit eigenvector for `values[j]`. */
    Tensor vectors;
};

/** @brief PowerIteration()'s result. */
struct DominantEigenResult {
    /** @brief The dominant eigenvalue (largest in magnitude), as a Rayleigh quotient. */
    float value;
    /** @brief Shape `(n)`: its unit eigenvector, sign made canonical. */
    Tensor vector;
    /** @brief Iterations run. Equals max_iterations when not converged. */
    int64_t iterations;
    /** @brief Whether successive vectors agreed to within the tolerance. */
    bool converged;
};

/** @brief QR()'s result, the thin factorization A = Q R. */
struct QRResult {
    /** @brief Shape `(m, n)`: orthonormal columns. */
    Tensor q;
    /** @brief Shape `(n, n)`: upper triangular, non-negative diagonal. */
    Tensor r;
};

/** @brief SVD()'s result, the thin factorization A = U diag(S) V^T with k = min(m, n). */
struct SVDResult {
    /** @brief Shape `(m, k)`: orthonormal left singular vectors. */
    Tensor u;
    /** @brief Shape `(k)`: singular values, largest first, non-negative. */
    Tensor s;
    /** @brief Shape `(n, k)`: orthonormal right singular vectors, signs made canonical. */
    Tensor v;
};

/**
 * @brief All eigenvalues and eigenvectors of a symmetric matrix, by Householder reduction to
 *        tridiagonal form followed by QL with implicit shifts.
 * @param a Square `(n, n)` CPU tensor, symmetric to within a relative 1e-5, all finite.
 * @note O(n^3): about 60 ms at n = 256 and 0.8 s at n = 512 (Release, one core). For a
 *       repeated eigenvalue, any orthonormal basis of its eigenspace is returned.
 * @throws std::invalid_argument if `a` is not square rank 2, is not on the CPU, has a
 *         non-finite entry, or is not symmetric.
 * @throws std::runtime_error if QL fails to converge in 60 iterations for some eigenvalue,
 *         which does not happen for finite symmetric input in practice.
 */
[[nodiscard]] EigenResult SymmetricEigen(const Tensor& a);

/**
 * @brief The dominant eigenpair of a symmetric matrix by power iteration -- cheaper than
 *        SymmetricEigen() when only the top eigenpair is needed (spectral norm, stable rank).
 * @param a Square symmetric `(n, n)` CPU tensor, all finite.
 * @param max_iterations Iteration budget; must be >= 1.
 * @param tolerance Converged when successive unit vectors differ (up to sign) by at most this.
 * @param seed Seeds the deterministic LCG that draws the starting vector.
 * @note Converges at the rate |lambda2 / lambda1|. When the two largest eigenvalues tie in
 *       magnitude (e.g. +1 and -1) the dominant vector is not unique: the iteration does not
 *       settle and the result reports `converged = false` rather than an arbitrary answer.
 * @throws std::invalid_argument for the same inputs SymmetricEigen() rejects, a
 *         max_iterations below 1, or a tolerance that is not positive.
 */
[[nodiscard]] DominantEigenResult PowerIteration(const Tensor& a, int64_t max_iterations = 1000,
                                                 float tolerance = 1e-6f, uint64_t seed = 0);

/**
 * @brief Thin QR factorization by Householder reflections.
 * @param a `(m, n)` CPU tensor with m >= n, all finite. Rank-deficient input is fine.
 * @throws std::invalid_argument if `a` is not rank 2, is wide (m < n), is not on the CPU, or
 *         has a non-finite entry.
 */
[[nodiscard]] QRResult QR(const Tensor& a);

/**
 * @brief Thin singular value decomposition by one-sided (Hestenes) Jacobi rotations.
 * @note Accurate to high relative precision for small singular values. About 50 ms for a
 *       256 x 128 matrix and 1.6 s for 512 x 256 (Release, one core).
 * @param a `(m, n)` CPU tensor of any shape, all finite. Rank-deficient input is fine: left
 *        singular vectors for zero singular values are completed to an orthonormal set.
 * @throws std::invalid_argument if `a` is not rank 2, is not on the CPU, or has a non-finite
 *         entry.
 */
[[nodiscard]] SVDResult SVD(const Tensor& a);

}  // namespace pulsatrix
