#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/matrix_decompositions.hpp"

namespace pulsatrix {
namespace {

// Dense row-major helpers in double, independent of the code under test.
using Mat = std::vector<double>;

Mat to_mat(const Tensor& t) { return Mat(t.data(), t.data() + t.numel()); }

Mat matmul(const Mat& a, const Mat& b, int64_t m, int64_t k, int64_t n) {
    Mat c(static_cast<size_t>(m * n), 0.0);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t p = 0; p < k; ++p)
            for (int64_t j = 0; j < n; ++j) c[i * n + j] += a[i * k + p] * b[p * n + j];
    return c;
}

Mat transpose(const Mat& a, int64_t m, int64_t n) {
    Mat t(a.size());
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j) t[j * m + i] = a[i * n + j];
    return t;
}

double max_abs_diff(const Mat& a, const Mat& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}

// Columns of q (rows x cols) are orthonormal: q^T q = I.
void expect_orthonormal_columns(const Tensor& q, double tol = 1e-5) {
    const int64_t rows = q.shape().dim(0), cols = q.shape().dim(1);
    Mat qtq = matmul(transpose(to_mat(q), rows, cols), to_mat(q), cols, rows, cols);
    for (int64_t i = 0; i < cols; ++i)
        for (int64_t j = 0; j < cols; ++j) EXPECT_NEAR(qtq[i * cols + j], i == j ? 1.0 : 0.0, tol) << i << "," << j;
}

// Canonical sign: each column's largest-magnitude entry (first one on ties) is positive.
void expect_canonical_signs(const Tensor& v) {
    const int64_t rows = v.shape().dim(0), cols = v.shape().dim(1);
    for (int64_t j = 0; j < cols; ++j) {
        int64_t best = 0;
        for (int64_t i = 1; i < rows; ++i)
            if (std::fabs(v.data()[i * cols + j]) > std::fabs(v.data()[best * cols + j])) best = i;
        EXPECT_GE(v.data()[best * cols + j], 0.0f) << "column " << j;
    }
}

std::vector<float> random_values(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

class MatrixDecompositionsTest : public ::testing::Test {
protected:
    CPUBackend backend;

    Tensor random_symmetric(int64_t n, unsigned seed) {
        std::vector<float> b = random_values(static_cast<size_t>(n * n), seed);
        std::vector<float> s(b.size());
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j) s[i * n + j] = b[i * n + j] + b[j * n + i];
        return Tensor(Shape({n, n}), &backend, s);
    }
    Tensor random_matrix(int64_t m, int64_t n, unsigned seed) {
        return Tensor(Shape({m, n}), &backend, random_values(static_cast<size_t>(m * n), seed));
    }
};

// --- SymmetricEigen --------------------------------------------------------------------

TEST_F(MatrixDecompositionsTest, EigenOfAKnownTwoByTwo) {
    Tensor a(Shape({2, 2}), &backend, {2, 1, 1, 2});
    EigenResult r = SymmetricEigen(a);
    EXPECT_NEAR(r.values.data()[0], 3.0f, 1e-6f);
    EXPECT_NEAR(r.values.data()[1], 1.0f, 1e-6f);
    const float h = 1.0f / std::sqrt(2.0f);
    // Column 0 = (1, 1)/sqrt2. Column 1 = (1, -1)/sqrt2 with its first max-|.| entry positive.
    EXPECT_NEAR(r.vectors.data()[0], h, 1e-6f);
    EXPECT_NEAR(r.vectors.data()[2], h, 1e-6f);
    EXPECT_NEAR(r.vectors.data()[1], h, 1e-6f);
    EXPECT_NEAR(r.vectors.data()[3], -h, 1e-6f);
}

TEST_F(MatrixDecompositionsTest, EigenSatisfiesAVEqualsVLambdaOnRandomSymmetric) {
    for (int64_t n : {int64_t{1}, int64_t{5}, int64_t{40}}) {
        Tensor a = random_symmetric(n, static_cast<unsigned>(n));
        EigenResult r = SymmetricEigen(a);
        ASSERT_EQ(r.values.shape(), Shape({n}));
        ASSERT_EQ(r.vectors.shape(), Shape({n, n}));
        Mat av = matmul(to_mat(a), to_mat(r.vectors), n, n, n);
        Mat v_lambda = to_mat(r.vectors);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j) v_lambda[i * n + j] *= r.values.data()[j];
        EXPECT_LT(max_abs_diff(av, v_lambda), 1e-4) << "n=" << n;
        expect_orthonormal_columns(r.vectors);
        expect_canonical_signs(r.vectors);
        double trace = 0.0, sum = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            trace += a.data()[i * n + i];
            sum += r.values.data()[i];
            if (i > 0) EXPECT_GE(r.values.data()[i - 1], r.values.data()[i]) << "not descending";
        }
        EXPECT_NEAR(trace, sum, 1e-4);
    }
}

TEST_F(MatrixDecompositionsTest, EigenOfRepeatedEigenvaluesStillGivesAnOrthonormalBasis) {
    Tensor identity(Shape({4, 4}), &backend, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
    EigenResult r = SymmetricEigen(identity);
    for (int64_t i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(r.values.data()[i], 1.0f);
    expect_orthonormal_columns(r.vectors);
}

TEST_F(MatrixDecompositionsTest, EigenIsDeterministic) {
    Tensor a = random_symmetric(12, 3);
    EigenResult first = SymmetricEigen(a), second = SymmetricEigen(a);
    EXPECT_EQ(to_mat(first.values), to_mat(second.values));
    EXPECT_EQ(to_mat(first.vectors), to_mat(second.vectors));
}

TEST_F(MatrixDecompositionsTest, EigenRejectsInvalidInput) {
    EXPECT_THROW((void)SymmetricEigen(Tensor(Shape({2, 3}), &backend)), std::invalid_argument);
    EXPECT_THROW((void)SymmetricEigen(Tensor(Shape({4}), &backend)), std::invalid_argument);
    EXPECT_THROW((void)SymmetricEigen(Tensor(Shape({2, 2}), &backend, {1, 2, 3, 4})), std::invalid_argument);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW((void)SymmetricEigen(Tensor(Shape({2, 2}), &backend, {1, nan, nan, 1})), std::invalid_argument);
}

// --- PowerIteration --------------------------------------------------------------------

TEST_F(MatrixDecompositionsTest, PowerIterationFindsTheDominantEigenpair) {
    Tensor b = random_matrix(10, 10, 21);
    // B^T B is symmetric positive semidefinite: its dominant eigenvalue is its largest.
    Mat btb = matmul(transpose(to_mat(b), 10, 10), to_mat(b), 10, 10, 10);
    Tensor a(Shape({10, 10}), &backend, std::vector<float>(btb.begin(), btb.end()));
    EigenResult full = SymmetricEigen(a);
    DominantEigenResult r = PowerIteration(a, 5000, 1e-7f);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.value, full.values.data()[0], 1e-3f * full.values.data()[0]);
    for (int64_t i = 0; i < 10; ++i) EXPECT_NEAR(r.vector.data()[i], full.vectors.data()[i * 10], 1e-3f) << i;
}

TEST_F(MatrixDecompositionsTest, PowerIterationIsDeterministicForAGivenSeed) {
    Tensor a = random_symmetric(8, 5);
    DominantEigenResult first = PowerIteration(a, 200, 1e-6f, 9), second = PowerIteration(a, 200, 1e-6f, 9);
    EXPECT_EQ(first.value, second.value);
    EXPECT_EQ(to_mat(first.vector), to_mat(second.vector));
    EXPECT_EQ(first.iterations, second.iterations);
}

TEST_F(MatrixDecompositionsTest, PowerIterationReportsNonConvergence) {
    // Eigenvalues +1 and -1 tie in magnitude: the iteration oscillates and never settles.
    Tensor a(Shape({2, 2}), &backend, {1, 0, 0, -1});
    DominantEigenResult r = PowerIteration(a, 50, 1e-6f);
    EXPECT_FALSE(r.converged);
    EXPECT_EQ(r.iterations, 50);
}

TEST_F(MatrixDecompositionsTest, PowerIterationRejectsInvalidInput) {
    EXPECT_THROW((void)PowerIteration(Tensor(Shape({2, 3}), &backend)), std::invalid_argument);
    EXPECT_THROW((void)PowerIteration(random_symmetric(3, 1), 0), std::invalid_argument);
}

// --- QR --------------------------------------------------------------------------------

TEST_F(MatrixDecompositionsTest, QRReconstructsWithOrthonormalQAndUpperTriangularR) {
    for (auto [m, n] : {std::pair<int64_t, int64_t>{7, 4}, {5, 5}, {6, 1}}) {
        Tensor a = random_matrix(m, n, static_cast<unsigned>(m * 10 + n));
        QRResult r = QR(a);
        ASSERT_EQ(r.q.shape(), Shape({m, n}));
        ASSERT_EQ(r.r.shape(), Shape({n, n}));
        EXPECT_LT(max_abs_diff(matmul(to_mat(r.q), to_mat(r.r), m, n, n), to_mat(a)), 1e-5) << m << "x" << n;
        expect_orthonormal_columns(r.q);
        for (int64_t i = 0; i < n; ++i) {
            EXPECT_GE(r.r.data()[i * n + i], 0.0f) << "diagonal of R must be non-negative";
            for (int64_t j = 0; j < i; ++j) EXPECT_EQ(r.r.data()[i * n + j], 0.0f) << i << "," << j;
        }
    }
}

TEST_F(MatrixDecompositionsTest, QRHandlesRankDeficientInput) {
    // Second column is twice the first.
    Tensor a(Shape({3, 2}), &backend, {1, 2, 2, 4, 3, 6});
    QRResult r = QR(a);
    EXPECT_LT(max_abs_diff(matmul(to_mat(r.q), to_mat(r.r), 3, 2, 2), to_mat(a)), 1e-5);
    expect_orthonormal_columns(r.q);
    EXPECT_NEAR(r.r.data()[3], 0.0f, 1e-5f);
}

TEST_F(MatrixDecompositionsTest, QRRejectsWideAndInvalidInput) {
    EXPECT_THROW((void)QR(random_matrix(2, 3, 1)), std::invalid_argument);
    EXPECT_THROW((void)QR(Tensor(Shape({4}), &backend)), std::invalid_argument);
}

// --- SVD -------------------------------------------------------------------------------

void expect_valid_svd(const Tensor& a, const SVDResult& r) {
    const int64_t m = a.shape().dim(0), n = a.shape().dim(1), k = std::min(m, n);
    ASSERT_EQ(r.u.shape(), Shape({m, k}));
    ASSERT_EQ(r.s.shape(), Shape({k}));
    ASSERT_EQ(r.v.shape(), Shape({n, k}));
    Mat us = to_mat(r.u);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < k; ++j) us[i * k + j] *= r.s.data()[j];
    Mat usvt = matmul(us, transpose(to_mat(r.v), n, k), m, k, n);
    EXPECT_LT(max_abs_diff(usvt, to_mat(a)), 1e-5) << m << "x" << n;
    expect_orthonormal_columns(r.u);
    expect_orthonormal_columns(r.v);
    expect_canonical_signs(r.v);
    for (int64_t j = 0; j < k; ++j) {
        EXPECT_GE(r.s.data()[j], 0.0f);
        if (j > 0) EXPECT_GE(r.s.data()[j - 1], r.s.data()[j]) << "not descending";
    }
}

TEST_F(MatrixDecompositionsTest, SVDOfTallSquareAndWideMatrices) {
    for (auto [m, n] : {std::pair<int64_t, int64_t>{9, 4}, {6, 6}, {3, 8}, {1, 5}, {5, 1}}) {
        Tensor a = random_matrix(m, n, static_cast<unsigned>(m * 100 + n));
        expect_valid_svd(a, SVD(a));
    }
}

TEST_F(MatrixDecompositionsTest, SingularValuesAreRootsOfTheGramEigenvalues) {
    Tensor a = random_matrix(8, 5, 77);
    SVDResult r = SVD(a);
    Mat ata = matmul(transpose(to_mat(a), 8, 5), to_mat(a), 5, 8, 5);
    EigenResult gram = SymmetricEigen(Tensor(Shape({5, 5}), &backend, std::vector<float>(ata.begin(), ata.end())));
    for (int64_t j = 0; j < 5; ++j) EXPECT_NEAR(r.s.data()[j], std::sqrt(gram.values.data()[j]), 1e-4f) << j;
}

TEST_F(MatrixDecompositionsTest, SVDOfRankDeficientMatrixHasZeroSingularValuesAndOrthonormalU) {
    // Rank 1: every row is a multiple of (1, 2, 3).
    Tensor a(Shape({4, 3}), &backend, {1, 2, 3, 2, 4, 6, -1, -2, -3, 0, 0, 0});
    SVDResult r = SVD(a);
    expect_valid_svd(a, r);
    EXPECT_NEAR(r.s.data()[1], 0.0f, 1e-5f);
    EXPECT_NEAR(r.s.data()[2], 0.0f, 1e-5f);
}

TEST_F(MatrixDecompositionsTest, SVDOfZeroMatrix) {
    Tensor a(Shape({3, 2}), &backend, {0, 0, 0, 0, 0, 0});
    SVDResult r = SVD(a);
    expect_valid_svd(a, r);
}

TEST_F(MatrixDecompositionsTest, SVDRejectsInvalidInput) {
    EXPECT_THROW((void)SVD(Tensor(Shape({4}), &backend)), std::invalid_argument);
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_THROW((void)SVD(Tensor(Shape({1, 2}), &backend, {1, inf})), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
