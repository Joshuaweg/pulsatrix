#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "pulsatrix/linear_algebra.hpp"

namespace pulsatrix {
namespace {

TEST(SolveLinearSystemTest, SolvesExactTwoByTwoSystem) {
    // [2 1][x0]   [5]      exact solution: x0=2, x1=1
    // [1 3][x1] = [5]
    std::vector<std::vector<float>> a = {{2.0f, 1.0f}, {1.0f, 3.0f}};
    std::vector<float> b = {5.0f, 5.0f};
    auto x = SolveLinearSystem(a, b);
    ASSERT_EQ(x.size(), 2u);
    EXPECT_NEAR(x[0], 2.0f, 1e-5f);
    EXPECT_NEAR(x[1], 1.0f, 1e-5f);
}

TEST(SolveLinearSystemTest, SolvesIdentityTrivially) {
    std::vector<std::vector<float>> a = {{1.0f, 0.0f}, {0.0f, 1.0f}};
    std::vector<float> b = {3.5f, -2.0f};
    auto x = SolveLinearSystem(a, b);
    EXPECT_NEAR(x[0], 3.5f, 1e-6f);
    EXPECT_NEAR(x[1], -2.0f, 1e-6f);
}

TEST(SolveLinearSystemTest, ThrowsOnSingularSystem) {
    // Rows are linearly dependent (row1 = 2*row0) -- singular.
    std::vector<std::vector<float>> a = {{1.0f, 1.0f}, {2.0f, 2.0f}};
    std::vector<float> b = {1.0f, 2.0f};
    EXPECT_THROW(SolveLinearSystem(a, b), std::runtime_error);
}

}  // namespace
}  // namespace pulsatrix
