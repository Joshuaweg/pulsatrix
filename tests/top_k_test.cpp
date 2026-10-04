#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/top_k.hpp"

namespace pulsatrix {
namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

std::vector<float> values_of(const Tensor& t) { return std::vector<float>(t.data(), t.data() + t.numel()); }

// The specified order, written independently of the implementation: NaN ranks above every
// number, ties keep the lower index first (a stable sort), largest-first or smallest-first.
std::vector<int64_t> reference_order(const std::vector<float>& row, bool largest) {
    auto greater = [](float a, float b) { return (std::isnan(a) && !std::isnan(b)) || a > b; };
    std::vector<int64_t> order(row.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int64_t i, int64_t j) {
        return largest ? greater(row[i], row[j]) : greater(row[j], row[i]);
    });
    return order;
}

// Coarsely quantized so equal values (ties) are common.
std::vector<float> tie_heavy_values(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    for (auto& x : v) {
        seed = seed * 1664525u + 1013904223u;
        x = static_cast<float>(static_cast<int>((seed >> 16) % 9) - 4) * 0.5f;
    }
    return v;
}

class TopKTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(TopKTest, LargestOnAVector) {
    Tensor x(Shape({8}), &backend, {3, 1, 4, 1, 5, 9, 2, 6});
    TopKResult r = top_k(x, 3);
    EXPECT_EQ(r.values.shape(), Shape({3}));
    EXPECT_EQ(values_of(r.values), (std::vector<float>{9, 6, 5}));
    EXPECT_EQ(values_of(r.indices), (std::vector<float>{5, 7, 4}));
}

TEST_F(TopKTest, SmallestBreaksTiesTowardTheLowerIndex) {
    Tensor x(Shape({8}), &backend, {3, 1, 4, 1, 5, 9, 2, 6});
    TopKResult r = top_k(x, 3, /*largest=*/false);
    EXPECT_EQ(values_of(r.values), (std::vector<float>{1, 1, 2}));
    EXPECT_EQ(values_of(r.indices), (std::vector<float>{1, 3, 6}));
}

TEST_F(TopKTest, LargestBreaksTiesTowardTheLowerIndex) {
    Tensor x(Shape({5}), &backend, {2, 7, 2, 7, 2});
    TopKResult r = top_k(x, 4);
    EXPECT_EQ(values_of(r.values), (std::vector<float>{7, 7, 2, 2}));
    EXPECT_EQ(values_of(r.indices), (std::vector<float>{1, 3, 0, 2}));
}

TEST_F(TopKTest, NaNRanksAboveEveryNumber) {
    Tensor x(Shape({5}), &backend, {1.0f, kNaN, kInf, -kInf, kNaN});
    TopKResult largest = top_k(x, 3);
    EXPECT_EQ(values_of(largest.indices), (std::vector<float>{1, 4, 2}));
    EXPECT_TRUE(std::isnan(largest.values.data()[0]));
    EXPECT_TRUE(std::isnan(largest.values.data()[1]));
    EXPECT_EQ(largest.values.data()[2], kInf);

    TopKResult smallest = top_k(x, 3, /*largest=*/false);
    EXPECT_EQ(values_of(smallest.values), (std::vector<float>{-kInf, 1.0f, kInf}));
    EXPECT_EQ(values_of(smallest.indices), (std::vector<float>{3, 0, 2}));
}

TEST_F(TopKTest, KEqualToRowLengthIsAStableFullSort) {
    std::vector<float> v = tie_heavy_values(12, 7);
    Tensor x(Shape({12}), &backend, v);
    TopKResult r = top_k(x, 12);
    std::vector<int64_t> order = reference_order(v, true);
    for (size_t i = 0; i < order.size(); ++i) {
        EXPECT_EQ(r.indices.data()[i], static_cast<float>(order[i])) << i;
        EXPECT_EQ(r.values.data()[i], v[static_cast<size_t>(order[i])]) << i;
    }
}

TEST_F(TopKTest, SelectsAlongTheLastDimensionOfAnyRank) {
    Tensor x(Shape({2, 3, 5}), &backend, tie_heavy_values(30, 3));
    TopKResult r = top_k(x, 2);
    EXPECT_EQ(r.values.shape(), Shape({2, 3, 2}));
    EXPECT_EQ(r.indices.shape(), Shape({2, 3, 2}));
}

TEST_F(TopKTest, MatchesAStableSortReferenceRowByRow) {
    constexpr int64_t rows = 37, cols = 53;
    std::vector<float> v = tie_heavy_values(rows * cols, 11);
    v[5] = kNaN;
    v[cols + 9] = -kInf;
    Tensor x(Shape({rows, cols}), &backend, v);
    for (bool largest : {true, false}) {
        for (int64_t k : {int64_t{1}, int64_t{4}, int64_t{17}, cols}) {
            TopKResult r = top_k(x, k, largest);
            for (int64_t row = 0; row < rows; ++row) {
                std::vector<float> row_values(v.begin() + row * cols, v.begin() + (row + 1) * cols);
                std::vector<int64_t> order = reference_order(row_values, largest);
                for (int64_t j = 0; j < k; ++j) {
                    const float index = r.indices.data()[row * k + j];
                    ASSERT_EQ(index, static_cast<float>(order[static_cast<size_t>(j)]))
                        << "largest=" << largest << " k=" << k << " row=" << row << " j=" << j;
                    const float expected = row_values[static_cast<size_t>(order[static_cast<size_t>(j)])];
                    const float actual = r.values.data()[row * k + j];
                    EXPECT_TRUE(actual == expected || (std::isnan(actual) && std::isnan(expected)));
                }
            }
        }
    }
}

TEST_F(TopKTest, RejectsKOutsideOneToRowLength) {
    Tensor x(Shape({2, 4}), &backend, {1, 2, 3, 4, 5, 6, 7, 8});
    EXPECT_THROW((void)top_k(x, 0), std::invalid_argument);
    EXPECT_THROW((void)top_k(x, -1), std::invalid_argument);
    EXPECT_THROW((void)top_k(x, 5), std::invalid_argument);
}

TEST_F(TopKTest, RejectsEmptyInput) {
    Tensor empty(Shape({0}), &backend);
    EXPECT_THROW((void)top_k(empty, 1), std::invalid_argument);
}

TEST_F(TopKTest, ResultLivesOnTheInputsDevice) {
    Tensor x(Shape({3}), &backend, {1, 2, 3});
    TopKResult r = top_k(x, 1);
    EXPECT_EQ(r.values.device(), x.device());
    EXPECT_EQ(r.indices.device(), x.device());
}

}  // namespace
}  // namespace pulsatrix
