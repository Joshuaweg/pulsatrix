#include <gtest/gtest.h>

#include <stdexcept>

#include "pulsatrix/shape.hpp"

namespace pulsatrix {
namespace {

TEST(ShapeTest, RankMatchesDimensionCount) {
    Shape s({2, 3, 4});
    EXPECT_EQ(s.rank(), 3);
}

TEST(ShapeTest, NumelIsProductOfDimensions) {
    Shape s({2, 3, 4});
    EXPECT_EQ(s.numel(), 24);
}

TEST(ShapeTest, RankZeroScalarHasOneElement) {
    Shape s({});
    EXPECT_EQ(s.rank(), 0);
    EXPECT_EQ(s.numel(), 1);
}

TEST(ShapeTest, ZeroSizedDimensionMakesNumelZero) {
    Shape s({2, 0, 4});
    EXPECT_EQ(s.numel(), 0);
}

TEST(ShapeTest, EqualShapesCompareEqual) {
    Shape a({2, 3});
    Shape b({2, 3});
    EXPECT_EQ(a, b);
}

TEST(ShapeTest, DifferentDimsCompareNotEqual) {
    Shape a({2, 3});
    Shape b({3, 2});
    EXPECT_NE(a, b);
}

TEST(ShapeTest, DifferentRankCompareNotEqual) {
    Shape a({2, 3});
    Shape b({2, 3, 1});
    EXPECT_NE(a, b);
}

TEST(ShapeTest, DimAccessorReturnsCorrectDimension) {
    Shape s({5, 6, 7});
    EXPECT_EQ(s.dim(0), 5);
    EXPECT_EQ(s.dim(1), 6);
    EXPECT_EQ(s.dim(2), 7);
}

TEST(ShapeTest, IsReshapeCompatibleTrueWhenNumelMatches) {
    Shape a({2, 6});
    Shape b({3, 4});
    EXPECT_TRUE(a.is_reshape_compatible(b));
}

TEST(ShapeTest, IsReshapeCompatibleFalseWhenNumelDiffers) {
    Shape a({2, 6});
    Shape b({3, 5});
    EXPECT_FALSE(a.is_reshape_compatible(b));
}

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 0):
// Shape is an external boundary -- Shape objects are built directly from Python-supplied
// dimension lists in bindings/pulsatrix_py.cpp. A negative dimension must be rejected, not
// silently accepted and propagated into every downstream numel()/allocation computation.
TEST(ShapeTest, ConstructorThrowsOnNegativeDimension) {
    EXPECT_THROW(Shape({2, -3, 4}), std::invalid_argument);
}

TEST(ShapeTest, NumelThrowsOnOverflow) {
    // int64_t max is ~9.2e18; three dimensions of 3e6 each multiply to ~2.7e19, overflowing.
    Shape s({3000000, 3000000, 3000000});
    EXPECT_THROW({ (void)s.numel(); }, std::overflow_error);
}

// dim() previously had zero bounds checking, not even assert-gated -- raw out-of-bounds
// std::vector access, unconditional in every build. Internal invariant (Mission 0's
// classification table): an PULSATRIX_ASSERT, not a throw, since every call site in this
// codebase computes the index from an already-known-valid rank.
TEST(ShapeDeathTest, DimAbortsOnOutOfRangeIndex) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Shape s({2, 3});
    EXPECT_DEATH({ (void)s.dim(2); }, "PULSATRIX_ASSERT failed");
}

// campaign_exai_dl_library_data_pipeline, Mission 0: runtime-sized constructor needed by
// Tensor::Stack, which computes an output rank/dims list at runtime.
TEST(ShapeTest, VectorConstructorMatchesInitializerListConstructor) {
    std::vector<int64_t> dims{5, 6, 7};
    Shape s(dims);
    EXPECT_EQ(s.rank(), 3);
    EXPECT_EQ(s.numel(), 210);
    EXPECT_EQ(s.dim(0), 5);
    EXPECT_EQ(s.dim(1), 6);
    EXPECT_EQ(s.dim(2), 7);
    EXPECT_EQ(s, Shape({5, 6, 7}));
}

TEST(ShapeTest, VectorConstructorThrowsOnNegativeDimension) {
    std::vector<int64_t> dims{2, -3, 4};
    EXPECT_THROW(Shape s(dims), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
