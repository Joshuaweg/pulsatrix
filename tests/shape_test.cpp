#include <gtest/gtest.h>

#include "exai/shape.hpp"

namespace exai {
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

}  // namespace
}  // namespace exai
