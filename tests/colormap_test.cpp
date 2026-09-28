#include <gtest/gtest.h>

#include "pulsatrix/viz/colormap.hpp"

// Pure colormap/normalization math for the native viz module (no ImGui/ImPlot include --
// see plans/okay-we-have-now-buzzing-moth.md Phase 0). Encodes
// hc_information_visualization.md's color-design rules: Viridis (perceptually uniform,
// colorblind-safe) for sequential/unsigned magnitude, a Blue-White-Red diverging map for
// signed attribution, never a rainbow/jet map. Kept free of any rendering dependency so it
// stays unit-testable via GoogleTest regardless of PULSATRIX_ENABLE_VIZ.
namespace pulsatrix {
namespace {

TEST(NormalizeUnsignedTest, ClampsResultToZeroOneRange) {
    EXPECT_NEAR(NormalizeUnsigned(0.0f, 10.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(NormalizeUnsigned(5.0f, 10.0f), 0.5f, 1e-6f);
    EXPECT_NEAR(NormalizeUnsigned(10.0f, 10.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(NormalizeUnsigned(15.0f, 10.0f), 1.0f, 1e-6f);   // over range clamps to 1
    EXPECT_NEAR(NormalizeUnsigned(-5.0f, 10.0f), 0.0f, 1e-6f);  // under range clamps to 0
}

TEST(NormalizeUnsignedTest, ZeroMaxAbsReturnsZeroInsteadOfDividingByZero) {
    EXPECT_NEAR(NormalizeUnsigned(0.0f, 0.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(NormalizeUnsigned(5.0f, 0.0f), 0.0f, 1e-6f);
}

TEST(NormalizeSignedTest, ClampsResultToNegativeOneOneRange) {
    EXPECT_NEAR(NormalizeSigned(0.0f, 10.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(NormalizeSigned(5.0f, 10.0f), 0.5f, 1e-6f);
    EXPECT_NEAR(NormalizeSigned(-5.0f, 10.0f), -0.5f, 1e-6f);
    EXPECT_NEAR(NormalizeSigned(20.0f, 10.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(NormalizeSigned(-20.0f, 10.0f), -1.0f, 1e-6f);
}

TEST(NormalizeSignedTest, ZeroMaxAbsReturnsZeroInsteadOfDividingByZero) {
    EXPECT_NEAR(NormalizeSigned(5.0f, 0.0f), 0.0f, 1e-6f);
}

TEST(ViridisColormapTest, AtZeroReturnsDarkPurpleAnchor) {
    RgbColor c = ViridisColormap(0.0f);
    EXPECT_NEAR(c.r, 0.267f, 1e-3f);
    EXPECT_NEAR(c.g, 0.004f, 1e-3f);
    EXPECT_NEAR(c.b, 0.329f, 1e-3f);
}

TEST(ViridisColormapTest, AtOneReturnsBrightYellowAnchor) {
    RgbColor c = ViridisColormap(1.0f);
    EXPECT_NEAR(c.r, 0.993f, 1e-3f);
    EXPECT_NEAR(c.g, 0.906f, 1e-3f);
    EXPECT_NEAR(c.b, 0.145f, 1e-3f);
}

TEST(ViridisColormapTest, IsMonotonicallyBrighterAsValueIncreases) {
    // Perceptual-uniformity proxy: total luminance should not decrease as the normalized
    // value increases from 0 to 1, unlike a rainbow/jet map's non-monotonic false peaks.
    float previous_luminance = -1.0f;
    for (float t = 0.0f; t <= 1.0f; t += 0.1f) {
        RgbColor c = ViridisColormap(t);
        float luminance = 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
        EXPECT_GE(luminance, previous_luminance - 1e-3f);
        previous_luminance = luminance;
    }
}

TEST(ViridisColormapTest, ClampsOutOfRangeInputToNearestAnchor) {
    RgbColor below = ViridisColormap(-0.5f);
    RgbColor at_zero = ViridisColormap(0.0f);
    EXPECT_NEAR(below.r, at_zero.r, 1e-6f);
    EXPECT_NEAR(below.g, at_zero.g, 1e-6f);
    EXPECT_NEAR(below.b, at_zero.b, 1e-6f);

    RgbColor above = ViridisColormap(1.5f);
    RgbColor at_one = ViridisColormap(1.0f);
    EXPECT_NEAR(above.r, at_one.r, 1e-6f);
    EXPECT_NEAR(above.g, at_one.g, 1e-6f);
    EXPECT_NEAR(above.b, at_one.b, 1e-6f);
}

TEST(DivergingColormapTest, AtZeroReturnsNearWhiteMidpoint) {
    RgbColor c = DivergingColormap(0.0f);
    EXPECT_GT(c.r, 0.8f);
    EXPECT_GT(c.g, 0.8f);
    EXPECT_GT(c.b, 0.8f);
}

TEST(DivergingColormapTest, AtNegativeOneIsBlueDominant) {
    RgbColor c = DivergingColormap(-1.0f);
    EXPECT_GT(c.b, c.r);
}

TEST(DivergingColormapTest, AtPositiveOneIsRedDominant) {
    RgbColor c = DivergingColormap(1.0f);
    EXPECT_GT(c.r, c.b);
}

TEST(DivergingColormapTest, ClampsOutOfRangeInputToNearestAnchor) {
    RgbColor below = DivergingColormap(-2.0f);
    RgbColor at_negative_one = DivergingColormap(-1.0f);
    EXPECT_NEAR(below.r, at_negative_one.r, 1e-6f);
    EXPECT_NEAR(below.b, at_negative_one.b, 1e-6f);

    RgbColor above = DivergingColormap(2.0f);
    RgbColor at_positive_one = DivergingColormap(1.0f);
    EXPECT_NEAR(above.r, at_positive_one.r, 1e-6f);
    EXPECT_NEAR(above.b, at_positive_one.b, 1e-6f);
}

}  // namespace
}  // namespace pulsatrix
