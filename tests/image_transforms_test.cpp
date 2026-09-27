#include <gtest/gtest.h>

#include <stdexcept>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/image_transforms.hpp"

namespace pulsatrix {
namespace {

class ImageTransformsTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// 1 channel, 4x4 image, values 0..15 row-major (so pixel (y,x) = y*4+x), for easy hand
// verification of resize/crop mappings.
Sample MakeSequentialSample(DeviceBackend* backend) {
    Tensor image(Shape({1, 1, 4, 4}), backend);
    for (int64_t y = 0; y < 4; ++y) {
        for (int64_t x = 0; x < 4; ++x) {
            image.at({0, 0, y, x}) = static_cast<float>(y * 4 + x);
        }
    }
    return Sample{{std::move(image)}};
}

TEST_F(ImageTransformsTest, ResizeDownsamplesByIntegerFactorUsingNearestNeighbor) {
    // 4x4 -> 2x2: nearest-neighbor with this mapping picks source (0,0),(0,2),(2,0),(2,2).
    Sample sample = MakeSequentialSample(&backend);
    ResizeTransform resize(2, 2, &backend);

    Sample result = resize.apply(std::move(sample));

    EXPECT_EQ(result.fields[0].shape(), Shape({1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 0, 0}), 0.0f);   // src (0,0)
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 0, 1}), 2.0f);   // src (0,2)
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 1, 0}), 8.0f);   // src (2,0)
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 1, 1}), 10.0f);  // src (2,2)
}

TEST_F(ImageTransformsTest, CenterCropExtractsCenteredRegion) {
    // 4x4 crop to 2x2: centered region is rows/cols [1,2] -> pixels 5,6,9,10.
    Sample sample = MakeSequentialSample(&backend);
    CenterCropTransform crop(2, 2, &backend);

    Sample result = crop.apply(std::move(sample));

    EXPECT_EQ(result.fields[0].shape(), Shape({1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 0, 1}), 6.0f);
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 1, 0}), 9.0f);
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 1, 1}), 10.0f);
}

TEST_F(ImageTransformsTest, CenterCropThrowsWhenCropExceedsImageSize) {
    Sample sample = MakeSequentialSample(&backend);
    CenterCropTransform crop(5, 5, &backend);
    EXPECT_THROW(crop.apply(std::move(sample)), std::invalid_argument);
}

TEST_F(ImageTransformsTest, NormalizeAppliesPerChannelMeanStd) {
    Tensor image(Shape({1, 2, 1, 1}), &backend, {10.0f, 20.0f});
    Sample sample{{std::move(image)}};
    NormalizeTransform normalize({0.0f, 10.0f}, {2.0f, 5.0f});

    Sample result = normalize.apply(std::move(sample));

    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 0, 0}), 5.0f);  // (10-0)/2
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 1, 0, 0}), 2.0f);  // (20-10)/5
}

TEST_F(ImageTransformsTest, NormalizeThrowsOnChannelCountMismatch) {
    Tensor image(Shape({1, 2, 1, 1}), &backend, {10.0f, 20.0f});
    Sample sample{{std::move(image)}};
    NormalizeTransform normalize({0.0f}, {2.0f});
    EXPECT_THROW(normalize.apply(std::move(sample)), std::invalid_argument);
}

TEST_F(ImageTransformsTest, HorizontalFlipMirrorsEachRow) {
    Sample sample = MakeSequentialSample(&backend);
    HorizontalFlipTransform flip;

    Sample result = flip.apply(std::move(sample));

    // Row 0 was [0,1,2,3] -> flipped [3,2,1,0].
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 0, 2}), 1.0f);
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 0, 3}), 0.0f);
}

}  // namespace
}  // namespace pulsatrix
