#include <gtest/gtest.h>

#include <stdexcept>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/video_transforms.hpp"

namespace pulsatrix {
namespace {

class VideoTransformsTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// 4 frames, 1 channel, 1x1 pixel each, value = frame index -- directly identifies which
// source frame was selected.
Sample MakeIndexedClip(DeviceBackend* backend, int64_t total_frames) {
    Tensor clip(Shape({total_frames, 1, 1, 1}), backend);
    for (int64_t t = 0; t < total_frames; ++t) {
        clip.at({t, 0, 0, 0}) = static_cast<float>(t);
    }
    return Sample{{std::move(clip)}};
}

TEST_F(VideoTransformsTest, SamplesEvenlySpacedFrames) {
    Sample sample = MakeIndexedClip(&backend, 4);
    UniformFrameSampleTransform sampler(/*num_frames=*/2, &backend);

    Sample result = sampler.apply(std::move(sample));

    EXPECT_EQ(result.fields[0].shape(), Shape({2, 1, 1, 1}));
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 0, 0}), 0.0f);  // src frame (0*4)/2 = 0
    EXPECT_FLOAT_EQ(result.fields[0].at({1, 0, 0, 0}), 2.0f);  // src frame (1*4)/2 = 2
}

TEST_F(VideoTransformsTest, SamplingAllFramesIsIdentity) {
    Sample sample = MakeIndexedClip(&backend, 4);
    UniformFrameSampleTransform sampler(/*num_frames=*/4, &backend);

    Sample result = sampler.apply(std::move(sample));

    EXPECT_EQ(result.fields[0].shape(), Shape({4, 1, 1, 1}));
    for (int64_t t = 0; t < 4; ++t) {
        EXPECT_FLOAT_EQ(result.fields[0].at({t, 0, 0, 0}), static_cast<float>(t));
    }
}

TEST_F(VideoTransformsTest, SamplesSingleMiddleFrame) {
    Sample sample = MakeIndexedClip(&backend, 6);
    UniformFrameSampleTransform sampler(/*num_frames=*/3, &backend);

    Sample result = sampler.apply(std::move(sample));

    EXPECT_EQ(result.fields[0].shape(), Shape({3, 1, 1, 1}));
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 0, 0}), 0.0f);  // (0*6)/3 = 0
    EXPECT_FLOAT_EQ(result.fields[0].at({1, 0, 0, 0}), 2.0f);  // (1*6)/3 = 2
    EXPECT_FLOAT_EQ(result.fields[0].at({2, 0, 0, 0}), 4.0f);  // (2*6)/3 = 4
}

TEST_F(VideoTransformsTest, ThrowsWhenRequestingMoreFramesThanAvailable) {
    Sample sample = MakeIndexedClip(&backend, 2);
    UniformFrameSampleTransform sampler(/*num_frames=*/3, &backend);
    EXPECT_THROW(sampler.apply(std::move(sample)), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
