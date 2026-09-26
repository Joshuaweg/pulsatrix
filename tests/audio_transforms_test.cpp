#include <gtest/gtest.h>

#include "pulsatrix/audio_transforms.hpp"
#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

class AudioTransformsTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// Mono linear ramp [0,10,20,30] at 4 units sample rate -- hand-computable interpolation.
Sample MakeRampSample(DeviceBackend* backend) {
    Tensor waveform(Shape({1, 1, 4}), backend, {0.0f, 10.0f, 20.0f, 30.0f});
    return Sample{{std::move(waveform)}};
}

TEST_F(AudioTransformsTest, UpsamplesByIntegerFactorViaLinearInterpolation) {
    Sample sample = MakeRampSample(&backend);
    ResampleTransform resample(/*target_sample_rate=*/8, /*source_sample_rate=*/4, &backend);

    Sample result = resample.apply(std::move(sample));

    EXPECT_EQ(result.fields[0].shape(), Shape({1, 1, 8}));
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 1}), 5.0f);   // halfway between 0 and 10
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 2}), 10.0f);
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 3}), 15.0f);  // halfway between 10 and 20
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 4}), 20.0f);
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 5}), 25.0f);  // halfway between 20 and 30
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 6}), 30.0f);
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 7}), 30.0f);  // past the end: clamped to last sample
}

TEST_F(AudioTransformsTest, DownsamplesByIntegerFactorViaLinearInterpolation) {
    Sample sample = MakeRampSample(&backend);
    ResampleTransform resample(/*target_sample_rate=*/2, /*source_sample_rate=*/4, &backend);

    Sample result = resample.apply(std::move(sample));

    EXPECT_EQ(result.fields[0].shape(), Shape({1, 1, 2}));
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 1}), 20.0f);
}

TEST_F(AudioTransformsTest, SameSampleRateIsIdentity) {
    Sample sample = MakeRampSample(&backend);
    ResampleTransform resample(/*target_sample_rate=*/4, /*source_sample_rate=*/4, &backend);

    Sample result = resample.apply(std::move(sample));

    EXPECT_EQ(result.fields[0].shape(), Shape({1, 1, 4}));
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 1}), 10.0f);
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 2}), 20.0f);
    EXPECT_FLOAT_EQ(result.fields[0].at({0, 0, 3}), 30.0f);
}

}  // namespace
}  // namespace pulsatrix
