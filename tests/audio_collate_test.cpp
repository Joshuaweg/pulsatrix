#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "pulsatrix/audio_collate.hpp"
#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

class AudioCollateTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(AudioCollateTest, ZeroPadsAllDifferentLengthsToBatchMax) {
    std::vector<Sample> samples{
        Sample{{Tensor(Shape({1, 1, 3}), &backend, {1.0f, 2.0f, 3.0f})}},  // len 3
        Sample{{Tensor(Shape({1, 1, 1}), &backend, {4.0f})}},              // len 1
        Sample{{Tensor(Shape({1, 1, 2}), &backend, {5.0f, 6.0f})}},        // len 2
    };

    CollateFn collate = AudioPadCollate();
    Batch batch = collate(std::move(samples), &backend);

    ASSERT_EQ(batch.fields.size(), 2u);
    EXPECT_EQ(batch.fields[0].shape(), Shape({3, 1, 3}));
    EXPECT_EQ(batch.fields[1].shape(), Shape({3}));

    EXPECT_FLOAT_EQ(batch.fields[0].at({1, 0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(batch.fields[0].at({1, 0, 1}), 0.0f);  // silence padding
    EXPECT_FLOAT_EQ(batch.fields[0].at({1, 0, 2}), 0.0f);

    EXPECT_FLOAT_EQ(batch.fields[1].data()[0], 3.0f);
    EXPECT_FLOAT_EQ(batch.fields[1].data()[1], 1.0f);
    EXPECT_FLOAT_EQ(batch.fields[1].data()[2], 2.0f);
}

TEST_F(AudioCollateTest, PreservesMultiChannelData) {
    std::vector<Sample> samples{
        Sample{{Tensor(Shape({1, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f})}},  // 2ch, len 2
        Sample{{Tensor(Shape({1, 2, 1}), &backend, {5.0f, 6.0f})}},              // 2ch, len 1
    };
    Batch batch = AudioPadCollate()(std::move(samples), &backend);

    EXPECT_EQ(batch.fields[0].shape(), Shape({2, 2, 2}));
    EXPECT_FLOAT_EQ(batch.fields[0].at({1, 0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(batch.fields[0].at({1, 1, 0}), 6.0f);
    EXPECT_FLOAT_EQ(batch.fields[0].at({1, 0, 1}), 0.0f);  // padding
}

TEST_F(AudioCollateTest, ThrowsOnChannelCountMismatch) {
    std::vector<Sample> samples{
        Sample{{Tensor(Shape({1, 1, 2}), &backend, {1.0f, 2.0f})}},
        Sample{{Tensor(Shape({1, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f})}},
    };
    EXPECT_THROW(AudioPadCollate()(std::move(samples), &backend), std::invalid_argument);
}

TEST_F(AudioCollateTest, ThrowsOnEmptySampleList) {
    EXPECT_THROW(AudioPadCollate()({}, &backend), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
