#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/text_collate.hpp"

namespace pulsatrix {
namespace {

class TextCollateTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(TextCollateTest, PadsAllDifferentLengthsToBatchMax) {
    std::vector<Sample> samples{
        Sample{{Tensor(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f})}},  // len 3
        Sample{{Tensor(Shape({1, 1}), &backend, {4.0f})}},              // len 1
        Sample{{Tensor(Shape({1, 2}), &backend, {5.0f, 6.0f})}},        // len 2
    };

    CollateFn collate = PadCollate(/*pad_index=*/0.0f);
    Batch batch = collate(std::move(samples), &backend);

    ASSERT_EQ(batch.fields.size(), 2u);
    EXPECT_EQ(batch.fields[0].shape(), Shape({3, 3}));  // padded tokens
    EXPECT_EQ(batch.fields[1].shape(), Shape({3}));      // lengths

    // Row 0: [1,2,3] (no padding needed).
    EXPECT_FLOAT_EQ(batch.fields[0].at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(batch.fields[0].at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(batch.fields[0].at({0, 2}), 3.0f);
    // Row 1: [4, pad, pad].
    EXPECT_FLOAT_EQ(batch.fields[0].at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(batch.fields[0].at({1, 1}), 0.0f);
    EXPECT_FLOAT_EQ(batch.fields[0].at({1, 2}), 0.0f);
    // Row 2: [5, 6, pad].
    EXPECT_FLOAT_EQ(batch.fields[0].at({2, 0}), 5.0f);
    EXPECT_FLOAT_EQ(batch.fields[0].at({2, 1}), 6.0f);
    EXPECT_FLOAT_EQ(batch.fields[0].at({2, 2}), 0.0f);

    EXPECT_FLOAT_EQ(batch.fields[1].data()[0], 3.0f);
    EXPECT_FLOAT_EQ(batch.fields[1].data()[1], 1.0f);
    EXPECT_FLOAT_EQ(batch.fields[1].data()[2], 2.0f);
}

TEST_F(TextCollateTest, UsesConfiguredPadIndex) {
    std::vector<Sample> samples{
        Sample{{Tensor(Shape({1, 2}), &backend, {1.0f, 2.0f})}},
        Sample{{Tensor(Shape({1, 1}), &backend, {3.0f})}},
    };
    CollateFn collate = PadCollate(/*pad_index=*/-1.0f);
    Batch batch = collate(std::move(samples), &backend);
    EXPECT_FLOAT_EQ(batch.fields[0].at({1, 1}), -1.0f);
}

TEST_F(TextCollateTest, EqualLengthSamplesNeedNoPadding) {
    std::vector<Sample> samples{
        Sample{{Tensor(Shape({1, 2}), &backend, {1.0f, 2.0f})}},
        Sample{{Tensor(Shape({1, 2}), &backend, {3.0f, 4.0f})}},
    };
    CollateFn collate = PadCollate();
    Batch batch = collate(std::move(samples), &backend);
    EXPECT_EQ(batch.fields[0].shape(), Shape({2, 2}));
    EXPECT_FLOAT_EQ(batch.fields[1].data()[0], 2.0f);
    EXPECT_FLOAT_EQ(batch.fields[1].data()[1], 2.0f);
}

TEST_F(TextCollateTest, ThrowsOnEmptySampleList) {
    CollateFn collate = PadCollate();
    EXPECT_THROW(collate({}, &backend), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
