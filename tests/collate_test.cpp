#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "pulsatrix/collate.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {
namespace {

class CollateTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(CollateTest, StacksEachFieldPositionAcrossSamples) {
    std::vector<Sample> samples{
        Sample{{Tensor(Shape({1, 2}), &backend, {1.0f, 2.0f}), Tensor(Shape({1}), &backend, {0.0f})}},
        Sample{{Tensor(Shape({1, 2}), &backend, {3.0f, 4.0f}), Tensor(Shape({1}), &backend, {1.0f})}},
    };

    Batch batch = DefaultCollate(std::move(samples), &backend);

    ASSERT_EQ(batch.fields.size(), 2u);
    EXPECT_EQ(batch.size(), 2);
    EXPECT_EQ(batch.fields[0].shape(), Shape({2, 2}));
    EXPECT_FLOAT_EQ(batch.fields[0].data()[0], 1.0f);
    EXPECT_FLOAT_EQ(batch.fields[0].data()[3], 4.0f);
    EXPECT_EQ(batch.fields[1].shape(), Shape({2}));
    EXPECT_FLOAT_EQ(batch.fields[1].data()[0], 0.0f);
    EXPECT_FLOAT_EQ(batch.fields[1].data()[1], 1.0f);
}

TEST_F(CollateTest, ThrowsOnEmptySampleList) {
    EXPECT_THROW(DefaultCollate({}, &backend), std::invalid_argument);
}

TEST_F(CollateTest, ThrowsOnFieldCountMismatch) {
    std::vector<Sample> samples{
        Sample{{Tensor(Shape({1}), &backend, {1.0f})}},
        Sample{{Tensor(Shape({1}), &backend, {2.0f}), Tensor(Shape({1}), &backend, {3.0f})}},
    };
    EXPECT_THROW(DefaultCollate(std::move(samples), &backend), std::invalid_argument);
}

TEST_F(CollateTest, ThrowsOnFieldShapeMismatch) {
    std::vector<Sample> samples{
        Sample{{Tensor(Shape({1, 2}), &backend, {1.0f, 2.0f})}},
        Sample{{Tensor(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f})}},
    };
    EXPECT_THROW(DefaultCollate(std::move(samples), &backend), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
