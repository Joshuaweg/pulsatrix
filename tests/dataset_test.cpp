#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/dataset.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {
namespace {

// Trivial in-memory Dataset used to exercise the abstract interface --
// campaign_exai_dl_library_data_pipeline, Mission 0, Objective 2.
class InMemoryDataset : public Dataset {
public:
    explicit InMemoryDataset(std::vector<Sample> samples) : samples_(std::move(samples)) {}

    [[nodiscard]] int64_t size() const override { return static_cast<int64_t>(samples_.size()); }

    [[nodiscard]] Sample get(int64_t index) const override {
        if (index < 0 || index >= size()) {
            throw std::out_of_range("InMemoryDataset::get: index out of range");
        }
        return samples_[static_cast<size_t>(index)];
    }

private:
    std::vector<Sample> samples_;
};

class DatasetTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(DatasetTest, SizeReturnsSampleCount) {
    InMemoryDataset dataset({Sample{{Tensor(Shape({1}), &backend, {1.0f})}},
                              Sample{{Tensor(Shape({1}), &backend, {2.0f})}}});
    EXPECT_EQ(dataset.size(), 2);
}

TEST_F(DatasetTest, GetReturnsCorrectSample) {
    InMemoryDataset dataset({Sample{{Tensor(Shape({1}), &backend, {1.0f})}},
                              Sample{{Tensor(Shape({1}), &backend, {2.0f})}}});
    Sample s = dataset.get(1);
    ASSERT_EQ(s.fields.size(), 1u);
    EXPECT_FLOAT_EQ(s.fields[0].data()[0], 2.0f);
}

TEST_F(DatasetTest, GetThrowsOnOutOfRangeIndex) {
    InMemoryDataset dataset({Sample{{Tensor(Shape({1}), &backend, {1.0f})}}});
    EXPECT_THROW(dataset.get(1), std::out_of_range);
    EXPECT_THROW(dataset.get(-1), std::out_of_range);
}

TEST_F(DatasetTest, EmptyDatasetHasZeroSize) {
    InMemoryDataset dataset({});
    EXPECT_EQ(dataset.size(), 0);
    EXPECT_THROW(dataset.get(0), std::out_of_range);
}

}  // namespace
}  // namespace pulsatrix
