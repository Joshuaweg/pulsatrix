#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/iterable_dataset.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {
namespace {

// Trivial in-memory streaming Dataset -- campaign_exai_dl_library_data_pipeline, Mission 0,
// Objective 3.
class InMemoryIterableDataset : public IterableDataset {
public:
    explicit InMemoryIterableDataset(std::vector<Sample> samples) : samples_(std::move(samples)) {}

    void reset() override { position_ = 0; }

    [[nodiscard]] std::optional<Sample> next() override {
        if (position_ >= samples_.size()) {
            return std::nullopt;
        }
        return samples_[position_++];
    }

private:
    std::vector<Sample> samples_;
    size_t position_ = 0;
};

class IterableDatasetTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(IterableDatasetTest, NextYieldsSamplesInOrderThenNullopt) {
    InMemoryIterableDataset dataset({Sample{{Tensor(Shape({1}), &backend, {1.0f})}},
                                      Sample{{Tensor(Shape({1}), &backend, {2.0f})}}});
    dataset.reset();

    auto first = dataset.next();
    ASSERT_TRUE(first.has_value());
    EXPECT_FLOAT_EQ(first->fields[0].data()[0], 1.0f);

    auto second = dataset.next();
    ASSERT_TRUE(second.has_value());
    EXPECT_FLOAT_EQ(second->fields[0].data()[0], 2.0f);

    EXPECT_FALSE(dataset.next().has_value());
}

TEST_F(IterableDatasetTest, ResetRestartsIteration) {
    InMemoryIterableDataset dataset({Sample{{Tensor(Shape({1}), &backend, {1.0f})}}});
    dataset.reset();
    ASSERT_TRUE(dataset.next().has_value());
    EXPECT_FALSE(dataset.next().has_value());

    dataset.reset();
    auto again = dataset.next();
    ASSERT_TRUE(again.has_value());
    EXPECT_FLOAT_EQ(again->fields[0].data()[0], 1.0f);
}

TEST_F(IterableDatasetTest, EmptyStreamYieldsNulloptImmediately) {
    InMemoryIterableDataset dataset({});
    dataset.reset();
    EXPECT_FALSE(dataset.next().has_value());
}

}  // namespace
}  // namespace pulsatrix
