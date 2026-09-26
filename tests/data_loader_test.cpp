#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/data_loader.hpp"
#include "pulsatrix/dataset.hpp"
#include "pulsatrix/iterable_dataset.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {
namespace {

class RangeDataset : public Dataset {
public:
    explicit RangeDataset(int64_t count, DeviceBackend* backend) : count_(count), backend_(backend) {}

    [[nodiscard]] int64_t size() const override { return count_; }

    [[nodiscard]] Sample get(int64_t index) const override {
        if (index < 0 || index >= count_) {
            throw std::out_of_range("RangeDataset::get: index out of range");
        }
        return Sample{{Tensor(Shape({1}), backend_, {static_cast<float>(index)})}};
    }

private:
    int64_t count_;
    DeviceBackend* backend_;
};

class RangeIterableDataset : public IterableDataset {
public:
    RangeIterableDataset(int64_t count, DeviceBackend* backend) : count_(count), backend_(backend) {}

    void reset() override { position_ = 0; }

    [[nodiscard]] std::optional<Sample> next() override {
        if (position_ >= count_) {
            return std::nullopt;
        }
        return Sample{{Tensor(Shape({1}), backend_, {static_cast<float>(position_++)})}};
    }

private:
    int64_t count_;
    DeviceBackend* backend_;
    int64_t position_ = 0;
};

class DataLoaderTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(DataLoaderTest, SequentialOrderMatchesDatasetOrder) {
    DataLoaderOptions options;
    options.batch_size = 2;
    auto dataset = std::make_shared<RangeDataset>(4, &backend);
    DataLoader loader(dataset, &backend, options);

    auto batch0 = loader.next_batch();
    ASSERT_TRUE(batch0.has_value());
    EXPECT_FLOAT_EQ(batch0->fields[0].data()[0], 0.0f);
    EXPECT_FLOAT_EQ(batch0->fields[0].data()[1], 1.0f);

    auto batch1 = loader.next_batch();
    ASSERT_TRUE(batch1.has_value());
    EXPECT_FLOAT_EQ(batch1->fields[0].data()[0], 2.0f);
    EXPECT_FLOAT_EQ(batch1->fields[0].data()[1], 3.0f);

    EXPECT_FALSE(loader.next_batch().has_value());
}

TEST_F(DataLoaderTest, DropLastDropsIncompleteFinalBatch) {
    DataLoaderOptions options;
    options.batch_size = 3;
    options.drop_last = true;
    auto dataset = std::make_shared<RangeDataset>(5, &backend);
    DataLoader loader(dataset, &backend, options);

    ASSERT_TRUE(loader.next_batch().has_value());  // batch of 3
    EXPECT_FALSE(loader.next_batch().has_value());  // remaining 2 dropped
}

TEST_F(DataLoaderTest, KeepsIncompleteFinalBatchWhenDropLastIsFalse) {
    DataLoaderOptions options;
    options.batch_size = 3;
    options.drop_last = false;
    auto dataset = std::make_shared<RangeDataset>(5, &backend);
    DataLoader loader(dataset, &backend, options);

    ASSERT_TRUE(loader.next_batch().has_value());  // batch of 3
    auto last = loader.next_batch();
    ASSERT_TRUE(last.has_value());
    EXPECT_EQ(last->size(), 2);
    EXPECT_FALSE(loader.next_batch().has_value());
}

TEST_F(DataLoaderTest, NumBatchesAccountsForDropLast) {
    auto dataset = std::make_shared<RangeDataset>(5, &backend);

    DataLoaderOptions keep_options;
    keep_options.batch_size = 3;
    DataLoader keep_loader(dataset, &backend, keep_options);
    EXPECT_EQ(keep_loader.num_batches(), 2);

    DataLoaderOptions drop_options;
    drop_options.batch_size = 3;
    drop_options.drop_last = true;
    DataLoader drop_loader(dataset, &backend, drop_options);
    EXPECT_EQ(drop_loader.num_batches(), 1);
}

TEST_F(DataLoaderTest, ShuffleWithSameSeedIsDeterministic) {
    DataLoaderOptions options_a;
    options_a.batch_size = 1;
    options_a.shuffle = true;
    options_a.shuffle_seed = 99;
    auto dataset_a = std::make_shared<RangeDataset>(10, &backend);
    DataLoader loader_a(dataset_a, &backend, options_a);

    DataLoaderOptions options_b = options_a;
    auto dataset_b = std::make_shared<RangeDataset>(10, &backend);
    DataLoader loader_b(dataset_b, &backend, options_b);

    for (int i = 0; i < 10; ++i) {
        auto a = loader_a.next_batch();
        auto b = loader_b.next_batch();
        ASSERT_TRUE(a.has_value());
        ASSERT_TRUE(b.has_value());
        EXPECT_FLOAT_EQ(a->fields[0].data()[0], b->fields[0].data()[0]);
    }
}

TEST_F(DataLoaderTest, ResetEpochRestartsSequentialOrder) {
    DataLoaderOptions options;
    options.batch_size = 2;
    auto dataset = std::make_shared<RangeDataset>(2, &backend);
    DataLoader loader(dataset, &backend, options);

    ASSERT_TRUE(loader.next_batch().has_value());
    EXPECT_FALSE(loader.next_batch().has_value());

    loader.reset_epoch();
    auto batch = loader.next_batch();
    ASSERT_TRUE(batch.has_value());
    EXPECT_FLOAT_EQ(batch->fields[0].data()[0], 0.0f);
}

TEST_F(DataLoaderTest, ConstructorThrowsOnNullDataset) {
    EXPECT_THROW(DataLoader(std::shared_ptr<Dataset>(nullptr), &backend), std::invalid_argument);
}

TEST_F(DataLoaderTest, ConstructorThrowsOnNonPositiveBatchSize) {
    DataLoaderOptions options;
    options.batch_size = 0;
    auto dataset = std::make_shared<RangeDataset>(4, &backend);
    EXPECT_THROW(DataLoader(dataset, &backend, options), std::invalid_argument);
}

TEST_F(DataLoaderTest, IterableDatasetProducesBatchesInStreamOrder) {
    DataLoaderOptions options;
    options.batch_size = 2;
    auto dataset = std::make_shared<RangeIterableDataset>(4, &backend);
    DataLoader loader(dataset, &backend, options);

    auto batch0 = loader.next_batch();
    ASSERT_TRUE(batch0.has_value());
    EXPECT_FLOAT_EQ(batch0->fields[0].data()[0], 0.0f);
    EXPECT_FLOAT_EQ(batch0->fields[0].data()[1], 1.0f);

    auto batch1 = loader.next_batch();
    ASSERT_TRUE(batch1.has_value());
    EXPECT_FLOAT_EQ(batch1->fields[0].data()[0], 2.0f);
    EXPECT_FLOAT_EQ(batch1->fields[0].data()[1], 3.0f);

    EXPECT_FALSE(loader.next_batch().has_value());
}

TEST_F(DataLoaderTest, IterableDatasetConstructorThrowsOnMultipleWorkers) {
    DataLoaderOptions options;
    options.num_workers = 2;
    auto dataset = std::make_shared<RangeIterableDataset>(4, &backend);
    EXPECT_THROW(DataLoader(dataset, &backend, options), std::invalid_argument);
}

TEST_F(DataLoaderTest, NumBatchesThrowsForIterableDatasetSource) {
    auto dataset = std::make_shared<RangeIterableDataset>(4, &backend);
    DataLoader loader(dataset, &backend);
    EXPECT_THROW(loader.num_batches(), std::logic_error);
}

}  // namespace
}  // namespace pulsatrix
