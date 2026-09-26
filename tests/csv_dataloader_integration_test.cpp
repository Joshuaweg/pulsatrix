#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/csv_dataset.hpp"
#include "pulsatrix/data_loader.hpp"

namespace pulsatrix {
namespace {

std::string FixturePath(const std::string& name) {
    return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/" + name;
}

class CsvDataLoaderIntegrationTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(CsvDataLoaderIntegrationTest, ProducesCorrectlyShapedBatchesInSequentialOrder) {
    auto dataset = std::make_shared<CsvDataset>(FixturePath("simple.csv"), std::vector<std::string>{"x1", "x2"},
                                                 "label", &backend);
    DataLoaderOptions options;
    options.batch_size = 2;
    DataLoader loader(dataset, &backend, options);

    auto batch0 = loader.next_batch();
    ASSERT_TRUE(batch0.has_value());
    EXPECT_EQ(batch0->size(), 2);
    EXPECT_EQ(batch0->fields[0].shape(), Shape({2, 2}));
    EXPECT_FLOAT_EQ(batch0->fields[0].data()[0], 1.0f);
    EXPECT_FLOAT_EQ(batch0->fields[0].data()[1], 2.0f);
    EXPECT_FLOAT_EQ(batch0->fields[0].data()[2], 3.0f);
    EXPECT_FLOAT_EQ(batch0->fields[0].data()[3], 4.0f);
    EXPECT_FLOAT_EQ(batch0->fields[1].data()[0], 0.0f);
    EXPECT_FLOAT_EQ(batch0->fields[1].data()[1], 1.0f);

    auto batch1 = loader.next_batch();
    ASSERT_TRUE(batch1.has_value());
    EXPECT_EQ(batch1->size(), 1);
    EXPECT_FLOAT_EQ(batch1->fields[0].data()[0], 5.0f);
    EXPECT_FLOAT_EQ(batch1->fields[1].data()[0], 0.0f);

    EXPECT_FALSE(loader.next_batch().has_value());
}

TEST_F(CsvDataLoaderIntegrationTest, ShuffleVisitsEveryRowExactlyOnceAcrossAllBatches) {
    auto dataset = std::make_shared<CsvDataset>(FixturePath("simple.csv"), std::vector<std::string>{"x1", "x2"},
                                                 "label", &backend);
    DataLoaderOptions options;
    options.batch_size = 1;
    options.shuffle = true;
    options.shuffle_seed = 123;
    DataLoader loader(dataset, &backend, options);

    std::vector<float> seen_first_features;
    while (auto batch = loader.next_batch()) {
        seen_first_features.push_back(batch->fields[0].data()[0]);
    }
    std::sort(seen_first_features.begin(), seen_first_features.end());
    EXPECT_EQ(seen_first_features, (std::vector<float>{1.0f, 3.0f, 5.0f}));
}

}  // namespace
}  // namespace pulsatrix
