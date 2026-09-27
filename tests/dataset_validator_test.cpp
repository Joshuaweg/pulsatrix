#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/csv_dataset.hpp"
#include "pulsatrix/dataset_validator.hpp"

namespace pulsatrix {
namespace {

std::string FixturePath(const std::string& name) {
    return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/" + name;
}

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

class DatasetValidatorTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(DatasetValidatorTest, ComputesCorrectStatisticsAcrossAllSamplesAndElements) {
    // Field 0 values across all 3 samples, flattened: 1,2,3,4,5,6.
    InMemoryDataset dataset({
        Sample{{Tensor(Shape({1, 2}), &backend, {1.0f, 2.0f})}},
        Sample{{Tensor(Shape({1, 2}), &backend, {3.0f, 4.0f})}},
        Sample{{Tensor(Shape({1, 2}), &backend, {5.0f, 6.0f})}},
    });

    DatasetStatistics stats = DatasetValidator::ComputeStatistics(dataset);

    ASSERT_EQ(stats.fields.size(), 1u);
    const FieldStatistics& field0 = stats.fields[0];
    EXPECT_EQ(field0.count, 6);
    EXPECT_NEAR(field0.mean, 3.5f, 1e-4f);
    EXPECT_NEAR(field0.std_dev, 1.7078251f, 1e-4f);
    EXPECT_FLOAT_EQ(field0.min_value, 1.0f);
    EXPECT_FLOAT_EQ(field0.max_value, 6.0f);
}

TEST_F(DatasetValidatorTest, ComputesIndependentStatisticsPerFieldPosition) {
    InMemoryDataset dataset({
        Sample{{Tensor(Shape({1}), &backend, {10.0f}), Tensor(Shape({1}), &backend, {100.0f})}},
        Sample{{Tensor(Shape({1}), &backend, {20.0f}), Tensor(Shape({1}), &backend, {200.0f})}},
    });

    DatasetStatistics stats = DatasetValidator::ComputeStatistics(dataset);

    ASSERT_EQ(stats.fields.size(), 2u);
    EXPECT_NEAR(stats.fields[0].mean, 15.0f, 1e-4f);
    EXPECT_NEAR(stats.fields[1].mean, 150.0f, 1e-4f);
}

TEST_F(DatasetValidatorTest, ThrowsOnSchemaMismatchAcrossSamples) {
    InMemoryDataset dataset({
        Sample{{Tensor(Shape({1}), &backend, {1.0f})}},
        Sample{{Tensor(Shape({1}), &backend, {2.0f}), Tensor(Shape({1}), &backend, {3.0f})}},
    });
    EXPECT_THROW(DatasetValidator::ComputeStatistics(dataset), std::runtime_error);
}

TEST_F(DatasetValidatorTest, ThrowsOnEmptyDataset) {
    InMemoryDataset dataset({});
    EXPECT_THROW(DatasetValidator::ComputeStatistics(dataset), std::invalid_argument);
}

TEST_F(DatasetValidatorTest, WorksGenericallyAgainstARealCsvDataset) {
    // simple.csv: (x1=1,x2=2,label=0), (x1=3,x2=4,label=1), (x1=5,x2=6,label=0).
    CsvDataset dataset(FixturePath("simple.csv"), {"x1", "x2"}, "label", &backend);

    DatasetStatistics stats = DatasetValidator::ComputeStatistics(dataset);

    ASSERT_EQ(stats.fields.size(), 2u);
    // Feature field: flattened 1,2,3,4,5,6 -- same distribution as the synthetic test above.
    EXPECT_NEAR(stats.fields[0].mean, 3.5f, 1e-4f);
    // Label field: 0,1,0.
    EXPECT_EQ(stats.fields[1].count, 3);
    EXPECT_NEAR(stats.fields[1].mean, 1.0f / 3.0f, 1e-4f);
    EXPECT_FLOAT_EQ(stats.fields[1].min_value, 0.0f);
    EXPECT_FLOAT_EQ(stats.fields[1].max_value, 1.0f);
}

}  // namespace
}  // namespace pulsatrix
