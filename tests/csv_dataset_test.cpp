#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/csv_dataset.hpp"

namespace pulsatrix {
namespace {

std::string FixturePath(const std::string& name) {
    return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/" + name;
}

class CsvDatasetTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(CsvDatasetTest, SizeMatchesDataRowCount) {
    CsvDataset dataset(FixturePath("simple.csv"), {"x1", "x2"}, "label", &backend);
    EXPECT_EQ(dataset.size(), 3);
}

TEST_F(CsvDatasetTest, GetReturnsFeatureAndLabelTensorsMatchingFileContent) {
    CsvDataset dataset(FixturePath("simple.csv"), {"x1", "x2"}, "label", &backend);

    Sample row1 = dataset.get(1);
    ASSERT_EQ(row1.fields.size(), 2u);
    EXPECT_EQ(row1.fields[0].shape(), Shape({1, 2}));
    EXPECT_FLOAT_EQ(row1.fields[0].data()[0], 3.0f);
    EXPECT_FLOAT_EQ(row1.fields[0].data()[1], 4.0f);
    EXPECT_EQ(row1.fields[1].shape(), Shape({1}));
    EXPECT_FLOAT_EQ(row1.fields[1].data()[0], 1.0f);
}

TEST_F(CsvDatasetTest, FeatureColumnOrderMatchesRequestedOrder) {
    CsvDataset dataset(FixturePath("simple.csv"), {"x2", "x1"}, "label", &backend);
    Sample row0 = dataset.get(0);
    EXPECT_FLOAT_EQ(row0.fields[0].data()[0], 2.0f);
    EXPECT_FLOAT_EQ(row0.fields[0].data()[1], 1.0f);
}

TEST_F(CsvDatasetTest, GetThrowsOnOutOfRangeIndex) {
    CsvDataset dataset(FixturePath("simple.csv"), {"x1", "x2"}, "label", &backend);
    EXPECT_THROW(dataset.get(3), std::out_of_range);
    EXPECT_THROW(dataset.get(-1), std::out_of_range);
}

TEST_F(CsvDatasetTest, ConstructorThrowsOnMissingFeatureColumn) {
    EXPECT_THROW(CsvDataset(FixturePath("simple.csv"), {"does_not_exist"}, "label", &backend), std::runtime_error);
}

TEST_F(CsvDatasetTest, ConstructorThrowsOnMissingLabelColumn) {
    EXPECT_THROW(CsvDataset(FixturePath("simple.csv"), {"x1"}, "does_not_exist", &backend), std::runtime_error);
}

TEST_F(CsvDatasetTest, GetThrowsOnUnparsableFloatCell) {
    CsvDataset dataset(FixturePath("quoted.csv"), {"name"}, "value", &backend);
    EXPECT_THROW(dataset.get(0), std::runtime_error);
}

}  // namespace
}  // namespace pulsatrix
