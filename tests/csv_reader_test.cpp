#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "pulsatrix/csv_reader.hpp"

namespace pulsatrix {
namespace {

std::string FixturePath(const std::string& name) {
    return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/" + name;
}

TEST(CsvReaderTest, ParsesHeaderAndRows) {
    CsvTable table = CsvReader::Load(FixturePath("simple.csv"));
    EXPECT_EQ(table.header, (std::vector<std::string>{"x1", "x2", "label"}));
    ASSERT_EQ(table.rows.size(), 3u);
    EXPECT_EQ(table.rows[0], (std::vector<std::string>{"1.0", "2.0", "0.0"}));
    EXPECT_EQ(table.rows[2], (std::vector<std::string>{"5.0", "6.0", "0.0"}));
}

TEST(CsvReaderTest, ParsesQuotedFieldsWithEmbeddedCommaAndEscapedQuote) {
    CsvTable table = CsvReader::Load(FixturePath("quoted.csv"));
    EXPECT_EQ(table.header, (std::vector<std::string>{"name", "note", "value"}));
    ASSERT_EQ(table.rows.size(), 2u);
    EXPECT_EQ(table.rows[0][0], "Smith, John");
    EXPECT_EQ(table.rows[0][1], "says \"hi\"");
    EXPECT_EQ(table.rows[1][0], "Plain");
}

TEST(CsvReaderTest, HasHeaderFalseTreatsFirstRowAsData) {
    CsvTable table = CsvReader::Load(FixturePath("no_header.csv"), /*has_header=*/false);
    EXPECT_TRUE(table.header.empty());
    ASSERT_EQ(table.rows.size(), 2u);
    EXPECT_EQ(table.rows[0], (std::vector<std::string>{"1.0", "2.0"}));
    EXPECT_EQ(table.rows[1], (std::vector<std::string>{"3.0", "4.0"}));
}

TEST(CsvReaderTest, ThrowsOnFieldCountMismatch) {
    EXPECT_THROW(CsvReader::Load(FixturePath("malformed_row.csv")), std::runtime_error);
}

TEST(CsvReaderTest, ThrowsOnMissingFile) {
    EXPECT_THROW(CsvReader::Load(FixturePath("does_not_exist.csv")), std::runtime_error);
}

}  // namespace
}  // namespace pulsatrix
