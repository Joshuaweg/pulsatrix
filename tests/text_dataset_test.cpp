#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/text_dataset.hpp"

namespace pulsatrix {
namespace {

std::string FixturePath(const std::string& name) {
    return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/" + name;
}

class TextDatasetTest : public ::testing::Test {
protected:
    // Vocabulary: <unk>=0, the=1, cat=2, sat=3, dog=4, ran=5. "fox"/"jumped" are
    // deliberately absent, to exercise the <unk> fallback.
    TextDatasetTest() : vocabulary({"the", "cat", "sat", "dog", "ran"}) {}

    CPUBackend backend;
    Vocabulary vocabulary;
};

TEST_F(TextDatasetTest, SizeMatchesLineCount) {
    TextDataset dataset(FixturePath("text_corpus.txt"), &vocabulary, &backend);
    EXPECT_EQ(dataset.size(), 4);
}

TEST_F(TextDatasetTest, GetTokenizesAndIndexesLine) {
    TextDataset dataset(FixturePath("text_corpus.txt"), &vocabulary, &backend);

    Sample line0 = dataset.get(0);
    ASSERT_EQ(line0.fields.size(), 1u);
    EXPECT_EQ(line0.fields[0].shape(), Shape({1, 3}));
    EXPECT_FLOAT_EQ(line0.fields[0].data()[0], 1.0f);  // the
    EXPECT_FLOAT_EQ(line0.fields[0].data()[1], 2.0f);  // cat
    EXPECT_FLOAT_EQ(line0.fields[0].data()[2], 3.0f);  // sat

    Sample line1 = dataset.get(1);
    EXPECT_FLOAT_EQ(line1.fields[0].data()[0], 1.0f);  // the
    EXPECT_FLOAT_EQ(line1.fields[0].data()[1], 4.0f);  // dog
    EXPECT_FLOAT_EQ(line1.fields[0].data()[2], 5.0f);  // ran
}

TEST_F(TextDatasetTest, EmptyLineProducesZeroLengthTensor) {
    TextDataset dataset(FixturePath("text_corpus.txt"), &vocabulary, &backend);
    Sample empty_line = dataset.get(2);
    EXPECT_EQ(empty_line.fields[0].shape(), Shape({1, 0}));
    EXPECT_EQ(empty_line.fields[0].numel(), 0);
}

TEST_F(TextDatasetTest, OutOfVocabularyTokensResolveToUnk) {
    TextDataset dataset(FixturePath("text_corpus.txt"), &vocabulary, &backend);
    Sample line3 = dataset.get(3);  // "the fox jumped" -- fox/jumped are OOV
    EXPECT_FLOAT_EQ(line3.fields[0].data()[0], 1.0f);                         // the
    EXPECT_FLOAT_EQ(line3.fields[0].data()[1], Vocabulary::kUnkIndex);  // fox -> <unk>
    EXPECT_FLOAT_EQ(line3.fields[0].data()[2], Vocabulary::kUnkIndex);  // jumped -> <unk>
}

TEST_F(TextDatasetTest, GetThrowsOnOutOfRangeIndex) {
    TextDataset dataset(FixturePath("text_corpus.txt"), &vocabulary, &backend);
    EXPECT_THROW(dataset.get(4), std::out_of_range);
    EXPECT_THROW(dataset.get(-1), std::out_of_range);
}

TEST_F(TextDatasetTest, ConstructorThrowsOnMissingFile) {
    EXPECT_THROW(TextDataset(FixturePath("does_not_exist.txt"), &vocabulary, &backend), std::runtime_error);
}

}  // namespace
}  // namespace pulsatrix
