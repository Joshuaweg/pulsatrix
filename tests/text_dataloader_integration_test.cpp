#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/data_loader.hpp"
#include "pulsatrix/text_collate.hpp"
#include "pulsatrix/text_dataset.hpp"

namespace pulsatrix {
namespace {

std::string FixturePath(const std::string& name) {
    return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/" + name;
}

class TextDataLoaderIntegrationTest : public ::testing::Test {
protected:
    // <unk>=0, the=1, cat=2, sat=3, dog=4, ran=5 -- same fixture/vocab as text_dataset_test.cpp.
    TextDataLoaderIntegrationTest() : vocabulary({"the", "cat", "sat", "dog", "ran"}) {}

    CPUBackend backend;
    Vocabulary vocabulary;
};

TEST_F(TextDataLoaderIntegrationTest, ProducesCorrectlyPaddedBatchesAcrossVaryingLengthLines) {
    auto dataset = std::make_shared<TextDataset>(FixturePath("text_corpus.txt"), &vocabulary, &backend);
    DataLoaderOptions options;
    options.batch_size = 4;  // whole corpus (4 lines: 3, 3, 0, 3 tokens) in one batch
    options.collate_fn = PadCollate();
    DataLoader loader(dataset, &backend, options);

    auto batch = loader.next_batch();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->size(), 4);
    // Max length across the corpus is 3 (line 2, the empty line, has length 0).
    EXPECT_EQ(batch->fields[0].shape(), Shape({4, 3}));
    ASSERT_EQ(batch->fields[1].shape(), Shape({4}));

    EXPECT_FLOAT_EQ(batch->fields[1].data()[0], 3.0f);  // "the cat sat"
    EXPECT_FLOAT_EQ(batch->fields[1].data()[1], 3.0f);  // "the dog ran"
    EXPECT_FLOAT_EQ(batch->fields[1].data()[2], 0.0f);  // empty line
    EXPECT_FLOAT_EQ(batch->fields[1].data()[3], 3.0f);  // "the fox jumped" (fox/jumped -> unk)

    // Row 2 (the empty line) is entirely padding.
    EXPECT_FLOAT_EQ(batch->fields[0].at({2, 0}), 0.0f);
    EXPECT_FLOAT_EQ(batch->fields[0].at({2, 1}), 0.0f);
    EXPECT_FLOAT_EQ(batch->fields[0].at({2, 2}), 0.0f);

    EXPECT_FALSE(loader.next_batch().has_value());
}

}  // namespace
}  // namespace pulsatrix
