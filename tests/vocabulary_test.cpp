#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "pulsatrix/vocabulary.hpp"

namespace pulsatrix {
namespace {

TEST(VocabularyTest, UnkTokenIsReservedAtIndexZero) {
    Vocabulary vocab(std::vector<std::string>{"the", "cat"});
    EXPECT_EQ(vocab.TokenAt(0), "<unk>");
    EXPECT_EQ(vocab.IndexOf("<unk>"), Vocabulary::kUnkIndex);
}

TEST(VocabularyTest, IndexOfReturnsUnkForUnknownToken) {
    Vocabulary vocab(std::vector<std::string>{"the", "cat"});
    EXPECT_EQ(vocab.IndexOf("dog"), Vocabulary::kUnkIndex);
}

TEST(VocabularyTest, IndexOfAndTokenAtRoundTrip) {
    Vocabulary vocab(std::vector<std::string>{"the", "cat"});
    EXPECT_EQ(vocab.TokenAt(1), "the");
    EXPECT_EQ(vocab.IndexOf("the"), 1);
    EXPECT_EQ(vocab.TokenAt(2), "cat");
    EXPECT_EQ(vocab.IndexOf("cat"), 2);
}

TEST(VocabularyTest, SizeIncludesUnkToken) {
    Vocabulary vocab(std::vector<std::string>{"the", "cat"});
    EXPECT_EQ(vocab.size(), 3);
}

TEST(BuildVocabularyTest, RanksByDescendingFrequency) {
    std::vector<std::vector<std::string>> corpus = {
        {"the", "cat", "sat"},
        {"the", "cat"},
        {"the"},
    };
    Vocabulary vocab = BuildVocabulary(corpus);
    EXPECT_EQ(vocab.TokenAt(1), "the");
    EXPECT_EQ(vocab.TokenAt(2), "cat");
    EXPECT_EQ(vocab.TokenAt(3), "sat");
}

TEST(BuildVocabularyTest, TiesBrokenByFirstSeenOrder) {
    std::vector<std::vector<std::string>> corpus = {
        {"zebra", "apple"},  // both frequency 1; "zebra" seen first
    };
    Vocabulary vocab = BuildVocabulary(corpus);
    EXPECT_EQ(vocab.TokenAt(1), "zebra");
    EXPECT_EQ(vocab.TokenAt(2), "apple");
}

TEST(BuildVocabularyTest, TruncatesToMaxVocabSize) {
    std::vector<std::vector<std::string>> corpus = {
        {"the", "cat", "sat", "on", "mat"},
        {"the", "cat"},
        {"the"},
    };
    Vocabulary vocab = BuildVocabulary(corpus, /*max_vocab_size=*/2);
    EXPECT_EQ(vocab.size(), 3);  // <unk> + top 2
    EXPECT_EQ(vocab.TokenAt(1), "the");
    EXPECT_EQ(vocab.TokenAt(2), "cat");
}

TEST(BuildVocabularyTest, EmptyCorpusProducesUnkOnlyVocabulary) {
    Vocabulary vocab = BuildVocabulary({});
    EXPECT_EQ(vocab.size(), 1);
}

}  // namespace
}  // namespace pulsatrix
