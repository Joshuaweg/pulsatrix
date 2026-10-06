// TOK-4: per-word scores from per-token scores.

#include "pulsatrix/word_scores.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/json.hpp"
#include "pulsatrix/tokenizer_json.hpp"

namespace pulsatrix {
namespace {

std::vector<std::string> Texts(const WordScores& w) {
    std::vector<std::string> out;
    for (const WordScore& word : w.words) out.push_back(word.text);
    return out;
}

// "Hello, world!" as a byte-level tokenizer might cut it, after a BOS token.
const std::string kText = "Hello, world!";
const std::vector<Offset> kOffsets = {{0, 0}, {0, 5}, {5, 6}, {6, 10}, {10, 12}, {12, 13}};  // BOS Hello , " wor" ld !
const std::vector<float> kScores = {3.0f, 1.0f, 0.5f, 2.0f, 1.0f, -1.5f};

TEST(WordScoresTest, SumConservesTheTotal) {
    const WordScores w = AggregateToWords(kText, kOffsets, kScores, SplitWords(kText));
    EXPECT_EQ(Texts(w), (std::vector<std::string>{"Hello", ",", "world", "!"}));
    EXPECT_FLOAT_EQ(w.words[0].score, 1.0f);
    EXPECT_FLOAT_EQ(w.words[1].score, 0.5f);
    EXPECT_FLOAT_EQ(w.words[2].score, 3.0f);  // " wor" + "ld": the space is in no word, so all of " wor" counts
    EXPECT_FLOAT_EQ(w.words[3].score, -1.5f);
    EXPECT_FLOAT_EQ(w.unassigned, 3.0f);  // the BOS token
    EXPECT_EQ(w.words[2].tokens, (std::vector<size_t>{3, 4}));
    EXPECT_EQ(w.words[2].span, (Offset{7, 12}));
}

TEST(WordScoresTest, MeanMaxAndMaxAbs) {
    const std::vector<Offset> words = SplitWords(kText);
    EXPECT_FLOAT_EQ(AggregateToWords(kText, kOffsets, kScores, words, WordAggregation::Mean).words[2].score, 1.5f);
    EXPECT_FLOAT_EQ(AggregateToWords(kText, kOffsets, kScores, words, WordAggregation::Max).words[2].score, 2.0f);
    // A word of tokens 0.5 and -2: Max keeps the positive one, MaxAbs the strong negative one.
    const std::string t = "ab";
    const std::vector<Offset> o = {{0, 1}, {1, 2}};
    const std::vector<float> s = {0.5f, -2.0f};
    EXPECT_FLOAT_EQ(AggregateToWords(t, o, s, SplitWords(t), WordAggregation::Max).words[0].score, 0.5f);
    EXPECT_FLOAT_EQ(AggregateToWords(t, o, s, SplitWords(t), WordAggregation::MaxAbs).words[0].score, -2.0f);
}

TEST(WordScoresTest, ATokenAcrossWordsSplitsItsScoreByBytes) {
    // "b c" covers one byte of "ab" and one of "cde" (the space counts for neither): half each.
    const std::string t = "ab cde";
    const std::vector<Offset> o = {{0, 1}, {1, 4}, {4, 6}};
    const std::vector<float> s = {1.0f, 3.0f, 2.0f};
    const WordScores sum = AggregateToWords(t, o, s, SplitWords(t));
    EXPECT_FLOAT_EQ(sum.words[0].score, 2.5f);
    EXPECT_FLOAT_EQ(sum.words[1].score, 3.5f);
    EXPECT_EQ(sum.words[0].tokens, (std::vector<size_t>{0, 1}));
    EXPECT_EQ(sum.words[1].tokens, (std::vector<size_t>{1, 2}));
    // Mean weights by share: (1 * 1 + 0.5 * 3) / 1.5.
    EXPECT_FLOAT_EQ(AggregateToWords(t, o, s, SplitWords(t), WordAggregation::Mean).words[0].score, 2.5f / 1.5f);
    // Max uses whole scores, not shares.
    EXPECT_FLOAT_EQ(AggregateToWords(t, o, s, SplitWords(t), WordAggregation::Max).words[0].score, 3.0f);
}

TEST(WordScoresTest, WordSplits) {
    const std::string t = "Don't stop—now!  x_1 \t\xC3\xA9t\xC3\xA9";
    EXPECT_EQ(Texts(AggregateToWords(t, {}, {}, SplitWords(t))),
              (std::vector<std::string>{"Don", "'", "t", "stop", "—", "now", "!", "x_1", "\xC3\xA9t\xC3\xA9"}));
    EXPECT_EQ(Texts(AggregateToWords(t, {}, {}, SplitWords(t, WordSplit::Whitespace))),
              (std::vector<std::string>{"Don't", "stop—now!", "x_1", "\xC3\xA9t\xC3\xA9"}));
    EXPECT_EQ(Texts(AggregateToWords(t, {}, {}, SplitWords(t, UnicodeRegex("\\p{Lu}\\p{Ll}*")))),
              (std::vector<std::string>{"Don"}));
}

TEST(WordScoresTest, AWordNoTokenCoversScoresZero) {
    const std::string t = "ab cd";
    const WordScores w = AggregateToWords(t, {{0, 2}}, {4.0f}, SplitWords(t), WordAggregation::MaxAbs);
    EXPECT_FLOAT_EQ(w.words[1].score, 0.0f);
    EXPECT_TRUE(w.words[1].tokens.empty());
}

TEST(WordScoresTest, RefusesMismatchedInput) {
    const std::vector<Offset> words = SplitWords(kText);
    EXPECT_THROW((void)AggregateToWords(kText, kOffsets, {1.0f}, words), std::invalid_argument);
    EXPECT_THROW((void)AggregateToWords(kText, {{0, 99}}, {1.0f}, words), std::invalid_argument);
    EXPECT_THROW((void)AggregateToWords(kText, {}, {}, {{0, 5}, {3, 8}}), std::invalid_argument);  // overlapping
    EXPECT_THROW((void)AggregateToWords(kText, {}, {}, {{7, 9}, {0, 5}}), std::invalid_argument);  // out of order
}

// With a real pipeline (Llama 3.2's, tiny vocabulary) on every CI corpus line: a BOS token,
// byte-level tokens, partial characters and special tokens. Sum conserves the total, up to float
// rounding, under both word splits.
TEST(WordScoresTest, ConservesRelevanceOnTheTokenizerCorpus) {
    const std::string dir = std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/tokenizers/llama3.2";
    const TextTokenizer tok = LoadTokenizerJson(dir + "/tokenizer.json");
    std::ifstream in(dir + "/reference.jsonl");
    std::mt19937 gen(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    size_t lines = 0;
    for (std::string raw; std::getline(in, raw);) {
        const std::string text = ParseJson(raw).find("text")->as_string();
        const Encoding e = tok.encode(text);
        std::vector<float> scores(e.size());
        double total = 0.0;
        for (float& s : scores) total += (s = dist(gen));
        for (WordSplit split : {WordSplit::WordsAndPunctuation, WordSplit::Whitespace}) {
            const WordScores w = AggregateToWords(text, e, scores, WordAggregation::Sum, split);
            double sum = w.unassigned;
            for (const WordScore& word : w.words) sum += word.score;
            EXPECT_NEAR(sum, total, 1e-4 * (1.0 + std::abs(total))) << text;
        }
        if (e.size() > 0) {
            EXPECT_EQ(e.offsets[0], (Offset{0, 0}));  // BOS: never in a word
        }
        ++lines;
    }
    EXPECT_EQ(lines, 90u);
}

}  // namespace
}  // namespace pulsatrix
