// VIZ-6a: token relevance documents from real explanations, and the token strip's word view.

#include "pulsatrix/viz/text_relevance.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/json.hpp"
#include "pulsatrix/tokenizer_components.hpp"
#include "pulsatrix/tokenizer_json.hpp"
#include "pulsatrix/viz/svg.hpp"

namespace pulsatrix {
namespace {

std::string Fixture(const std::string& path) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/" + path; }

std::string Joined(const TokenRelevanceDocument& doc, bool scored_only = false) {
    std::string s;
    for (size_t i = 0; i < doc.tokens.size(); ++i) {
        if (!scored_only || doc.is_scored(i)) s += doc.tokens[i];
    }
    return s;
}

size_t Count(const std::string& haystack, const std::string& needle) {
    size_t n = 0;
    for (size_t at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + 1)) ++n;
    return n;
}

// Byte-level tokens that split an emoji become one piece, decoded; BOS is shown or set aside.
TEST(TextRelevanceTest, TokenDocumentDecodesByteLevelTokens) {
    const TextTokenizer tok = LoadTokenizerJson(Fixture("tokenizers/llama3.2/tokenizer.json"));
    const std::string text = "Hi \xF0\x9F\x98\x80!";  // "Hi 😀!"
    const Encoding e = tok.encode(text);
    std::vector<float> scores(e.size());
    for (size_t i = 0; i < scores.size(); ++i) scores[i] = static_cast<float>(i + 1);

    const TokenRelevanceDocument with_bos = MakeTokenRelevanceDocument(text, e, scores, "attn_lrp", " x");
    EXPECT_EQ(with_bos.tokens[0], "<|begin_of_text|>");
    EXPECT_EQ(with_bos.relevance[0], 1.0f);
    EXPECT_EQ(Joined(with_bos), "<|begin_of_text|>" + text);
    // The emoji's byte tokens merged into one piece whose score is their sum.
    float emoji = 0.0f, total = 0.0f;
    for (size_t i = 0; i < with_bos.tokens.size(); ++i) {
        if (with_bos.tokens[i].find("\xF0\x9F\x98\x80") != std::string::npos) emoji = with_bos.relevance[i];
        total += with_bos.relevance[i];
    }
    EXPECT_GT(emoji, 0.0f);
    float all = 0.0f;
    for (float s : scores) all += s;
    EXPECT_FLOAT_EQ(total, all);  // nothing lost in merging
    EXPECT_FALSE(with_bos.unassigned.has_value());

    const TokenRelevanceDocument without = MakeTokenRelevanceDocument(text, e, scores, "attn_lrp", " x", false);
    EXPECT_EQ(Joined(without), text);
    ASSERT_TRUE(without.unassigned.has_value());
    EXPECT_FLOAT_EQ(*without.unassigned, 1.0f);
    EXPECT_THROW((void)MakeTokenRelevanceDocument(text, e, {1.0f}, "m"), std::invalid_argument);
}

// A tokenizer that skips whitespace leaves gaps: they become unscored context.
TEST(TextRelevanceTest, TextNoTokenCoversIsContext) {
    const TextTokenizer tok = MakeWordLevelTokenizer(BuildVocabulary({{"hello", "world"}}));
    const std::string text = "  Hello,  world ";
    const Encoding e = tok.encode(text);
    const TokenRelevanceDocument doc = MakeTokenRelevanceDocument(text, e, std::vector<float>(e.size(), 0.5f), "lrp");
    EXPECT_EQ(Joined(doc), text);
    EXPECT_EQ(Joined(doc, true), "Hello,world");
    EXPECT_EQ(doc.tokens.front(), "  ");
    EXPECT_FALSE(doc.is_scored(0));
}

// Every corpus line, both kinds of document: the pieces read as the text.
TEST(TextRelevanceTest, PiecesReadAsTheTextOnTheTokenizerCorpus) {
    const TextTokenizer tok = LoadTokenizerJson(Fixture("tokenizers/qwen2.5/tokenizer.json"));
    std::ifstream in(Fixture("tokenizers/qwen2.5/reference.jsonl"));
    std::mt19937 gen(3);
    std::normal_distribution<float> dist;
    size_t lines = 0;
    for (std::string raw; std::getline(in, raw); ++lines) {
        const std::string text = ParseJson(raw).find("text")->as_string();
        const Encoding e = tok.encode(text);
        std::vector<float> scores(e.size());
        for (float& s : scores) s = dist(gen);
        // Qwen's NFC can widen a composed character's span; the pieces still cover the text once.
        EXPECT_EQ(Joined(MakeTokenRelevanceDocument(text, e, scores, "m")), text);
        const WordScores words = AggregateToWords(text, e, scores);
        const TokenRelevanceDocument wd = MakeWordRelevanceDocument(text, words, "m");
        EXPECT_EQ(Joined(wd), text);
        EXPECT_EQ(wd.granularity, "word");
        (void)ParseTokenRelevanceDocument(ToJson(wd));  // valid
    }
    EXPECT_EQ(lines, 90u);
}

TEST(TextRelevanceTest, WordDocumentKeepsTheSpacesBetweenWords) {
    const std::string text = "The cat, sat.";
    WordScores w;
    w.words = {{{0, 3}, "The", 1.0f, {}}, {{4, 7}, "cat", 2.0f, {}}, {{7, 8}, ",", -0.5f, {}}, {{9, 12}, "sat", 0.0f, {}},
               {{12, 13}, ".", 0.25f, {}}};
    w.unassigned = 3.0f;
    const TokenRelevanceDocument doc = MakeWordRelevanceDocument(text, w, "attn_lrp", " on");
    EXPECT_EQ(doc.tokens, (std::vector<std::string>{"The", " ", "cat", ",", " ", "sat", "."}));
    EXPECT_EQ(doc.scored, (std::vector<bool>{true, false, true, true, false, true, true}));
    ASSERT_TRUE(doc.unassigned.has_value());
    EXPECT_FLOAT_EQ(*doc.unassigned, 3.0f);
    w.words[1].span = {1, 2};  // overlaps "The"
    EXPECT_THROW((void)MakeWordRelevanceDocument(text, w, "m"), std::invalid_argument);
}

TEST(TokenRelevanceDocumentTest, OptionalFieldsRoundTripAndOldDocumentsStayUnchanged) {
    // A document without the new fields writes exactly what it read.
    std::ifstream in(Fixture("viz/token_relevance.v1.json"));
    const std::string old((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const TokenRelevanceDocument parsed = ParseTokenRelevanceDocument(old);
    EXPECT_EQ(parsed.granularity, "token");
    EXPECT_TRUE(parsed.scored.empty());
    EXPECT_FALSE(parsed.unassigned.has_value());
    EXPECT_EQ(ToJson(parsed), old);

    TokenRelevanceDocument doc;
    doc.method = "attn_lrp";
    doc.tokens = {"a", " ", "b"};
    doc.relevance = {1.0f, 0.0f, -2.0f};
    doc.granularity = "word";
    doc.scored = {true, false, true};
    doc.unassigned = 0.5f;
    const TokenRelevanceDocument back = ParseTokenRelevanceDocument(ToJson(doc));
    EXPECT_EQ(back.granularity, "word");
    EXPECT_EQ(back.scored, doc.scored);
    EXPECT_FLOAT_EQ(*back.unassigned, 0.5f);

    doc.scored = {true};
    EXPECT_THROW((void)ToJson(doc), std::invalid_argument);
    doc.scored.clear();
    doc.granularity = "sentence";
    EXPECT_THROW((void)ToJson(doc), std::invalid_argument);
    EXPECT_THROW((void)ParseTokenRelevanceDocument(
                     R"({"schema": "pulsatrix.token_relevance.v1", "method": "m", "tokens": ["a"], "relevance": [1], "target": "", "scored": [true, false]})"),
                 std::invalid_argument);
}

TEST(TokenStripSvgTest, WordViewDrawsContextWithoutBoxes) {
    TokenRelevanceDocument doc;
    doc.method = "attn_lrp";
    doc.tokens = {"The", " ", "Eiffel", " ", "Tower"};
    doc.relevance = {-1.6f, 99.0f, 4.1f, 99.0f, -0.1f};  // context scores are ignored, not scaled to
    doc.granularity = "word";
    doc.scored = {true, false, true, false, true};
    doc.unassigned = 0.25f;
    const std::string svg = RenderTokenStripSvg(doc);
    EXPECT_EQ(Count(svg, "<rect class=\"token"), 3u);
    EXPECT_EQ(Count(svg, "class=\"context-text\""), 2u);
    EXPECT_NE(svg.find("Word relevance (attn_lrp)"), std::string::npos);
    EXPECT_NE(svg.find("relevance outside the pieces shown: 0.25"), std::string::npos);
    EXPECT_EQ(svg.find(": 99"), std::string::npos);  // context relevance never shown in a tooltip
}

// Widths follow display columns: a CJK character takes two, a combining accent none.
TEST(TokenStripSvgTest, WidthsFollowDisplayColumns) {
    TokenRelevanceDocument doc;
    doc.method = "m";
    doc.tokens = {"ab", "\xE4\xB8\xAD\xE6\x96\x87", "e\xCC\x81"};  // "ab", "中文", "e" + combining acute
    doc.relevance = {1.0f, 1.0f, 1.0f};
    const std::string svg = RenderTokenStripSvg(doc);
    std::vector<double> widths;
    for (size_t at = svg.find("<rect class=\"token"); at != std::string::npos; at = svg.find("<rect class=\"token", at + 1)) {
        const size_t w = svg.find("width=\"", at) + 7;
        widths.push_back(std::stod(svg.substr(w, svg.find('"', w) - w)));
    }
    ASSERT_EQ(widths.size(), 3u);
    EXPECT_DOUBLE_EQ(widths[1], 2.0 * widths[0]);  // two wide characters = four columns
    EXPECT_DOUBLE_EQ(widths[2], widths[0] / 2.0);  // "é" from two code points = one column
}

}  // namespace
}  // namespace pulsatrix
