#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "pulsatrix/tokenizer.hpp"

namespace pulsatrix {
namespace {

TEST(TokenizerTest, SplitsWordsAndPunctuation) {
    auto tokens = Tokenizer::Tokenize("Hello, world!");
    EXPECT_EQ(tokens, (std::vector<std::string>{"hello", ",", "world", "!"}));
}

TEST(TokenizerTest, KeepsApostropheInsideWord) {
    auto tokens = Tokenizer::Tokenize("Don't stop.");
    EXPECT_EQ(tokens, (std::vector<std::string>{"don't", "stop", "."}));
}

TEST(TokenizerTest, CollapsesRepeatedWhitespace) {
    auto tokens = Tokenizer::Tokenize("  multiple   spaces  ");
    EXPECT_EQ(tokens, (std::vector<std::string>{"multiple", "spaces"}));
}

TEST(TokenizerTest, EmptyStringProducesNoTokens) {
    EXPECT_TRUE(Tokenizer::Tokenize("").empty());
}

TEST(TokenizerTest, WhitespaceOnlyStringProducesNoTokens) {
    EXPECT_TRUE(Tokenizer::Tokenize("   \t\n  ").empty());
}

TEST(TokenizerTest, MultiplePunctuationMarksEachBecomeTheirOwnToken) {
    auto tokens = Tokenizer::Tokenize("Wait... really?!");
    EXPECT_EQ(tokens, (std::vector<std::string>{"wait", ".", ".", ".", "really", "?", "!"}));
}

}  // namespace
}  // namespace pulsatrix
