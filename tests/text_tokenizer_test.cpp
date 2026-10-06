// TOK-1: the tokenizer interface, its pipeline, and the word-level, byte and character tokenizers.

#include "pulsatrix/text_tokenizer.hpp"

#include <gtest/gtest.h>

#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/tokenizer.hpp"
#include "pulsatrix/tokenizer_components.hpp"
#include "pulsatrix/vocabulary.hpp"

namespace pulsatrix {
namespace {

std::string Slice(const std::string& text, Offset o) { return text.substr(o.begin, o.end - o.begin); }

// The word-level tokenizer reproduces Tokenizer::Tokenize and Vocabulary::IndexOf on ASCII text.
TEST(WordLevelTokenizerTest, MatchesTheLegacyTokenizerOnAscii) {
    const std::vector<std::string> texts = {"Hello, world!", "Don't stop.", "  multiple   spaces  ", "",
                                            "Wait... really?!", "MiXeD case 123 and\ttabs\nnewlines"};
    std::vector<std::vector<std::string>> corpus;
    for (const std::string& t : texts) corpus.push_back(Tokenizer::Tokenize(t));
    corpus.pop_back();  // leave the last text's words out of the vocabulary: they map to <unk>
    const Vocabulary vocab = BuildVocabulary(corpus);
    const TextTokenizer tok = MakeWordLevelTokenizer(vocab);
    for (const std::string& t : texts) {
        const Encoding e = tok.encode(t);
        EXPECT_EQ(e.tokens.size(), Tokenizer::Tokenize(t).size()) << t;
        const std::vector<std::string> legacy = Tokenizer::Tokenize(t);
        for (size_t i = 0; i < e.size(); ++i) {
            EXPECT_EQ(e.ids[i], vocab.IndexOf(legacy[i])) << t;
            EXPECT_EQ(e.tokens[i], vocab.TokenAt(e.ids[i]));
            // Offsets point at the original (not lowercased) text.
            std::string source = Slice(t, e.offsets[i]);
            for (char& c : source) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            EXPECT_EQ(source, legacy[i]) << t;
            EXPECT_EQ(e.special_tokens_mask[i], 0);
        }
    }
    EXPECT_EQ(tok.decode(tok.encode("Hello, world!").ids), "hello , world !");  // no decoder: spaces
}

TEST(WordLevelTokenizerTest, KeepsNonAsciiCharactersWhole) {
    const TextTokenizer tok = MakeWordLevelTokenizer(BuildVocabulary({{"caf"}}));
    const std::string text = "Café ok";
    const Encoding e = tok.encode(text);
    ASSERT_EQ(e.size(), 3u);
    EXPECT_EQ(Slice(text, e.offsets[0]), "Caf");
    EXPECT_EQ(Slice(text, e.offsets[1]), "é");  // one token, both bytes
    EXPECT_EQ(e.ids[1], Vocabulary::kUnkIndex);
    EXPECT_EQ(Slice(text, e.offsets[2]), "ok");
}

TEST(ByteTokenizerTest, BytesAreIdsAndSpecialTokensSplitOutFirst) {
    const TextTokenizer tok = MakeByteTokenizer({"<eos>", "<pad>"});
    EXPECT_EQ(tok.vocab_size(), 258);
    const std::string text = "hé<eos>";
    const Encoding e = tok.encode(text);
    EXPECT_EQ(e.ids, (std::vector<int64_t>{'h', 0xC3, 0xA9, 256}));
    EXPECT_EQ(e.offsets[1], (Offset{1, 2}));
    EXPECT_EQ(e.offsets[3], (Offset{3, 8}));
    EXPECT_EQ(e.special_tokens_mask, (std::vector<uint8_t>{0, 0, 0, 1}));
    EXPECT_EQ(tok.decode(e.ids), text);
    EXPECT_EQ(tok.decode(e.ids, /*skip_special_tokens=*/true), "hé");
    EXPECT_EQ(tok.token_to_id("<pad>"), 257);
    EXPECT_EQ(tok.id_to_token(65), "A");
    EXPECT_THROW((void)tok.decode({300}), std::out_of_range);
}

TEST(CharTokenizerTest, CharactersWithAnUnknownToken) {
    const TextTokenizer tok = MakeCharTokenizer({"a", "b", "é"}, {"<bos>"});
    EXPECT_EQ(tok.token_to_id("<unk>"), 3);
    EXPECT_EQ(tok.token_to_id("<bos>"), 4);
    const Encoding e = tok.encode("<bos>abéz");
    EXPECT_EQ(e.ids, (std::vector<int64_t>{4, 0, 1, 2, 3}));
    EXPECT_EQ(e.offsets[3], (Offset{7, 9}));
    EXPECT_EQ(tok.decode({0, 2, 1}), "aéb");
    EXPECT_THROW((void)MakeCharTokenizer({"ab"}), std::invalid_argument);  // not one character
    EXPECT_THROW((void)MakeCharTokenizer({"a", "a"}), std::invalid_argument);
}

TEST(TextTokenizerTest, AddedTokensMatchLongestFirst) {
    TextTokenizer tok = MakeByteTokenizer();
    tok.add_token({"<s>", 300, true}).add_token({"<s>x", 301, false});
    const Encoding e = tok.encode("<s>x<s>");
    EXPECT_EQ(e.ids, (std::vector<int64_t>{301, 300}));
    EXPECT_EQ(e.special_tokens_mask, (std::vector<uint8_t>{0, 1}));
    EXPECT_EQ(tok.decode(e.ids, true), "<s>x");  // only special tokens are skipped
    EXPECT_THROW(tok.add_token({"<s>", 302}), std::invalid_argument);
    EXPECT_THROW(tok.add_token({"<t>", 300}), std::invalid_argument);
    EXPECT_THROW(tok.add_token({"", 303}), std::invalid_argument);
}

TEST(TextTokenizerTest, RefusesInvalidUtf8) {
    const TextTokenizer tok = MakeByteTokenizer();
    EXPECT_THROW((void)tok.encode(std::string("a\xC3")), std::invalid_argument);       // truncated
    EXPECT_THROW((void)tok.encode(std::string("\xED\xA0\x80")), std::invalid_argument);  // surrogate
    EXPECT_THROW((void)tok.encode(std::string("\xC0\xAF")), std::invalid_argument);      // overlong
}

class BosEos : public PostProcessor {
public:
    Encoding process(Encoding e, bool add_special_tokens) const override {
        if (!add_special_tokens) return e;
        Encoding out;
        out.push_back(1000, "<bos>", {0, 0}, true);
        for (size_t i = 0; i < e.size(); ++i) out.push_back(e.ids[i], e.tokens[i], e.offsets[i], e.special_tokens_mask[i] != 0);
        out.push_back(1001, "<eos>", {0, 0}, true);
        return out;
    }
};

TEST(TextTokenizerTest, PostProcessorRunsLastAndCanBeSkipped) {
    TextTokenizer tok = MakeByteTokenizer();
    tok.set_post_processor(std::make_shared<BosEos>());
    const Encoding e = tok.encode("hi");
    EXPECT_EQ(e.ids, (std::vector<int64_t>{1000, 'h', 'i', 1001}));
    EXPECT_EQ(e.special_tokens_mask, (std::vector<uint8_t>{1, 0, 0, 1}));
    EXPECT_EQ(tok.encode("hi", /*add_special_tokens=*/false).ids, (std::vector<int64_t>{'h', 'i'}));
}

TEST(NormalizedStringTest, RebuildKeepsWhereEachByteCameFrom) {
    // "e" + combining acute (U+0301) composed into "é", as NFC would.
    NormalizedString s("xe\xCC\x81y", 10);
    s.rebuild({{"x", 0, 1}, {"\xC3\xA9", 1, 4}, {">", 4, 4}, {"y", 4, 5}});
    EXPECT_EQ(s.text(), "x\xC3\xA9>y");
    EXPECT_EQ(s.original(1, 3), (Offset{11, 14}));  // the composed character covers both source characters
    EXPECT_EQ(s.original(3, 4), (Offset{14, 14}));  // an inserted byte has an empty span
    EXPECT_EQ(s.original(4, 5), (Offset{14, 15}));
    EXPECT_EQ(s.slice(1, 3).original(0, 2), (Offset{11, 14}));
    EXPECT_EQ(s.original(2, 2), (Offset{11, 11}));
    EXPECT_THROW((void)s.original(3, 9), std::out_of_range);
}

}  // namespace
}  // namespace pulsatrix
