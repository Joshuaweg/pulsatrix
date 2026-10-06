// TOK-2: byte-level BPE from tokenizer.json. In CI it runs the tiny tokenizers in
// tests/fixtures/tokenizers: each target model's real pipeline (normalizer, pre-tokenizer,
// post-processor, decoder, added tokens, BPE options, merge format) with a small trained
// vocabulary, against Hugging Face tokenizers on the CI corpus (tools/tokenizers/). With
// PULSATRIX_TOKENIZER_DIR set to a directory of NAME/tokenizer.json + NAME/reference.jsonl (the
// real tokenizers on the 10,000-line corpus), it checks every one of those too.

#include "pulsatrix/tokenizer_json.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/byte_level_bpe.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/tokenizer_parity.hpp"
#include "pulsatrix/unicode_regex.hpp"

namespace pulsatrix {
namespace {

std::string Fixture(const std::string& name) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/tokenizers/" + name; }

std::string Describe(const TokenizerParityReport& r) {
    std::string s = std::to_string(r.id_mismatches) + " id, " + std::to_string(r.offset_mismatches) + " offset, " +
                    std::to_string(r.decode_mismatches) + " decode mismatches";
    for (const auto& m : r.examples) s += "\n  line " + std::to_string(m.line) + " " + m.field + ": " + m.detail;
    return s;
}

class TinyPipelineTest : public ::testing::TestWithParam<const char*> {};

TEST_P(TinyPipelineTest, MatchesHuggingFaceOnTheCiCorpus) {
    const TextTokenizer tok = LoadTokenizerJson(Fixture(GetParam()) + "/tokenizer.json");
    const TokenizerParityReport r = CompareToTokenizerReference(tok, Fixture(GetParam()) + "/reference.jsonl");
    EXPECT_TRUE(r.passed()) << Describe(r);
    EXPECT_EQ(r.lines, 90u);
}

// SmolLM2 (digits, GPT-2 regex), Qwen2.5 and Qwen3 (NFC, Qwen regex, string and pair merges),
// Llama 3.2 (Llama 3 regex, ignore_merges, BOS template) and gpt-oss (o200k regex, letter case).
INSTANTIATE_TEST_SUITE_P(Pipelines, TinyPipelineTest, ::testing::Values("smollm2", "qwen2.5", "qwen3", "llama3.2", "gpt-oss"),
                         [](const auto& info) {
                             std::string n = info.param;
                             for (char& c : n) c = (c == '.' || c == '-') ? '_' : c;
                             return n;
                         });

TEST(TokenizerJsonTest, LlamaAddsItsBeginningOfTextToken) {
    const TextTokenizer tok = LoadTokenizerJson(Fixture("llama3.2") + "/tokenizer.json");
    const Encoding e = tok.encode("Hi");
    ASSERT_GE(e.size(), 2u);
    EXPECT_EQ(e.tokens[0], "<|begin_of_text|>");
    EXPECT_EQ(e.special_tokens_mask[0], 1);
    EXPECT_EQ(e.offsets[0], (Offset{0, 0}));
    EXPECT_NE(tok.encode("Hi", false).tokens[0], "<|begin_of_text|>");
    EXPECT_EQ(tok.decode(e.ids, /*skip_special_tokens=*/true), "Hi");
}

TEST(TokenizerJsonTest, ByteLevelTokensInsideACharacterCoverJustTheirBytes) {
    const TextTokenizer tok = LoadTokenizerJson(Fixture("gpt-oss") + "/tokenizer.json");
    const std::string text = "\xF0\x9F\x98\x80";  // U+1F600, four bytes, no merge for it in the tiny vocabulary
    const Encoding e = tok.encode(text);
    ASSERT_GE(e.size(), 2u);
    size_t covered = 0;
    for (const Offset& o : e.offsets) {
        EXPECT_EQ(o.begin, covered);
        covered = o.end;
    }
    EXPECT_EQ(covered, 4u);
    EXPECT_EQ(tok.decode(e.ids), text);
}

TEST(TokenizerJsonTest, RefusesWhatItCantRunByName) {
    auto with = [](const std::string& normalizer, const std::string& model_extra, const std::string& added) {
        return R"({"normalizer": )" + normalizer + R"(, "pre_tokenizer": null, "post_processor": null, "decoder": null,
                   "added_tokens": [)" + added + R"(],
                   "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"])" + model_extra + "}}";
    };
    EXPECT_NO_THROW((void)ParseTokenizerJson(with("null", "", "")));
    auto refuses = [&](const std::string& json, const std::string& mention) {
        try {
            (void)ParseTokenizerJson(json);
            ADD_FAILURE() << "accepted: " << mention;
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find(mention), std::string::npos) << e.what();
        }
    };
    refuses(with(R"({"type": "Lowercase"})", "", ""), "Lowercase");
    refuses(with("null", R"(, "byte_fallback": true)", ""), "byte_fallback");
    refuses(with("null", R"(, "dropout": 0.1)", ""), "dropout");
    refuses(with("null", "", R"({"id": 3, "content": "<s>", "special": true, "lstrip": true})"), "lstrip");
    refuses(with(R"({"type": "NFC"})", "", R"({"id": 3, "content": "<s>", "special": false, "normalized": true})"),
            "normalization");
    refuses(R"({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": ["a c"]}})", "isn't in the vocabulary");
}

// ---- BPE -------------------------------------------------------------------------------

TEST(BpeModelTest, MergesLowestRankFirstAndLeftmostOnTies) {
    BpeModel bpe({{"a", 0}, {"b", 1}, {"ab", 2}, {"ba", 3}, {"aba", 4}}, {{"b", "a"}, {"a", "b"}, {"ab", "a"}});
    std::vector<int64_t> ids;
    for (const ModelToken& t : bpe.tokenize("abab")) ids.push_back(t.id);
    // "b a" (rank 0) first: a [ba] b; then nothing joins "a" + "ba" or "ba" + "b".
    EXPECT_EQ(ids, (std::vector<int64_t>{0, 3, 1}));
    ids.clear();
    for (const ModelToken& t : bpe.tokenize("aaa")) ids.push_back(t.id);
    EXPECT_EQ(ids, (std::vector<int64_t>{0, 0, 0}));
}

TEST(BpeModelTest, IgnoreMergesTakesAWholeVocabularyEntry) {
    const std::unordered_map<std::string, int64_t> vocab = {{"a", 0}, {"b", 1}, {"c", 2}, {"abc", 3}};
    EXPECT_EQ(BpeModel(vocab, {}).tokenize("abc").size(), 3u);
    BpeOptions options;
    options.ignore_merges = true;
    const std::vector<ModelToken> whole = BpeModel(vocab, {}, options).tokenize("abc");
    ASSERT_EQ(whole.size(), 1u);
    EXPECT_EQ(whole[0].id, 3);
    EXPECT_EQ(whole[0].end, 3u);
}

TEST(BpeModelTest, DropsCharactersOutsideTheVocabularyWithoutAnUnknownToken) {
    const std::vector<ModelToken> t = BpeModel({{"a", 0}, {"<unk>", 1}}, {}).tokenize("axa");
    ASSERT_EQ(t.size(), 2u);
    EXPECT_EQ(t[1].begin, 2u);  // the second "a" keeps its own offset
    BpeOptions options;
    options.unk_token = "<unk>";
    EXPECT_EQ(BpeModel({{"a", 0}, {"<unk>", 1}}, {}, options).tokenize("axa")[1].id, 1);
}

TEST(BpeModelTest, ARepeatedMergeTakesItsLastRank) {
    // ("b","c") appears at ranks 0 and 2; Hugging Face keeps rank 2, so ("a","b") at rank 1 wins.
    BpeModel bpe({{"a", 0}, {"b", 1}, {"c", 2}, {"ab", 3}, {"bc", 4}}, {{"b", "c"}, {"a", "b"}, {"b", "c"}});
    EXPECT_EQ(bpe.tokenize("abc")[0].id, 3);
}

TEST(ByteLevelTest, DecoderReplacesInvalidUtf8LikeRust) {
    const std::vector<std::string>& alphabet = ByteLevelAlphabet();
    ASSERT_EQ(alphabet.size(), 256u);
    EXPECT_EQ(alphabet[' '], "\xC4\xA0");  // 'Ġ'
    EXPECT_EQ(alphabet['A'], "A");
    // 0xE2 0x82 is a truncated three-byte sequence: one U+FFFD for both bytes, then "x".
    const std::string truncated = alphabet[0xE2] + alphabet[0x82] + "x";
    EXPECT_EQ(ByteLevelDecoder().decode({truncated}), "\xEF\xBF\xBDx");
    EXPECT_EQ(ByteLevelDecoder().decode({alphabet[0xFF], alphabet[0xC3] + alphabet[0xA9]}), "\xEF\xBF\xBD\xC3\xA9");
}

// ---- NFC -------------------------------------------------------------------------------

TEST(NfcNormalizerTest, ComposesReordersAndKeepsEachCharactersSource) {
    NfcNormalizer nfc;
    auto normalize = [&](const std::string& s) {
        NormalizedString n(s);
        nfc.normalize(n);
        return n;
    };
    EXPECT_EQ(normalize("e\xCC\x81").text(), "\xC3\xA9");                      // e + acute -> é
    EXPECT_EQ(normalize("\xE2\x84\xAA").text(), "K");                          // Kelvin sign -> K (singleton)
    EXPECT_EQ(normalize("\xE1\x84\x92\xE1\x85\xA1\xE1\x86\xAB").text(), "\xED\x95\x9C");  // jamo -> 한
    EXPECT_EQ(normalize("\xE0\xA4\x95\xE0\xA4\xBC").text(), "\xE0\xA4\x95\xE0\xA4\xBC");  // क़ is excluded from composition
    EXPECT_EQ(normalize("\xEF\xA4\x80").text(), "\xE8\xB1\x88");               // CJK compatibility U+F900 -> U+8C48
    // a, acute (230), circumflex (230), dot below (220): the dot moves first and composes with a;
    // the acute and circumflex stay, each still pointing at its own source bytes.
    const NormalizedString r = normalize("a\xCC\x81\xCC\x82\xCC\xA3");
    EXPECT_EQ(r.text(), "\xE1\xBA\xA1\xCC\x81\xCC\x82");                       // ạ, acute, circumflex
    EXPECT_EQ(r.original(0, 3), (Offset{0, 7}));  // ạ comes from a (0) and the dot below (5..7)
    EXPECT_EQ(r.original(3, 5), (Offset{1, 3}));  // the acute
    EXPECT_EQ(r.original(5, 7), (Offset{3, 5}));  // the circumflex
    const std::string ascii = "plain ASCII stays as it is";
    EXPECT_EQ(normalize(ascii).text(), ascii);
}

// ---- Regex -----------------------------------------------------------------------------

std::vector<std::string> Matches(const std::string& pattern, const std::string& text) {
    std::vector<std::string> out;
    for (const auto& m : UnicodeRegex(pattern).find_all(text)) out.push_back(text.substr(m.begin, m.end - m.begin));
    return out;
}

TEST(UnicodeRegexTest, LeftmostFirstAlternationAndQuantifiers) {
    EXPECT_EQ(Matches("a|ab", "ab"), (std::vector<std::string>{"a"}));  // first alternative wins, not the longest
    EXPECT_EQ(Matches("\\p{N}{1,3}", "1234567"), (std::vector<std::string>{"123", "456", "7"}));
    EXPECT_EQ(Matches("a+?", "aaa"), (std::vector<std::string>{"a", "a", "a"}));
    EXPECT_EQ(Matches("x(?:ab)*y", "xababy xy xaby"), (std::vector<std::string>{"xababy", "xy", "xaby"}));
    EXPECT_EQ(Matches("\\s+(?!\\S)|\\s+", "a   b"), (std::vector<std::string>{"  ", " "}));  // the GPT-2 trick
    EXPECT_EQ(Matches("[^\\s\\p{L}]+", "ab, 12!"), (std::vector<std::string>{",", "12!"}));
}

TEST(UnicodeRegexTest, UnicodeCategoriesAndCaseInsensitiveContractions) {
    EXPECT_EQ(Matches("\\p{L}+", "Grüße, мир 世界"), (std::vector<std::string>{"Grüße", "мир", "世界"}));
    EXPECT_EQ(Matches("\\p{Lu}\\p{Ll}+", "HTTPServer"), (std::vector<std::string>{"Server"}));
    EXPECT_EQ(Matches("[\\p{Lu}\\p{Lt}]", "aǅB"), (std::vector<std::string>{"ǅ", "B"}));
    EXPECT_EQ(Matches("\\P{L}+", "ab12cd"), (std::vector<std::string>{"12"}));
    EXPECT_EQ(Matches("\\p{M}", "e\xCC\x81"), (std::vector<std::string>{"\xCC\x81"}));
    EXPECT_EQ(Matches("(?i:'s|'ll)", "it's IT'S we'LL it'ſ"), (std::vector<std::string>{"'s", "'S", "'LL", "'ſ"}));
    EXPECT_EQ(Matches("\\s", "a\xC2\xA0" "b\xE2\x80\xA8" "c\x1C" "d"), (std::vector<std::string>{"\xC2\xA0", "\xE2\x80\xA8"}));
}

TEST(UnicodeRegexTest, RefusesSyntaxItDoesntSupport) {
    for (const char* bad : {"^a", "a$", "(?<name>a)", "\\1", "\\b", "a**", "(a", "a)", "[a", "\\p{Bogus}", "*a", "(?=a)+"}) {
        EXPECT_THROW(UnicodeRegex{bad}, std::invalid_argument) << bad;
    }
    EXPECT_THROW((void)UnicodeRegex("a").find_all(std::string("\xFF")), std::invalid_argument);
}

TEST(JsonTest, FindsDuplicateKeysInLargeObjects) {
    std::string json = "{";
    for (int i = 0; i < 100; ++i) json += "\"k" + std::to_string(i) + "\": " + std::to_string(i) + ", ";
    EXPECT_NO_THROW((void)ParseJson(json + "\"last\": 0}"));
    EXPECT_THROW((void)ParseJson(json + "\"k7\": 0}"), std::invalid_argument);   // duplicate of an early key
    EXPECT_THROW((void)ParseJson(json + "\"k99\": 0}"), std::invalid_argument);  // and of a late one
}

// Real tokenizers on the full corpus, when a directory is given.
TEST(RealTokenizersTest, EveryTokenizerInPulsatrixTokenizerDirMatches) {
    const char* dir = std::getenv("PULSATRIX_TOKENIZER_DIR");
    if (dir == nullptr) GTEST_SKIP() << "set PULSATRIX_TOKENIZER_DIR to a directory of NAME/{tokenizer.json,reference.jsonl}";
    int checked = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!std::filesystem::exists(entry.path() / "reference.jsonl")) continue;
        const TextTokenizer tok = LoadTokenizerJson((entry.path() / "tokenizer.json").string());
        const TokenizerParityReport r = CompareToTokenizerReference(tok, (entry.path() / "reference.jsonl").string());
        EXPECT_TRUE(r.passed()) << entry.path() << ": " << Describe(r);
        ++checked;
    }
    EXPECT_GT(checked, 0) << "no NAME/reference.jsonl in " << dir;
}

}  // namespace
}  // namespace pulsatrix
