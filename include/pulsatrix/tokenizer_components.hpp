/** @file tokenizer_components.hpp
 *  @brief Pipeline components for TextTokenizer (TOK-1), and the word-level, byte and character
 *         tokenizers built from them.
 *  @ingroup data_pipeline
 */
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "pulsatrix/text_tokenizer.hpp"
#include "pulsatrix/vocabulary.hpp"

namespace pulsatrix {

/** @brief Lowercases ASCII letters and leaves every other character alone. */
class AsciiLowercase : public Normalizer {
public:
    void normalize(NormalizedString& text) const override;
};

/**
 * @brief The split Tokenizer::Tokenize makes: runs of ASCII letters, digits and apostrophes are
 *        words, ASCII whitespace separates, and every other character is a token of its own.
 * @note Tokenizer::Tokenize works byte by byte and splits a non-ASCII character into its bytes;
 *       this keeps the character whole. On ASCII text the two agree.
 */
class WordPunctuationSplit : public PreTokenizer {
public:
    void pre_tokenize(std::vector<NormalizedString>& splits) const override;
};

/** @brief Each piece is one token, looked up in a vocabulary; unknown pieces become the unknown
 *         token. */
class WordLevelModel : public TokenModel {
public:
    /** @param vocab Token strings by id. @throws std::invalid_argument if @p unk_token isn't in it, or a token repeats. */
    WordLevelModel(std::vector<std::string> vocab, const std::string& unk_token);
    [[nodiscard]] std::vector<ModelToken> tokenize(std::string_view piece) const override;
    [[nodiscard]] std::optional<int64_t> token_to_id(std::string_view token) const override;
    [[nodiscard]] std::optional<std::string> id_to_token(int64_t id) const override;
    [[nodiscard]] int64_t vocab_size() const override { return static_cast<int64_t>(vocab_.size()); }

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int64_t> ids_;
    int64_t unk_id_ = 0;
};

/** @brief Every byte is a token whose id is the byte's value (0..255); the token string is the
 *         byte itself. */
class ByteModel : public TokenModel {
public:
    [[nodiscard]] std::vector<ModelToken> tokenize(std::string_view piece) const override;
    [[nodiscard]] std::optional<int64_t> token_to_id(std::string_view token) const override;
    [[nodiscard]] std::optional<std::string> id_to_token(int64_t id) const override;
    [[nodiscard]] int64_t vocab_size() const override { return 256; }
};

/** @brief Every character (code point) is a token, looked up in a character list; unknown
 *         characters become the unknown token. */
class CharModel : public TokenModel {
public:
    /** @param chars One UTF-8 character per entry, by id. @throws std::invalid_argument if an
     *         entry isn't one character, repeats, or @p unk_token isn't among them. */
    CharModel(std::vector<std::string> chars, const std::string& unk_token);
    [[nodiscard]] std::vector<ModelToken> tokenize(std::string_view piece) const override;
    [[nodiscard]] std::optional<int64_t> token_to_id(std::string_view token) const override;
    [[nodiscard]] std::optional<std::string> id_to_token(int64_t id) const override;
    [[nodiscard]] int64_t vocab_size() const override { return static_cast<int64_t>(chars_.size()); }

private:
    std::vector<std::string> chars_;
    std::unordered_map<std::string, int64_t> ids_;
    int64_t unk_id_ = 0;
};

/** @brief Concatenates the tokens (Hugging Face's Fuse decoder). */
class FuseDecoder : public Decoder {
public:
    [[nodiscard]] std::string decode(const std::vector<std::string>& tokens) const override;
};

/**
 * @brief The word-level tokenizer over a Vocabulary: AsciiLowercase, WordPunctuationSplit and a
 *        WordLevelModel whose unknown token is `<unk>` (id 0). On ASCII text its tokens are
 *        Tokenizer::Tokenize's and its ids are Vocabulary::IndexOf's.
 */
[[nodiscard]] TextTokenizer MakeWordLevelTokenizer(const Vocabulary& vocab);

/** @brief Bytes as ids 0..255, then @p special_tokens as ids 256, 257, ...; decodes by
 *         concatenation. */
[[nodiscard]] TextTokenizer MakeByteTokenizer(const std::vector<std::string>& special_tokens = {});

/**
 * @brief Characters by id from @p chars, then `<unk>` (unless @p chars has it), then
 *        @p special_tokens; decodes by concatenation.
 * @throws std::invalid_argument as CharModel does.
 */
[[nodiscard]] TextTokenizer MakeCharTokenizer(std::vector<std::string> chars,
                                              const std::vector<std::string>& special_tokens = {});

}  // namespace pulsatrix
