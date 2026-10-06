/** @file text_tokenizer.hpp
 *  @brief The tokenizer interface (TOK-1): a pipeline of normalizer, pre-tokenizer, model,
 *         post-processor and decoder, as in Hugging Face `tokenizers`, with added tokens split
 *         out first. encode() returns ids, token strings, offsets into the input and a special-
 *         token mask; decode() returns text.
 *  @ingroup data_pipeline
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pulsatrix {

/** @brief A byte range `[begin, end)` of the text passed to TextTokenizer::encode(). */
struct Offset {
    size_t begin = 0;
    size_t end = 0;
    bool operator==(const Offset& other) const { return begin == other.begin && end == other.end; }
};

/**
 * @brief The result of encoding one text.
 * @note Offsets are UTF-8 byte offsets into the input, so `text.substr(o.begin, o.end - o.begin)`
 *       is the token's source. Hugging Face reports offsets in characters instead. A byte-level
 *       token can cover part of a multi-byte character, and then its offset covers just those
 *       bytes. A character a normalizer composed (NFC) maps to all of its source characters,
 *       where Hugging Face maps it to the first only. Tokens inserted by a post-processor (BOS,
 *       EOS) have an empty offset.
 */
struct Encoding {
    std::vector<int64_t> ids;
    std::vector<std::string> tokens;
    std::vector<Offset> offsets;
    /** @brief 1 for special tokens: added tokens marked special, and tokens a post-processor
     *         inserted. */
    std::vector<uint8_t> special_tokens_mask;

    [[nodiscard]] size_t size() const { return ids.size(); }
    /** @brief Appends one token. */
    void push_back(int64_t id, std::string token, Offset offset, bool special);
};

/**
 * @brief Text being normalized and pre-tokenized, remembering where each byte came from in the
 *        original input, so offsets survive changes such as NFC composition or byte-to-character
 *        mapping.
 */
class NormalizedString {
public:
    NormalizedString() = default;
    /** @brief @p text as it appears at byte @p offset of the original input, unchanged. */
    explicit NormalizedString(std::string text, size_t offset = 0);

    [[nodiscard]] const std::string& text() const { return text_; }
    [[nodiscard]] bool empty() const { return text_.empty(); }
    [[nodiscard]] size_t size() const { return text_.size(); }

    /**
     * @brief The original bytes that normalized bytes `[begin, end)` came from. An empty range
     *        gives an empty span where it sits.
     * @throws std::out_of_range if the range doesn't fit the text.
     */
    [[nodiscard]] Offset original(size_t begin, size_t end) const;

    /** @brief Normalized bytes `[begin, end)`, keeping their origins. @throws std::out_of_range. */
    [[nodiscard]] NormalizedString slice(size_t begin, size_t end) const;

    /** @brief One piece of a rebuilt text: its new bytes, and the range of the current text they
     *         replace (an empty range inserts). */
    struct Piece {
        std::string text;
        size_t source_begin = 0;
        size_t source_end = 0;
    };
    /**
     * @brief Replaces the text with @p pieces, in order. Every byte of a piece comes from the
     *        original span of its source range, which is how normalizers and byte-level mapping
     *        keep offsets.
     * @throws std::out_of_range if a source range doesn't fit the current text.
     */
    void rebuild(const std::vector<Piece>& pieces);

private:
    std::string text_;
    std::vector<Offset> origin_;  ///< per byte of text_: the original bytes it came from
};

/** @brief Changes text before it is split (lowercasing, NFC, ...). */
class Normalizer {
public:
    virtual ~Normalizer() = default;
    virtual void normalize(NormalizedString& text) const = 0;
};

/** @brief Splits text into the pieces the model tokenizes separately, and may change them (byte-
 *         level mapping). */
class PreTokenizer {
public:
    virtual ~PreTokenizer() = default;
    /** @brief Replaces @p splits with their further-split (or transformed) pieces, in order. */
    virtual void pre_tokenize(std::vector<NormalizedString>& splits) const = 0;
};

/** @brief One token from a TokenModel: its id and string, and the bytes of the piece it covers. */
struct ModelToken {
    int64_t id = 0;
    std::string value;
    size_t begin = 0;
    size_t end = 0;
};

/** @brief Turns one pre-tokenized piece into tokens (word-level lookup, BPE, ...). */
class TokenModel {
public:
    virtual ~TokenModel() = default;
    [[nodiscard]] virtual std::vector<ModelToken> tokenize(std::string_view piece) const = 0;
    [[nodiscard]] virtual std::optional<int64_t> token_to_id(std::string_view token) const = 0;
    [[nodiscard]] virtual std::optional<std::string> id_to_token(int64_t id) const = 0;
    /** @brief One more than the largest id the model can produce. */
    [[nodiscard]] virtual int64_t vocab_size() const = 0;
};

/** @brief Adds tokens around an encoding (BOS and EOS, sentence pairs, ...). */
class PostProcessor {
public:
    virtual ~PostProcessor() = default;
    [[nodiscard]] virtual Encoding process(Encoding encoding, bool add_special_tokens) const = 0;
};

/** @brief Turns token strings back into text. */
class Decoder {
public:
    virtual ~Decoder() = default;
    [[nodiscard]] virtual std::string decode(const std::vector<std::string>& tokens) const = 0;
};

/** @brief A token matched in the raw input before anything else runs, such as `<|im_start|>`. */
struct AddedToken {
    std::string content;
    int64_t id = 0;
    /** @brief Marked in the special-tokens mask and dropped by decode(skip_special_tokens). */
    bool special = true;
};

/**
 * @brief A tokenizer: added tokens are split out of the raw text first (longest match, earliest
 *        first); each remaining segment is normalized, pre-tokenized and tokenized by the model;
 *        then the post-processor runs. Every component but the model is optional.
 * @note Input must be valid UTF-8.
 */
class TextTokenizer {
public:
    explicit TextTokenizer(std::shared_ptr<const TokenModel> model);

    TextTokenizer& set_normalizer(std::shared_ptr<const Normalizer> normalizer);
    TextTokenizer& set_pre_tokenizer(std::shared_ptr<const PreTokenizer> pre_tokenizer);
    TextTokenizer& set_post_processor(std::shared_ptr<const PostProcessor> post_processor);
    TextTokenizer& set_decoder(std::shared_ptr<const Decoder> decoder);
    /** @throws std::invalid_argument for empty content, or content or an id already added. */
    TextTokenizer& add_token(AddedToken token);

    /** @throws std::invalid_argument if @p text isn't valid UTF-8. */
    [[nodiscard]] Encoding encode(std::string_view text, bool add_special_tokens = true) const;
    /**
     * @brief The text for @p ids: their token strings through the decoder, or joined with spaces
     *        when there is none (as Hugging Face does).
     * @throws std::out_of_range for an id the tokenizer doesn't have.
     */
    [[nodiscard]] std::string decode(const std::vector<int64_t>& ids, bool skip_special_tokens = false) const;

    /** @brief One more than the largest id, over the model and the added tokens. */
    [[nodiscard]] int64_t vocab_size() const;
    [[nodiscard]] std::optional<int64_t> token_to_id(std::string_view token) const;
    [[nodiscard]] std::optional<std::string> id_to_token(int64_t id) const;
    [[nodiscard]] const std::vector<AddedToken>& added_tokens() const { return added_; }

private:
    void encode_segment(std::string_view text, size_t offset, Encoding& out) const;

    std::shared_ptr<const TokenModel> model_;
    std::shared_ptr<const Normalizer> normalizer_;
    std::shared_ptr<const PreTokenizer> pre_tokenizer_;
    std::shared_ptr<const PostProcessor> post_processor_;
    std::shared_ptr<const Decoder> decoder_;
    std::vector<AddedToken> added_;
    std::unordered_map<std::string, size_t> added_by_content_;
    std::unordered_map<int64_t, size_t> added_by_id_;
    /** @brief Added tokens by first byte, longest first. */
    std::unordered_map<unsigned char, std::vector<size_t>> added_by_first_byte_;
};

}  // namespace pulsatrix
