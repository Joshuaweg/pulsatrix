/** @file sentencepiece_bpe.hpp
 *  @brief The components SentencePiece-style BPE tokenizers are built from (TOK-3): spaces as `▁`
 *         (Replace, Prepend or Metaspace), byte fallback in BpeModel (`BpeOptions::byte_fallback`),
 *         and the decoder chain that undoes them, each following Hugging Face `tokenizers`.
 *         Gemma 3, Llama 2, Mistral and TinyLlama tokenize this way.
 *  @ingroup data_pipeline
 */
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "pulsatrix/text_tokenizer.hpp"
#include "pulsatrix/unicode_regex.hpp"

namespace pulsatrix {

/** @brief Replaces every match of a literal string or a regex (Hugging Face's Replace normalizer). */
class ReplaceNormalizer : public Normalizer {
public:
    /** @brief A literal pattern. @throws std::invalid_argument if it's empty. */
    ReplaceNormalizer(std::string pattern, std::string content);
    /** @brief A regex pattern. */
    static ReplaceNormalizer Regex(const std::string& pattern, std::string content);
    void normalize(NormalizedString& text) const override;

private:
    ReplaceNormalizer(std::optional<UnicodeRegex> regex, std::string literal, std::string content);
    std::optional<UnicodeRegex> regex_;
    std::string literal_;
    std::string content_;
};

/** @brief Puts a string in front of non-empty text (Hugging Face's Prepend normalizer). */
class PrependNormalizer : public Normalizer {
public:
    explicit PrependNormalizer(std::string prepend);
    void normalize(NormalizedString& text) const override;

private:
    std::string prepend_;
};

/** @brief When Metaspace puts its replacement in front of a piece. */
enum class PrependScheme { Always, First, Never };

/**
 * @brief Spaces become a replacement character (`▁`), one is put in front of each piece (or of
 *        the text's first piece only), and pieces are optionally split before each one
 *        (Hugging Face's Metaspace pre-tokenizer).
 */
class MetaspacePreTokenizer : public PreTokenizer {
public:
    MetaspacePreTokenizer(std::string replacement, PrependScheme scheme, bool split);
    void pre_tokenize(std::vector<NormalizedString>& splits) const override;

private:
    std::string replacement_;
    PrependScheme scheme_;
    bool split_;
};

/** @brief Replaces a literal string or regex in every token (Hugging Face's Replace decoder). */
class ReplaceDecoder : public Decoder {
public:
    ReplaceDecoder(std::string pattern, std::string content);
    static ReplaceDecoder Regex(const std::string& pattern, std::string content);
    [[nodiscard]] std::vector<std::string> decode_chain(std::vector<std::string> tokens) const override;

private:
    ReplaceDecoder(std::optional<UnicodeRegex> regex, std::string literal, std::string content);
    std::optional<UnicodeRegex> regex_;
    std::string literal_;
    std::string content_;
};

/** @brief Turns runs of `<0x41>`-style byte tokens back into text; a run that isn't valid UTF-8
 *         becomes one U+FFFD per byte (Hugging Face's ByteFallback decoder). */
class ByteFallbackDecoder : public Decoder {
public:
    [[nodiscard]] std::vector<std::string> decode_chain(std::vector<std::string> tokens) const override;
};

/** @brief Removes up to `start` leading and `stop` trailing copies of a character from every
 *         token (Hugging Face's Strip decoder). */
class StripDecoder : public Decoder {
public:
    StripDecoder(std::string content, size_t start, size_t stop);
    [[nodiscard]] std::vector<std::string> decode_chain(std::vector<std::string> tokens) const override;

private:
    std::string content_;
    size_t start_;
    size_t stop_;
};

/** @brief The replacement character back to spaces (Hugging Face's Metaspace decoder). Unless
 *         the scheme is Never, every replacement in the first token is dropped, as Hugging Face
 *         does, not only the one the pre-tokenizer put in front. */
class MetaspaceDecoder : public Decoder {
public:
    MetaspaceDecoder(std::string replacement, PrependScheme scheme);
    [[nodiscard]] std::vector<std::string> decode_chain(std::vector<std::string> tokens) const override;

private:
    std::string replacement_;
    PrependScheme scheme_;
};

/** @brief Runs decoders in order, each on the previous one's tokens. */
class SequenceDecoder : public Decoder {
public:
    explicit SequenceDecoder(std::vector<std::shared_ptr<const Decoder>> decoders);
    [[nodiscard]] std::vector<std::string> decode_chain(std::vector<std::string> tokens) const override;

private:
    std::vector<std::shared_ptr<const Decoder>> decoders_;
};

}  // namespace pulsatrix
