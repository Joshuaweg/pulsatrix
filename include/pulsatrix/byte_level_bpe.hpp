/** @file byte_level_bpe.hpp
 *  @brief The components byte-level BPE tokenizers are built from (TOK-2): NFC, regex and digit
 *         splitting, byte-level mapping, BPE, template post-processing and byte-level decoding,
 *         each following Hugging Face `tokenizers`.
 *  @ingroup data_pipeline
 */
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pulsatrix/text_tokenizer.hpp"
#include "pulsatrix/unicode_regex.hpp"

namespace pulsatrix {

/** @brief Unicode Normalization Form C (canonical composition), keeping offsets. */
class NfcNormalizer : public Normalizer {
public:
    void normalize(NormalizedString& text) const override;
};

/** @brief Runs normalizers in order. */
class SequenceNormalizer : public Normalizer {
public:
    explicit SequenceNormalizer(std::vector<std::shared_ptr<const Normalizer>> normalizers);
    void normalize(NormalizedString& text) const override;

private:
    std::vector<std::shared_ptr<const Normalizer>> normalizers_;
};

/** @brief What a split does with the pieces that match (Hugging Face's SplitDelimiterBehavior). */
enum class SplitBehavior { Removed, Isolated, MergedWithPrevious, MergedWithNext, Contiguous };

/** @brief Splits on a regex or a literal string (Hugging Face's Split pre-tokenizer). */
class SplitPreTokenizer : public PreTokenizer {
public:
    /** @brief A regex pattern. @throws std::invalid_argument as UnicodeRegex does. */
    SplitPreTokenizer(const std::string& regex, SplitBehavior behavior, bool invert = false);
    /** @brief A literal string pattern. */
    static SplitPreTokenizer Literal(const std::string& text, SplitBehavior behavior, bool invert = false);
    void pre_tokenize(std::vector<NormalizedString>& splits) const override;

private:
    SplitPreTokenizer(std::optional<UnicodeRegex> regex, std::string literal, SplitBehavior behavior, bool invert);
    std::optional<UnicodeRegex> regex_;
    std::string literal_;
    SplitBehavior behavior_;
    bool invert_;
};

/** @brief Splits off numeric characters (Unicode N), one by one or in runs (Hugging Face's Digits). */
class DigitsPreTokenizer : public PreTokenizer {
public:
    explicit DigitsPreTokenizer(bool individual_digits);
    void pre_tokenize(std::vector<NormalizedString>& splits) const override;

private:
    bool individual_;
};

/**
 * @brief Byte-level pre-tokenization (Hugging Face's ByteLevel): optionally prepends a space,
 *        optionally splits with the GPT-2 regex, then maps every byte to a printable character
 *        with GPT-2's table, so BPE runs over characters that stand for bytes.
 */
class ByteLevelPreTokenizer : public PreTokenizer {
public:
    ByteLevelPreTokenizer(bool add_prefix_space, bool use_regex);
    void pre_tokenize(std::vector<NormalizedString>& splits) const override;

private:
    bool add_prefix_space_;
    std::optional<UnicodeRegex> regex_;
};

/** @brief Runs pre-tokenizers in order. */
class SequencePreTokenizer : public PreTokenizer {
public:
    explicit SequencePreTokenizer(std::vector<std::shared_ptr<const PreTokenizer>> pre_tokenizers);
    void pre_tokenize(std::vector<NormalizedString>& splits) const override;

private:
    std::vector<std::shared_ptr<const PreTokenizer>> pre_tokenizers_;
};

/** @brief BPE options beyond the vocabulary and merges, as `tokenizer.json` names them. */
struct BpeOptions {
    /** @brief A piece that is a whole vocabulary entry is one token, without merging (Llama 3). */
    bool ignore_merges = false;
    /** @brief Added before every character but a piece's first. */
    std::string continuing_subword_prefix;
    /** @brief Added after a piece's last character. */
    std::string end_of_word_suffix;
    /** @brief For characters the vocabulary lacks; without one they are dropped, as in Hugging Face. */
    std::optional<std::string> unk_token;
};

/** @brief Byte-pair encoding: start from characters and apply merges, lowest rank first. */
class BpeModel : public TokenModel {
public:
    /**
     * @param vocab Token strings and their ids.
     * @param merges Merge rules, highest priority first.
     * @throws std::invalid_argument if a merge's parts or result aren't in the vocabulary, or the
     *         unknown token isn't.
     */
    BpeModel(std::unordered_map<std::string, int64_t> vocab, const std::vector<std::pair<std::string, std::string>>& merges,
             BpeOptions options = {});
    [[nodiscard]] std::vector<ModelToken> tokenize(std::string_view piece) const override;
    [[nodiscard]] std::optional<int64_t> token_to_id(std::string_view token) const override;
    [[nodiscard]] std::optional<std::string> id_to_token(int64_t id) const override;
    [[nodiscard]] int64_t vocab_size() const override { return vocab_size_; }

private:
    struct PairHash {
        size_t operator()(const std::pair<int64_t, int64_t>& p) const noexcept {
            return std::hash<int64_t>()(p.first * 1000003 + p.second);
        }
    };
    std::unordered_map<std::string, int64_t> vocab_;
    std::unordered_map<int64_t, std::string> tokens_;
    /** @brief (left id, right id) -> (rank, merged id). */
    std::unordered_map<std::pair<int64_t, int64_t>, std::pair<int64_t, int64_t>, PairHash> merges_;
    BpeOptions options_;
    std::optional<int64_t> unk_id_;
    int64_t vocab_size_ = 0;
};

/**
 * @brief Byte-level post-processing (Hugging Face's ByteLevel post-processor): with trim_offsets,
 *        each token's offsets leave out the spaces at its edges.
 */
class ByteLevelPostProcessor : public PostProcessor {
public:
    ByteLevelPostProcessor(bool add_prefix_space, bool trim_offsets);
    [[nodiscard]] Encoding process(Encoding encoding, bool add_special_tokens) const override;

private:
    bool add_prefix_space_;
    bool trim_offsets_;
};

/** @brief Special tokens around a single sequence (Hugging Face's TemplateProcessing, `single`). */
class TemplatePostProcessor : public PostProcessor {
public:
    /** @brief One template item: the sequence itself (`special` empty), or special tokens. */
    struct Item {
        std::vector<int64_t> special_ids;
        std::vector<std::string> special_tokens;
        [[nodiscard]] bool is_sequence() const { return special_ids.empty(); }
    };
    explicit TemplatePostProcessor(std::vector<Item> single);
    [[nodiscard]] Encoding process(Encoding encoding, bool add_special_tokens) const override;

private:
    std::vector<Item> single_;
};

/** @brief Runs post-processors in order. */
class SequencePostProcessor : public PostProcessor {
public:
    explicit SequencePostProcessor(std::vector<std::shared_ptr<const PostProcessor>> processors);
    [[nodiscard]] Encoding process(Encoding encoding, bool add_special_tokens) const override;

private:
    std::vector<std::shared_ptr<const PostProcessor>> processors_;
};

/** @brief Maps byte-level characters back to bytes; invalid UTF-8 becomes U+FFFD, as in Hugging
 *         Face. */
class ByteLevelDecoder : public Decoder {
public:
    [[nodiscard]] std::string decode(const std::vector<std::string>& tokens) const override;
};

/** @brief The GPT-2 byte-to-character table byte-level BPE uses: printable bytes stand for
 *         themselves, the rest for U+0100 onward. */
[[nodiscard]] const std::vector<std::string>& ByteLevelAlphabet();

}  // namespace pulsatrix
