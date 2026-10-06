/** @file word_scores.hpp
 *  @brief Per-word scores from per-token scores (TOK-4): explanations are made per token but read
 *         per word, so relevance, attributions or probe outputs are merged into the words of the
 *         input using the tokenizer's offsets.
 *  @ingroup data_pipeline
 */
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/text_tokenizer.hpp"
#include "pulsatrix/unicode_regex.hpp"

namespace pulsatrix {

/** @brief How the text is cut into words. */
enum class WordSplit {
    /** @brief Runs of letters, marks, digits and underscores, and runs of other non-space
     *         characters (`\w+|[^\w\s]+`, Hugging Face's Whitespace pre-tokenizer): "don't!" is
     *         "don", "'", "t", "!". */
    WordsAndPunctuation,
    /** @brief Runs of non-space characters (`\S+`): "don't!" is one word. */
    Whitespace,
};

/** @brief The words of valid UTF-8 @p text, as byte ranges in order. Scripts written without
 *         spaces (Chinese, Japanese, Thai) come out as long runs; pass a pattern for those.
 *  @throws std::invalid_argument if @p text isn't valid UTF-8. */
[[nodiscard]] std::vector<Offset> SplitWords(std::string_view text, WordSplit split = WordSplit::WordsAndPunctuation);
/** @brief The words of @p text: every match of @p pattern. */
[[nodiscard]] std::vector<Offset> SplitWords(std::string_view text, const UnicodeRegex& pattern);

/**
 * @brief How a word's score is made from the scores of the tokens that overlap it.
 *
 * A token overlapping several words gives each a share: the fraction of its bytes inside words
 * that fall inside that word. Bytes outside every word (the space in a byte-level "Ġworld") don't
 * count, so a token's shares add up to 1.
 */
enum class WordAggregation {
    /** @brief Σ share × score. Conserves the total: the words' scores plus the unassigned score add
     *         up to the tokens' (what relevance needs). */
    Sum,
    /** @brief Σ share × score / Σ share: the share-weighted mean of the overlapping tokens. */
    Mean,
    /** @brief The largest score among the overlapping tokens. */
    Max,
    /** @brief The score with the largest magnitude among the overlapping tokens, sign kept (for
     *         signed attributions, where Max would hide strong negative evidence). */
    MaxAbs,
};

/** @brief One word and its score. */
struct WordScore {
    Offset span;
    std::string text;
    float score = 0.0f;
    /** @brief Indices of the tokens that overlap the word, in order. Empty when none does (a
     *         character the vocabulary dropped), and then the score is 0. */
    std::vector<size_t> tokens;
};

/** @brief Per-word scores for one text. */
struct WordScores {
    std::vector<WordScore> words;
    /** @brief The total score of tokens that overlap no word: BOS, EOS and other tokens with an
     *         empty offset, and tokens of whitespace only. */
    float unassigned = 0.0f;
};

/**
 * @brief Merges per-token scores into the given words.
 * @param text The text the offsets point into.
 * @param token_offsets One byte range of @p text per token (Encoding::offsets).
 * @param token_scores One score per token.
 * @param words Byte ranges of @p text, in order and not overlapping (SplitWords()).
 * @throws std::invalid_argument if the sizes differ, or an offset or word lies outside the text,
 *         or the words overlap or are out of order.
 */
[[nodiscard]] WordScores AggregateToWords(std::string_view text, const std::vector<Offset>& token_offsets,
                                          const std::vector<float>& token_scores, const std::vector<Offset>& words,
                                          WordAggregation aggregation = WordAggregation::Sum);

/** @brief AggregateToWords over an Encoding of @p text, with words from SplitWords(@p text, @p split). */
[[nodiscard]] WordScores AggregateToWords(std::string_view text, const Encoding& encoding,
                                          const std::vector<float>& token_scores,
                                          WordAggregation aggregation = WordAggregation::Sum,
                                          WordSplit split = WordSplit::WordsAndPunctuation);

}  // namespace pulsatrix
