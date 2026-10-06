/** @file text_relevance.hpp
 *  @brief Token relevance documents from real explanations (VIZ-6a): a tokenizer's Encoding or
 *         TOK-4's word scores, turned into a `pulsatrix.token_relevance.v1` document whose pieces
 *         read as the original text.
 *  @ingroup visualization
 */
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/text_tokenizer.hpp"
#include "pulsatrix/viz/document.hpp"
#include "pulsatrix/word_scores.hpp"

namespace pulsatrix {

/**
 * @brief A token-level document for @p text from its Encoding and one score per token.
 *
 * Each token is shown as the text its offset covers, so byte-level tokens appear decoded, as
 * UTF-8. Tokens that share a character (byte-level tokens that split one, such as an emoji) are
 * shown as one piece whose score is their sum. Text no token covers becomes an unscored piece, so
 * the pieces read as @p text. Tokens with an empty offset (BOS, EOS) are shown as their token
 * strings when @p include_special_tokens is set; otherwise their relevance goes to `unassigned`.
 *
 * @throws std::invalid_argument if the scores don't match the tokens or an offset lies outside
 *         @p text.
 */
[[nodiscard]] TokenRelevanceDocument MakeTokenRelevanceDocument(std::string_view text, const Encoding& encoding,
                                                                const std::vector<float>& token_scores, std::string method,
                                                                std::string target = "", bool include_special_tokens = true);

/**
 * @brief A word-level document for @p text from AggregateToWords(): each word scored, the text
 *        between words as unscored pieces, and the words' unassigned relevance carried over.
 * @throws std::invalid_argument if a word's span lies outside @p text or the words are out of order.
 */
[[nodiscard]] TokenRelevanceDocument MakeWordRelevanceDocument(std::string_view text, const WordScores& words,
                                                               std::string method, std::string target = "");

}  // namespace pulsatrix
