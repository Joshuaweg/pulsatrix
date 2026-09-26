/** @file tokenizer.hpp
 *  @brief Minimal deterministic whitespace/punctuation word-level tokenizer.
 *  @ingroup dl_modules
 */
#pragma once

#include <string>
#include <vector>

namespace pulsatrix {

/**
 * @brief Splits text into lowercase word/punctuation tokens -- pulsatrix's first text
 *        primitive (campaign_exai_dl_library_data_pipeline, Phase 3, Decision Point 6:
 *        a minimal whitespace/punctuation tokenizer, not BPE -- training a real
 *        subword-merge algorithm is a project-sized undertaking on its own).
 * @note ASCII-oriented (uses std::tolower/std::isalnum/std::isspace per byte) -- Unicode
 *       normalization is out of scope for this phase.
 */
class Tokenizer {
public:
    /**
     * @brief Tokenizes text: lowercase-normalizes, splits on whitespace (a pure separator,
     *        never emitted), emits runs of alphanumeric/apostrophe characters as single
     *        word tokens, and emits every other non-whitespace character as its own
     *        single-character punctuation token.
     * @param text Input text.
     * @return The token sequence, in order. Empty if text has no non-whitespace content.
     */
    [[nodiscard]] static std::vector<std::string> Tokenize(const std::string& text);
};

}  // namespace pulsatrix
