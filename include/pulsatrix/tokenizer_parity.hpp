/** @file tokenizer_parity.hpp
 *  @brief Compares a TextTokenizer with Hugging Face `tokenizers` on a reference corpus (TOK-2),
 *         as written by tools/tokenizers/make_tokenizer_reference.py.
 *  @ingroup data_pipeline
 */
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "pulsatrix/text_tokenizer.hpp"

namespace pulsatrix {

/** @brief The first difference on one line of the corpus. */
struct TokenizerMismatch {
    size_t line = 0;
    /** @brief "ids", "plain_ids", "offsets", "decoded" or "decoded_skip". */
    std::string field;
    std::string text;
    std::string detail;
};

/** @brief How a tokenizer compares with the reference, field by field. */
struct TokenizerParityReport {
    size_t lines = 0;
    size_t id_mismatches = 0;      ///< lines whose ids (with or without special tokens) differ
    /** @brief Lines that round-trip exactly whose ids match but offsets don't. */
    size_t offset_mismatches = 0;
    /**
     * @brief Lines that don't round-trip exactly whose offsets differ: counted, not failed. Such a
     *        line was changed by the normalizer (NFC) or lost characters the vocabulary lacks
     *        (dropped without an unknown token, as Hugging Face drops them). Hugging Face's offsets
     *        there are positional: a composed character gets only its first source character,
     *        characters after a canonical reordering get their neighbours' spans, and every token
     *        after a dropped character shifts left. pulsatrix tracks each byte's actual source.
     */
    size_t inexact_line_offset_differences = 0;
    size_t decode_mismatches = 0;  ///< lines whose decoded text differs
    /** @brief The first few mismatches, for diagnosis. */
    std::vector<TokenizerMismatch> examples;
    [[nodiscard]] bool passed() const { return lines > 0 && id_mismatches == 0 && offset_mismatches == 0 && decode_mismatches == 0; }
};

/**
 * @brief Encodes and decodes every line of a reference file (JSON lines with `text`, `ids`,
 *        `plain_ids`, `offsets` in characters, `decoded`, `decoded_skip`, and `normalized`: whether
 *        the normalizer changed the line) and counts the lines that differ. A line round-trips
 *        exactly when the normalizer leaves it alone and decoding its plain ids gives it back;
 *        offsets are compared exactly on those lines.
 * @param max_examples How many mismatches to keep in the report.
 * @throws std::runtime_error if the file can't be read; std::invalid_argument if a line is
 *         malformed.
 */
[[nodiscard]] TokenizerParityReport CompareToTokenizerReference(const TextTokenizer& tokenizer, const std::string& path,
                                                                size_t max_examples = 5);

/** @brief A byte offset range of @p text as a character (code point) range, widened to whole
 *         characters, which is how Hugging Face reports offsets. */
[[nodiscard]] Offset ByteToCharOffset(std::string_view text, Offset bytes);

}  // namespace pulsatrix
