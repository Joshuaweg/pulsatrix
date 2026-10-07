/** @file protein_sequences.hpp
 *  @brief Protein sequence input (PLM-2): ESM's tokenizer from its `vocab.txt`, and FASTA files.
 *  @ingroup data_pipeline
 */
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/text_tokenizer.hpp"

namespace pulsatrix {

/**
 * @brief ESM's tokenizer (Hugging Face's `EsmTokenizer`) over a vocabulary listed one token per
 *        line, by id: `<cls> <pad> <eos> <unk> L A G V ... <mask>` for ESM-2.
 *
 * - Every vocabulary token is matched anywhere in the text, so `MKT<mask>Y` gives M, K, T,
 *   `<mask>`, Y.
 * - Whitespace is dropped.
 * - A run of other characters (between matches and whitespace) becomes one `<unk>`: lowercase
 *   `mkta` is a single `<unk>`, as in `EsmTokenizer`.
 * - Encoding adds `<cls>` before and `<eos>` after. decode() joins tokens with spaces.
 *
 * @throws std::invalid_argument if the vocabulary lacks `<cls>`, `<eos>` or `<unk>`, or a token
 *         repeats.
 */
[[nodiscard]] TextTokenizer MakeEsmTokenizer(const std::vector<std::string>& vocab);
/** @brief MakeEsmTokenizer on a `vocab.txt`. @throws std::runtime_error if it can't be read. */
[[nodiscard]] TextTokenizer LoadEsmTokenizer(const std::string& vocab_path);

/** @brief One FASTA record. */
struct FastaRecord {
    /** @brief The header's first word, without the `>`. */
    std::string id;
    /** @brief The rest of the header line, after the first run of whitespace. */
    std::string description;
    /** @brief The sequence lines joined, with whitespace removed. Case and characters are kept
     *         as written, including a terminal `*`. */
    std::string sequence;
};

/**
 * @brief Parses FASTA text: records start with `>`, sequences may span lines, and blank lines and
 *        `;` comment lines are skipped. Line endings may be `\n` or `\r\n`.
 * @throws std::invalid_argument for sequence text before the first header, or an empty header.
 */
[[nodiscard]] std::vector<FastaRecord> ParseFasta(std::string_view text);
/** @brief ParseFasta on a file. @throws std::runtime_error if it can't be read. */
[[nodiscard]] std::vector<FastaRecord> ReadFasta(const std::string& path);
/** @brief FASTA text for @p records, sequences wrapped at @p width residues (0: one line). */
[[nodiscard]] std::string WriteFasta(const std::vector<FastaRecord>& records, size_t width = 60);

}  // namespace pulsatrix
