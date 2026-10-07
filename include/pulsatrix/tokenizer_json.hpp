/** @file tokenizer_json.hpp
 *  @brief Loads a Hugging Face `tokenizer.json` into a TextTokenizer (TOK-2).
 *  @ingroup data_pipeline
 */
#pragma once

#include <string>
#include <string_view>

#include "pulsatrix/text_tokenizer.hpp"

namespace pulsatrix {

/**
 * @brief Builds the tokenizer a `tokenizer.json` describes.
 *
 * Supported components (anything else is refused by name, never approximated):
 * - normalizers: `NFC`, `Replace`, `Prepend`, `Sequence`
 * - pre-tokenizers: `Split` (regex or string pattern, every behavior, `invert`), `Digits`,
 *   `ByteLevel`, `Metaspace`, `Sequence`
 * - models: `BPE` (merges as `"a b"` strings or `["a", "b"]` pairs, `ignore_merges`, subword
 *   prefix and suffix, `unk_token`, `byte_fallback`, `fuse_unk`; not `dropout`), `WordLevel`
 * - post-processors: `ByteLevel`, `TemplateProcessing` (single sequences), `Sequence`
 * - decoders: `ByteLevel`, `ByteFallback`, `Fuse`, `Replace`, `Strip`, `Metaspace`, `Sequence`
 * - added tokens without `lstrip`, `rstrip` or `single_word`, and with `normalized` only when
 *   there is no normalizer
 *
 * @throws std::invalid_argument for malformed JSON or an unsupported component or option.
 */
[[nodiscard]] TextTokenizer ParseTokenizerJson(std::string_view json);

/** @brief ParseTokenizerJson on a file. @throws std::runtime_error if it can't be read. */
[[nodiscard]] TextTokenizer LoadTokenizerJson(const std::string& path);

}  // namespace pulsatrix
