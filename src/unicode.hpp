// Unicode character properties and NFC for the tokenizers (TOK-2), from src/unicode_data.inc.
// Private to the library.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace pulsatrix::unicode {

/** @brief General categories, in the order the generated table numbers them. */
enum class Category : uint8_t {
    Lu, Ll, Lt, Lm, Lo, Mn, Mc, Me, Nd, Nl, No, Pc, Pd, Ps, Pe, Pi, Pf, Po, Sm, Sc, Sk, So, Zs, Zl, Zp,
    Cc, Cf, Cs, Co, Cn
};

/** @brief The Unicode version the tables come from. */
[[nodiscard]] const char* Version();

[[nodiscard]] Category GetCategory(char32_t cp);
[[nodiscard]] inline bool IsLetter(char32_t cp) { return GetCategory(cp) <= Category::Lo; }
[[nodiscard]] inline bool IsMark(char32_t cp) {
    const Category c = GetCategory(cp);
    return c >= Category::Mn && c <= Category::Me;
}
[[nodiscard]] inline bool IsNumber(char32_t cp) {
    const Category c = GetCategory(cp);
    return c >= Category::Nd && c <= Category::No;
}
/** @brief The White_Space property, which is what Oniguruma's `\s` matches. */
[[nodiscard]] bool IsWhiteSpace(char32_t cp);
[[nodiscard]] uint8_t CombiningClass(char32_t cp);
/** @brief Unicode simple case folding, for the letters case-insensitive regex literals use:
 *         ASCII, plus U+017F (long s) to 's' and U+212A (Kelvin) to 'k'. Other code points are
 *         returned unchanged. */
[[nodiscard]] char32_t FoldCase(char32_t cp);

/** @brief One character of an NFC result, and the byte range of the input it came from. */
struct NfcChar {
    char32_t cp = 0;
    size_t source_begin = 0;
    size_t source_end = 0;
};

/**
 * @brief NFC (canonical decomposition, canonical reordering, canonical composition) of valid UTF-8
 *        @p text. A composed character's source is the union of its parts'.
 * @return Empty when @p text is already in NFC (no character it holds can change), so callers
 *         can skip rebuilding.
 */
[[nodiscard]] std::vector<NfcChar> Nfc(std::string_view text);

}  // namespace pulsatrix::unicode
