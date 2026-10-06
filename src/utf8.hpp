// UTF-8 decoding and encoding for the tokenizers (TOK-1). Private to the library.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace pulsatrix::utf8 {

/** @brief The code point starting at byte @p i of @p s, and its length in bytes. */
struct CodePoint {
    char32_t value = 0;
    size_t length = 0;
};

/**
 * @brief Decodes the code point at byte @p i.
 * @return {0, 0} for an invalid sequence: a stray continuation byte, a truncated sequence, an
 *         overlong encoding, a surrogate or a value above U+10FFFF.
 */
[[nodiscard]] inline CodePoint Decode(std::string_view s, size_t i) {
    const auto byte = [&](size_t k) { return static_cast<unsigned char>(s[k]); };
    const unsigned char b0 = byte(i);
    if (b0 < 0x80) return {b0, 1};
    size_t length = 0;
    char32_t cp = 0;
    if ((b0 & 0xE0) == 0xC0) {
        length = 2;
        cp = b0 & 0x1F;
    } else if ((b0 & 0xF0) == 0xE0) {
        length = 3;
        cp = b0 & 0x0F;
    } else if ((b0 & 0xF8) == 0xF0) {
        length = 4;
        cp = b0 & 0x07;
    } else {
        return {};
    }
    if (i + length > s.size()) return {};
    for (size_t k = 1; k < length; ++k) {
        if ((byte(i + k) & 0xC0) != 0x80) return {};
        cp = (cp << 6) | (byte(i + k) & 0x3F);
    }
    static constexpr char32_t kMin[] = {0, 0, 0x80, 0x800, 0x10000};
    if (cp < kMin[length] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return {};
    return {cp, length};
}

/** @brief Whether every byte of @p s belongs to a valid UTF-8 sequence. */
[[nodiscard]] inline bool IsValid(std::string_view s) {
    for (size_t i = 0; i < s.size();) {
        const CodePoint c = Decode(s, i);
        if (c.length == 0) return false;
        i += c.length;
    }
    return true;
}

/** @brief Appends @p cp to @p out as UTF-8. @throws std::invalid_argument for a surrogate or a value above U+10FFFF. */
inline void Append(std::string& out, char32_t cp) {
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) throw std::invalid_argument("utf8::Append: not a Unicode scalar value");
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

[[nodiscard]] inline std::string Encode(char32_t cp) {
    std::string out;
    Append(out, cp);
    return out;
}

}  // namespace pulsatrix::utf8
