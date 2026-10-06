/** @file unicode_regex.hpp
 *  @brief A small backtracking regex engine over Unicode code points, for the pre-tokenizer
 *         patterns in `tokenizer.json` files (TOK-2). `std::regex` can't match Unicode
 *         categories.
 *  @ingroup data_pipeline
 */
#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace pulsatrix {

/**
 * @brief A compiled pattern, matched leftmost-first with backtracking as Oniguruma (which Hugging
 *        Face `tokenizers` uses) and Perl do.
 *
 * Supported syntax:
 * - alternation `|`; groups `(...)`, `(?:...)`, `(?i:...)` (case-insensitive), lookaheads `(?=...)`
 *   and `(?!...)`
 * - quantifiers `?`, `*`, `+`, `{n}`, `{n,}`, `{n,m}`, each optionally lazy (`??`, `*?`, ...)
 * - `.` (any character but `\n`), literals, character classes `[...]` and `[^...]` with ranges
 * - `\p{X}` and `\P{X}` for general categories (`L`, `Lu`, `Ll`, `Lt`, `Lm`, `Lo`, `M`, `Mn`, `N`,
 *   `Nd`, `P`, `S`, `Z`, `C`, ... and `L&`); `\s` (White_Space), `\d` (`Nd`), `\w` (letters, marks,
 *   `Nd` and `Pc`) and their negations; `\r`, `\n`, `\t`, `\f`, `\v`, `\0`, `\xHH`, `\x{H...}`,
 *   `\uHHHH`, and escaped punctuation
 *
 * Case-insensitive matching folds ASCII letters, plus U+017F (long s) and U+212A (Kelvin sign),
 * which is all the contraction patterns (`(?i:'s|'t|...)`) can meet.
 */
class UnicodeRegex {
public:
    /** @throws std::invalid_argument for a syntax error or unsupported syntax, naming the position. */
    explicit UnicodeRegex(std::string_view pattern);
    ~UnicodeRegex();
    UnicodeRegex(UnicodeRegex&&) noexcept;
    UnicodeRegex& operator=(UnicodeRegex&&) noexcept;

    /** @brief A match: a byte range `[begin, end)` of the subject. */
    struct Match {
        size_t begin = 0;
        size_t end = 0;
    };

    /**
     * @brief Every non-overlapping match in valid UTF-8 @p text, scanning left to right. Empty
     *        matches are skipped.
     * @throws std::invalid_argument if @p text isn't valid UTF-8.
     */
    [[nodiscard]] std::vector<Match> find_all(std::string_view text) const;

    /** @brief The pattern as given. */
    [[nodiscard]] const std::string& pattern() const { return pattern_; }

private:
    struct Program;
    std::string pattern_;
    std::unique_ptr<Program> program_;
};

}  // namespace pulsatrix
