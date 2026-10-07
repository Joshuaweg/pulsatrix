/** @file fonts.hpp
 *  @brief Fonts for the ImGui windows that cover every script: the default font first, then system
 *         fonts merged in as fallbacks, so text in any language and emoji show instead of "?".
 *  @ingroup visualization
 */
#pragma once

#include <string>
#include <vector>

namespace pulsatrix {

/** @brief A font file, and the face to use when it holds several (`.ttc`). */
struct VizFontFile {
    std::string path;
    int index = 0;
};

/** @brief Which fonts a VizWindow loads. */
struct VizFontOptions {
    /** @brief Fonts merged after ImGui's default font, in priority order. */
    std::vector<VizFontFile> fonts;
    /** @brief Also merge the system fonts FindFallbackFonts() finds, after `fonts`. */
    bool system_fallbacks = true;
    /** @brief Font size in pixels (ImGui's default font is 13). */
    float size = 13.0f;
};

/**
 * @brief The system fonts that together cover as many scripts as this machine has fonts for, best
 *        first.
 *
 * - On Linux (with fontconfig), each of about forty sample characters, one per script plus
 *   symbols, math and emoji, is matched to the best installed font.
 * - On Windows and macOS, the standard fonts that ship with the system are used.
 * - Without fontconfig on Linux, common Noto, DejaVu and WenQuanYi files are looked for in the
 *   usual font directories.
 *
 * Fonts ImGui can't draw at every size are left out: bitmap-only color emoji fonts (Noto Color
 * Emoji, Apple Color Emoji) hold one fixed size. Vector color emoji fonts (Segoe UI Emoji on
 * Windows, Twemoji) are drawn in color when ImGui uses FreeType.
 */
[[nodiscard]] std::vector<VizFontFile> FindFallbackFonts();

/**
 * @brief The font options a VizWindow uses when given none. With the environment variable
 *        PULSATRIX_VIZ_FONTS set to font paths (separated by ';', or ':' outside Windows), those
 *        fonts come first. Set to "none", only ImGui's default font is used.
 */
[[nodiscard]] VizFontOptions DefaultVizFontOptions();

/**
 * @brief Loads ImGui's default font and merges @p options' fonts into it, in the current ImGui
 *        context. Since ImGui 1.92 glyphs are drawn on first use, so a merged font costs only
 *        its file in memory.
 * @return The fonts that loaded (missing or unreadable files are skipped).
 */
std::vector<VizFontFile> LoadVizFonts(const VizFontOptions& options);

/** @brief Whether this build's ImGui uses FreeType (color emoji) rather than stb_truetype. */
[[nodiscard]] bool VizFontsUseFreeType();

}  // namespace pulsatrix
