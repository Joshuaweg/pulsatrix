#include "pulsatrix/viz/fonts.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <string_view>
#include <utility>

#include <imgui.h>
#ifdef IMGUI_ENABLE_FREETYPE
#include <ft2build.h>
#include FT_FREETYPE_H
#include <misc/freetype/imgui_freetype.h>
#endif
#ifdef PULSATRIX_VIZ_WITH_FONTCONFIG
#include <fontconfig/fontconfig.h>
#endif

namespace pulsatrix {

namespace {

namespace fs = std::filesystem;

/** @brief Whether a file is a bitmap color-emoji font, which only FreeType can draw. */
bool IsColorEmojiFont(const std::string& path) {
    const std::string name = fs::path(path).filename().string();
    return name.find("ColorEmoji") != std::string::npos || name.find("Color Emoji") != std::string::npos ||
           name.find("Twemoji") != std::string::npos;
}

/** @brief Whether a font is an emoji font: it then supplies the emoji blocks, which the others leave out. */
bool IsEmojiFont(const std::string& path) {
    const std::string name = fs::path(path).filename().string();
    return name.find("Emoji") != std::string::npos || name.find("emoji") != std::string::npos ||
           name.find("seguiemj") != std::string::npos;
}

// The emoji and pictograph blocks (U+1F000 to U+1FAFF). Text fonts that hold a few of them in
// black and white leave them to the emoji font, so all emoji look alike.
const ImWchar kEmojiBlocks[] = {0x1F000, 0x1FAFF, 0};

/**
 * @brief Whether ImGui can draw the font at any size. Bitmap-only fonts (Noto Color Emoji's CBDT,
 *        Apple Color Emoji's sbix) hold one fixed size, which ImGui's loaders can't scale, so their
 *        glyphs would never appear; vector color fonts (COLR, as in Segoe UI Emoji and Twemoji) work.
 */
bool Scalable(const VizFontFile& f) {
#ifdef IMGUI_ENABLE_FREETYPE
    FT_Library library = nullptr;
    if (FT_Init_FreeType(&library) != 0) return false;
    FT_Face face = nullptr;
    bool scalable = false;
    if (FT_New_Face(library, f.path.c_str(), f.index, &face) == 0) {
        scalable = FT_IS_SCALABLE(face);
        FT_Done_Face(face);
    }
    FT_Done_FreeType(library);
    return scalable;
#else
    return !IsColorEmojiFont(f.path);  // stb_truetype reads outline fonts only
#endif
}

bool Usable(const VizFontFile& f) {
    std::error_code ec;
    return fs::is_regular_file(f.path, ec) && Scalable(f);
}

void AddUnique(std::vector<VizFontFile>& out, std::set<std::pair<std::string, int>>& seen, VizFontFile f) {
    if (!Usable(f)) return;
    if (seen.insert({f.path, f.index}).second) out.push_back(std::move(f));
}

#ifdef PULSATRIX_VIZ_WITH_FONTCONFIG
// One character per script (and symbols, math, emoji), so each script gets its best font.
constexpr char32_t kSamples[] = {
    U'A',     U'\u03B1', U'\u0416', U'\u0531', U'\u05D0', U'\u0628', U'\u0710', U'\u0789', U'\u07CA', U'\u0915',
    U'\u0995', U'\u0A15', U'\u0A95', U'\u0B15', U'\u0B95', U'\u0C15', U'\u0C95', U'\u0D15', U'\u0D9A', U'\u0E01',
    U'\u0E81', U'\u0F40', U'\u1000', U'\u10D0', U'\u1200', U'\u13A0', U'\u1401', U'\u1780', U'\u1820', U'\u2D30',
    U'\u3042', U'\u30A2', U'\u4E2D', U'\uAC00', U'\uA000', U'\uA984', U'\u2211', U'\u2192', U'\u2500', U'\u263A',
    U'\u2713', U'\U0001D400', U'\U0001D11E',
};

// Emoji are matched against fontconfig's "emoji" family, so a color emoji font wins over text
// fonts that happen to hold a few black-and-white emoticons.
constexpr char32_t kEmojiSamples[] = {U'\U0001F600', U'\U0001F680', U'\U0001F44D', U'\U0001F3FD', U'\U0001F1FA'};

void MatchSample(FcConfig* config, const char* family, char32_t cp, std::vector<VizFontFile>& out,
                 std::set<std::pair<std::string, int>>& seen) {
    {
        FcPattern* pattern = FcNameParse(reinterpret_cast<const FcChar8*>(family));
        FcCharSet* chars = FcCharSetCreate();
        FcCharSetAddChar(chars, static_cast<FcChar32>(cp));
        FcPatternAddCharSet(pattern, FC_CHARSET, chars);
        FcConfigSubstitute(config, pattern, FcMatchPattern);
        FcDefaultSubstitute(pattern);
        FcResult result = FcResultNoMatch;
        FcPattern* match = FcFontMatch(config, pattern, &result);
        if (match != nullptr) {
            FcCharSet* has = nullptr;
            FcChar8* file = nullptr;
            int index = 0;
            if (FcPatternGetCharSet(match, FC_CHARSET, 0, &has) == FcResultMatch && FcCharSetHasChar(has, static_cast<FcChar32>(cp)) &&
                FcPatternGetString(match, FC_FILE, 0, &file) == FcResultMatch) {
                FcPatternGetInteger(match, FC_INDEX, 0, &index);
                AddUnique(out, seen, {reinterpret_cast<const char*>(file), index});
            }
            FcPatternDestroy(match);
        }
        FcCharSetDestroy(chars);
        FcPatternDestroy(pattern);
    }
}

std::vector<VizFontFile> QueryFontconfig() {
    std::vector<VizFontFile> out;
    std::set<std::pair<std::string, int>> seen;
    FcConfig* config = FcInitLoadConfigAndFonts();
    if (config == nullptr) return out;
    for (char32_t cp : kSamples) MatchSample(config, "sans-serif", cp, out, seen);
    for (char32_t cp : kEmojiSamples) MatchSample(config, "emoji", cp, out, seen);
    FcConfigDestroy(config);
    return out;
}
#endif

/** @brief The first existing file named in @p names under any of @p dirs (searched recursively). */
std::vector<VizFontFile> FindNamed(const std::vector<fs::path>& dirs, const std::vector<std::vector<std::string>>& groups) {
    std::vector<VizFontFile> out;
    std::set<std::pair<std::string, int>> seen;
    // Index the directories once: file name -> path.
    std::vector<std::pair<std::string, std::string>> files;
    for (const fs::path& d : dirs) {
        std::error_code ec;
        if (!fs::is_directory(d, ec)) continue;
        for (fs::recursive_directory_iterator it(d, fs::directory_options::skip_permission_denied, ec), end; it != end;
             it.increment(ec)) {
            if (ec) break;
            if (it->is_regular_file(ec)) files.emplace_back(it->path().filename().string(), it->path().string());
        }
    }
    for (const auto& group : groups) {  // the first name of each group that exists
        for (const std::string& name : group) {
            const auto hit = std::find_if(files.begin(), files.end(), [&](const auto& f) { return f.first == name; });
            if (hit != files.end()) {
                AddUnique(out, seen, {hit->second, 0});
                break;
            }
        }
    }
    return out;
}

std::vector<VizFontFile> CuratedFonts() {
#if defined(_WIN32)
    const char* windir = std::getenv("WINDIR");
    const fs::path fonts = fs::path(windir != nullptr ? windir : "C:\\Windows") / "Fonts";
    return FindNamed({fonts}, {{"segoeui.ttf"},  {"msyh.ttc", "msyh.ttf"}, {"msjh.ttc"}, {"YuGothM.ttc", "meiryo.ttc"},
                               {"malgun.ttf"},   {"Nirmala.ttc", "Nirmala.ttf"}, {"LeelawUI.ttf"}, {"ebrima.ttf"},
                               {"himalaya.ttf"}, {"mmrtext.ttf"}, {"seguisym.ttf"}, {"seguihis.ttf"}, {"seguiemj.ttf"},
                               {"cambria.ttc"}});
#elif defined(__APPLE__)
    return FindNamed({"/System/Library/Fonts", "/Library/Fonts"},
                     {{"Helvetica.ttc"}, {"PingFang.ttc", "Hiragino Sans GB.ttc"}, {"AppleSDGothicNeo.ttc"},
                      {"Arial Unicode.ttf"}, {"GeezaPro.ttc"}, {"Kohinoor.ttc", "Devanagari Sangam MN.ttc"},
                      {"Thonburi.ttc"}, {"STIXTwoMath-Regular.otf"}, {"Apple Symbols.ttf"}, {"Apple Color Emoji.ttc"}});
#else
    std::vector<fs::path> dirs = {"/usr/share/fonts", "/usr/local/share/fonts"};
    if (const char* home = std::getenv("HOME")) {
        dirs.push_back(fs::path(home) / ".local/share/fonts");
        dirs.push_back(fs::path(home) / ".fonts");
    }
    return FindNamed(dirs, {{"NotoSans-Regular.ttf", "DejaVuSans.ttf"},
                            {"NotoSansCJK-Regular.ttc", "NotoSansCJKsc-Regular.otf", "wqy-microhei.ttc", "DroidSansFallbackFull.ttf"},
                            {"NotoSansArabic-Regular.ttf", "NotoNaskhArabic-Regular.ttf"},
                            {"NotoSansHebrew-Regular.ttf"},
                            {"NotoSansDevanagari-Regular.ttf"},
                            {"NotoSansBengali-Regular.ttf"},
                            {"NotoSansTamil-Regular.ttf"},
                            {"NotoSansThai-Regular.ttf"},
                            {"NotoSansGeorgian-Regular.ttf"},
                            {"NotoSansArmenian-Regular.ttf"},
                            {"NotoSansEthiopic-Regular.ttf"},
                            {"NotoSansSymbols-Regular.ttf"},
                            {"NotoSansSymbols2-Regular.ttf"},
                            {"NotoSansMath-Regular.ttf"},
                            {"NotoColorEmoji.ttf", "NotoEmoji-Regular.ttf"},
                            {"DejaVuSans.ttf"}});
#endif
}

}  // namespace

bool VizFontsUseFreeType() {
#ifdef IMGUI_ENABLE_FREETYPE
    return true;
#else
    return false;
#endif
}

std::vector<VizFontFile> FindFallbackFonts() {
#ifdef PULSATRIX_VIZ_WITH_FONTCONFIG
    std::vector<VizFontFile> found = QueryFontconfig();
    if (!found.empty()) return found;
#endif
    return CuratedFonts();
}

VizFontOptions DefaultVizFontOptions() {
    VizFontOptions options;
    const char* env = std::getenv("PULSATRIX_VIZ_FONTS");
    if (env == nullptr || *env == '\0') return options;
    const std::string_view list = env;
    if (list == "none") {
        options.system_fallbacks = false;
        return options;
    }
#ifdef _WIN32
    constexpr char kSeparator = ';';
#else
    constexpr char kSeparator = ':';
#endif
    size_t start = 0;
    while (start <= list.size()) {
        size_t end = list.find_first_of(std::string(1, kSeparator) + ";", start);
        if (end == std::string_view::npos) end = list.size();
        if (end > start) options.fonts.push_back({std::string(list.substr(start, end - start)), 0});
        start = end + 1;
    }
    return options;
}

std::vector<VizFontFile> LoadVizFonts(const VizFontOptions& options) {
    ImGuiIO& io = ImGui::GetIO();
    ImFontConfig base;
    base.SizePixels = options.size;
    io.Fonts->AddFontDefault(&base);
#ifdef IMGUI_ENABLE_FREETYPE
    io.Fonts->FontLoaderFlags |= ImGuiFreeTypeLoaderFlags_LoadColor;  // color emoji
#endif
    std::vector<VizFontFile> wanted = options.fonts;
    if (options.system_fallbacks) {
        for (VizFontFile& f : FindFallbackFonts()) wanted.push_back(std::move(f));
    }
    std::vector<VizFontFile> usable;
    std::set<std::pair<std::string, int>> seen;
    bool have_emoji_font = false;
    for (const VizFontFile& f : wanted) {
        if (!seen.insert({f.path, f.index}).second || !Usable(f)) continue;
        usable.push_back(f);
        have_emoji_font = have_emoji_font || IsEmojiFont(f.path);
    }
    std::vector<VizFontFile> loaded;
    for (const VizFontFile& f : usable) {
        ImFontConfig cfg;
        cfg.MergeMode = true;  // a fallback: used only for glyphs the fonts before it lack
        cfg.FontNo = static_cast<ImU32>(f.index);
        // With an emoji font, text fonts leave the emoji blocks to it; without one, their
        // black-and-white emoji are better than none.
        if (have_emoji_font && !IsEmojiFont(f.path)) cfg.GlyphExcludeRanges = kEmojiBlocks;
        if (io.Fonts->AddFontFromFileTTF(f.path.c_str(), options.size, &cfg) != nullptr) loaded.push_back(f);
    }
    return loaded;
}

}  // namespace pulsatrix
