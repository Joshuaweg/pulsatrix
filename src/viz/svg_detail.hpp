// Shared layout helpers for the SVG renderers (svg.cpp and svg_whatif.cpp). Private to src/.
#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/viz/colormap.hpp"
#include "pulsatrix/viz/plot_data.hpp"
#include "pulsatrix/viz/svg.hpp"
#include "../unicode.hpp"
#include "../utf8.hpp"

namespace pulsatrix {
namespace svg_detail {

inline constexpr const char* kTextColor = "#222222";
inline constexpr const char* kMutedColor = "#666666";
inline constexpr const char* kAxisColor = "#999999";
inline constexpr const char* kMissingColor = "#bdbdbd";
inline constexpr const char* kSansFont = "Helvetica, Arial, sans-serif";
inline constexpr const char* kMonoFont = "DejaVu Sans Mono, Menlo, Consolas, monospace";
// No font library, so text widths are estimated. 0.6 em is a monospace advance (DejaVu Sans
// Mono's is 0.602 em) and a safe upper bound for most sans-serif text.
inline constexpr double kCharWidthEm = 0.6;
inline constexpr int64_t kMaxVectorCells = 4096;

// ---- text ----------------------------------------------------------------------------------

// The length of the valid UTF-8 sequence at s[i], or 0.
inline size_t Utf8Length(std::string_view s, size_t i) {
    auto byte = [&](size_t k) { return static_cast<unsigned char>(s[k]); };
    auto cont = [&](size_t k) { return k < s.size() && (byte(k) & 0xC0) == 0x80; };
    unsigned char c = byte(i);
    if (c < 0x80) return 1;
    if (c >= 0xC2 && c <= 0xDF) return cont(i + 1) ? 2 : 0;
    if (c >= 0xE0 && c <= 0xEF) {
        if (!cont(i + 1) || !cont(i + 2)) return 0;
        unsigned char c1 = byte(i + 1);
        return ((c == 0xE0 && c1 < 0xA0) || (c == 0xED && c1 >= 0xA0)) ? 0 : 3;
    }
    if (c >= 0xF0 && c <= 0xF4) {
        if (!cont(i + 1) || !cont(i + 2) || !cont(i + 3)) return 0;
        unsigned char c1 = byte(i + 1);
        return ((c == 0xF0 && c1 < 0x90) || (c == 0xF4 && c1 >= 0x90)) ? 0 : 4;
    }
    return 0;
}

// Splits @p s into characters (code points), with every byte XML can't hold -- invalid UTF-8 and
// control characters -- replaced by U+FFFD.
inline std::vector<std::string> Characters(std::string_view s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t n = Utf8Length(s, i);
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (n == 0 || (n == 1 && c < 0x20) || c == 0x7F) {
            out.emplace_back("\xEF\xBF\xBD");
            i += (n == 0) ? 1 : n;
            continue;
        }
        out.emplace_back(s.substr(i, n));
        i += n;
    }
    return out;
}

inline std::string Escape(std::string_view s) {
    std::string out;
    for (const std::string& ch : Characters(s)) {
        if (ch == "&") out += "&amp;";
        else if (ch == "<") out += "&lt;";
        else if (ch == ">") out += "&gt;";
        else if (ch == "\"") out += "&quot;";
        else if (ch == "'") out += "&apos;";
        else out += ch;
    }
    return out;
}

// @p s cut to at most @p max_chars characters, ending in an ellipsis when cut.
// How many monospace columns a character takes, as terminals count them (wcwidth): 0 for
// combining marks, format characters (zero-width joiner, soft hyphen) and Hangul's combining
// jamo, 2 for East Asian wide and fullwidth characters and emoji, 1 otherwise. Text widths are
// estimated from this, so CJK labels get twice the room of Latin ones and accents none.
inline size_t Columns(const std::string& ch) {
    const char32_t c = utf8::Decode(ch, 0).value;
    const unicode::Category cat = unicode::GetCategory(c);
    if (cat == unicode::Category::Mn || cat == unicode::Category::Me || cat == unicode::Category::Cf ||
        (c >= 0x1160 && c <= 0x11FF)) {
        return 0;
    }
    const bool wide = (c >= 0x1100 && c <= 0x115F) || c == 0x2329 || c == 0x232A || (c >= 0x2E80 && c <= 0xA4CF && c != 0x303F) ||
                      (c >= 0xAC00 && c <= 0xD7A3) || (c >= 0xF900 && c <= 0xFAFF) || (c >= 0xFE10 && c <= 0xFE19) ||
                      (c >= 0xFE30 && c <= 0xFE6F) || (c >= 0xFF00 && c <= 0xFF60) || (c >= 0xFFE0 && c <= 0xFFE6) ||
                      (c >= 0x1F300 && c <= 0x1F64F) || (c >= 0x1F680 && c <= 0x1F6FF) || (c >= 0x1F900 && c <= 0x1F9FF) ||
                      (c >= 0x20000 && c <= 0x3FFFD);
    return wide ? 2 : 1;
}

inline size_t Columns(std::string_view s) {
    size_t n = 0;
    for (const std::string& ch : Characters(s)) n += Columns(ch);
    return n;
}

// At most max_columns columns of s, ending in an ellipsis when cut.
inline std::string Truncate(std::string_view s, size_t max_columns) {
    std::vector<std::string> chars = Characters(s);
    std::string out;
    size_t used = 0;
    for (size_t i = 0; i < chars.size(); ++i) {
        used += Columns(chars[i]);
        if (used > max_columns) break;
        out += chars[i];
        if (i + 1 == chars.size()) return out;
    }
    if (Columns(std::string_view(out)) == Columns(s)) return out;
    // Cut: drop characters until the ellipsis (one column) fits.
    std::vector<std::string> kept = Characters(out);
    while (!kept.empty() && Columns(std::string_view(out)) + 1 > max_columns) {
        kept.pop_back();
        out.clear();
        for (const auto& c : kept) out += c;
    }
    return out + "\xE2\x80\xA6";
}

inline double TextWidth(std::string_view s, double font_size) {
    return static_cast<double>(Columns(s)) * kCharWidthEm * font_size;
}

// ---- numbers -------------------------------------------------------------------------------

// A coordinate: at most two decimals, trailing zeros dropped, locale-independent.
inline std::string Num(double v) {
    if (std::abs(v) < 0.005) {
        return "0";
    }
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::fixed, 2);
    std::string s(buf, r.ptr);
    s.erase(s.find_last_not_of('0') + 1);
    if (s.back() == '.') {
        s.pop_back();
    }
    return s;
}

// A data value for a label: three significant digits.
inline std::string ValueText(double v) {
    if (v == 0.0) {
        return "0";
    }
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::general, 3);
    return std::string(buf, r.ptr);
}

inline std::string Hex(RgbColor c) {
    auto byte = [](float v) { return static_cast<int>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
    char buf[8];
    std::snprintf(buf, sizeof buf, "#%02x%02x%02x", byte(c.r), byte(c.g), byte(c.b));
    return buf;
}

// Text that reads well on @p c: dark on light colors, white on dark ones.
inline const char* TextOn(RgbColor c) {
    double luminance = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    return luminance > 0.5 ? kTextColor : "#ffffff";
}

// Tick values at 1, 2 or 5 times a power of ten covering [lo, hi], about @p target of them.
inline std::vector<double> NiceTicks(double lo, double hi, int target = 5) {
    if (!(hi > lo) || !std::isfinite(hi - lo)) {
        return {lo};
    }
    double raw = (hi - lo) / target;
    double magnitude = std::pow(10.0, std::floor(std::log10(raw)));
    double step = magnitude;
    for (double m : {2.0, 5.0, 10.0}) {
        if (std::abs(m * magnitude - raw) < std::abs(step - raw)) {
            step = m * magnitude;
        }
    }
    // Count ticks by index, not by adding step to a double: when the values are huge next to
    // their spread, t + step == t and the loop would never end.
    const double first = std::ceil(lo / step - 1e-9);
    const double last = std::floor(hi / step + 1e-9);
    if (!std::isfinite(first) || !std::isfinite(last) || last - first > 50.0 || last < first) {
        return {lo, hi};
    }
    std::vector<double> ticks;
    for (double k = first; k <= last; k += 1.0) {
        const double t = k * step;
        ticks.push_back(std::abs(t) < step * 1e-9 ? 0.0 : t);
    }
    return ticks;
}

// [lo, hi] always containing zero, padded so marks don't touch the edges.
inline std::pair<double, double> ZeroAnchoredDomain(double min_v, double max_v, double pad = 0.05) {
    double lo = std::min(0.0, min_v);
    double hi = std::max(0.0, max_v);
    if (hi - lo == 0.0) {
        hi = 1.0;
    }
    double span = hi - lo;
    return {lo < 0 ? lo - span * pad : lo, hi > 0 ? hi + span * pad : hi};
}

// ---- document --------------------------------------------------------------------------------

inline void CheckOptions(const SvgOptions& o) {
    if (o.width < 200) {
        throw std::invalid_argument("SVG width must be at least 200 pixels");
    }
    if (o.font_size < 1 || o.font_size > 200) {
        throw std::invalid_argument("SVG font_size must be in [1, 200]");
    }
}

// Collects the body of one figure, then wraps it in the <svg> element once the height is known.
class Figure {
public:
    Figure(const SvgOptions& o, std::string accessible_name) : options_(o), fs_(o.font_size) {
        name_ = o.title.empty() ? std::move(accessible_name) : o.title;
        top_ = fs_;
        if (!o.title.empty()) {
            body_ += "<text class=\"chart-title\" x=\"" + Num(fs_) + "\" y=\"" + Num(fs_ * 1.5) + "\" font-size=\"" +
                     Num(fs_ * 1.25) + "\" font-weight=\"bold\" fill=\"" + kTextColor + "\">" + Escape(o.title) +
                     "</text>\n";
            top_ = fs_ * 2.6;
        }
    }

    double fs() const { return fs_; }
    double width() const { return options_.width; }
    double top() const { return top_; }
    std::string& body() { return body_; }

    void line(const char* cls, double x1, double y1, double x2, double y2, const char* stroke, double width = 1.0,
              const char* dash = nullptr) {
        body_ += std::string("<line class=\"") + cls + "\" x1=\"" + Num(x1) + "\" y1=\"" + Num(y1) + "\" x2=\"" + Num(x2) +
                 "\" y2=\"" + Num(y2) + "\" stroke=\"" + stroke + "\" stroke-width=\"" + Num(width) + "\"";
        if (dash != nullptr) {
            body_ += std::string(" stroke-dasharray=\"") + dash + "\"";
        }
        body_ += "/>\n";
    }

    void text(const char* cls, double x, double y, std::string_view s, const char* anchor = "start",
              const char* fill = kTextColor, const std::string& extra = "") {
        body_ += std::string("<text class=\"") + cls + "\" x=\"" + Num(x) + "\" y=\"" + Num(y) + "\" text-anchor=\"" +
                 anchor + "\" fill=\"" + fill + "\"" + extra + ">" + Escape(s) + "</text>\n";
    }

    // Ticks and labels for a horizontal value axis at @p y, mapping [lo, hi] to [x0, x1].
    void x_axis(double y, double x0, double x1, double lo, double hi, const std::string& caption) {
        line("axis", x0, y, x1, y, kAxisColor);
        for (double t : NiceTicks(lo, hi)) {
            double x = x0 + (t - lo) / (hi - lo) * (x1 - x0);
            line("tick", x, y, x, y + fs_ * 0.35, kAxisColor);
            text("tick-label", x, y + fs_ * 1.35, ValueText(t), "middle", kMutedColor, " font-size=\"" + Num(fs_ * 0.85) + "\"");
        }
        if (!caption.empty()) {
            text("axis-label", (x0 + x1) / 2, y + fs_ * 2.6, caption, "middle", kMutedColor);
        }
    }

    // Ticks and labels for a vertical value axis at @p x, mapping [lo, hi] to [y_bottom, y_top].
    // The caption is drawn rotated, left of the tick labels; @p label_w is their width.
    void y_axis(double x, double y_top, double y_bottom, double lo, double hi, const std::string& caption,
                double label_w) {
        line("axis", x, y_top, x, y_bottom, kAxisColor);
        for (double t : NiceTicks(lo, hi)) {
            double y = y_bottom - (t - lo) / (hi - lo) * (y_bottom - y_top);
            line("tick", x - fs_ * 0.35, y, x, y, kAxisColor);
            text("tick-label", x - fs_ * 0.5, y + fs_ * 0.3, ValueText(t), "end", kMutedColor,
                 " font-size=\"" + Num(fs_ * 0.85) + "\"");
        }
        if (!caption.empty()) {
            const double cx = x - label_w - fs_ * 1.2;
            const double cy = (y_top + y_bottom) / 2;
            text("axis-label", cx, cy, caption, "middle", kMutedColor,
                 " transform=\"rotate(-90 " + Num(cx) + " " + Num(cy) + ")\"");
        }
    }

    // @p width_used, when given, narrows the figure to its content (never wider than the option).
    std::string finish(double height, double width_used = 0.0) const {
        const double w = width_used > 0.0 ? std::min<double>(options_.width, std::ceil(width_used)) : options_.width;
        std::string out = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        out += "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" + Num(w) + "\" height=\"" +
               Num(std::ceil(height)) + "\" viewBox=\"0 0 " + Num(w) + " " + Num(std::ceil(height)) +
               "\" role=\"img\" font-family=\"" + kSansFont + "\" font-size=\"" + Num(fs_) + "\">\n";
        out += "<title>" + Escape(name_) + "</title>\n";
        out += "<rect class=\"background\" width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n";
        out += body_;
        out += "</svg>\n";
        return out;
    }

private:
    SvgOptions options_;
    double fs_;
    double top_;
    std::string name_;
    std::string body_;
};

inline std::vector<std::string> FeatureNames(const AttributionDocument& doc, size_t count) {
    std::vector<std::string> names;
    auto it = doc.metadata.find("feature_names");
    if (it != doc.metadata.end()) {
        std::string_view s = it->second;
        size_t start = 0;
        while (true) {
            size_t comma = s.find(',', start);
            names.emplace_back(s.substr(start, comma == std::string_view::npos ? std::string_view::npos : comma - start));
            if (comma == std::string_view::npos) break;
            start = comma + 1;
        }
    }
    for (size_t i = names.size(); i < count; ++i) {
        names.push_back("feature_" + std::to_string(i));
    }
    return names;
}

inline void CheckFinite(const std::vector<float>& values, const char* what) {
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            throw std::invalid_argument(std::string(what) + ": value " + std::to_string(i) + " is not finite");
        }
    }
}

inline Attribution ToHostAttribution(const AttributionDocument& doc) {
    static CPUBackend backend;
    return ToAttribution(doc, &backend);
}

// A vertical color bar for [lo, hi] at (x, y), @p h tall, sampled from @p color_at(t in [0,1]).
template <typename ColorAt>
inline void ColorBar(Figure& f, double x, double y, double h, double lo, double hi, ColorAt color_at, const std::string& id) {
    const double w = f.fs();
    std::string& b = f.body();
    b += "<g class=\"colorbar\">\n<defs><linearGradient id=\"" + id + "\" x1=\"0\" y1=\"1\" x2=\"0\" y2=\"0\">\n";
    constexpr int kStops = 11;
    for (int i = 0; i < kStops; ++i) {
        double t = static_cast<double>(i) / (kStops - 1);
        b += "<stop offset=\"" + Num(t) + "\" stop-color=\"" + Hex(color_at(static_cast<float>(t))) + "\"/>\n";
    }
    b += "</linearGradient></defs>\n";
    b += "<rect x=\"" + Num(x) + "\" y=\"" + Num(y) + "\" width=\"" + Num(w) + "\" height=\"" + Num(h) + "\" fill=\"url(#" +
         id + ")\" stroke=\"" + kAxisColor + "\" stroke-width=\"0.5\"/>\n";
    for (double t : NiceTicks(lo, hi, 4)) {
        double ty = y + h - (t - lo) / (hi - lo) * h;
        f.line("tick", x + w, ty, x + w + f.fs() * 0.3, ty, kAxisColor);
        f.text("tick-label", x + w + f.fs() * 0.5, ty + f.fs() * 0.35, ValueText(t), "start", kMutedColor,
               " font-size=\"" + Num(f.fs() * 0.85) + "\"");
    }
    b += "</g>\n";
}

inline std::string Base64(const std::vector<unsigned char>& bytes) {
    static const char* kAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    for (size_t i = 0; i < bytes.size(); i += 3) {
        uint32_t chunk = static_cast<uint32_t>(bytes[i]) << 16;
        if (i + 1 < bytes.size()) chunk |= static_cast<uint32_t>(bytes[i + 1]) << 8;
        if (i + 2 < bytes.size()) chunk |= bytes[i + 2];
        out += kAlphabet[(chunk >> 18) & 63];
        out += kAlphabet[(chunk >> 12) & 63];
        out += (i + 1 < bytes.size()) ? kAlphabet[(chunk >> 6) & 63] : '=';
        out += (i + 2 < bytes.size()) ? kAlphabet[chunk & 63] : '=';
    }
    return out;
}

// A `data:image/png;base64,...` URI for @p width x @p height RGB pixels, row-major (svg.cpp).
// @throws std::invalid_argument if the image can't be encoded.
std::string PngDataUri(const std::vector<unsigned char>& rgb, int64_t width, int64_t height);

}  // namespace svg_detail
}  // namespace pulsatrix
