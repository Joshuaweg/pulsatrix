#include "pulsatrix/viz/protein_views.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

#include "pulsatrix/protein_contacts.hpp"
#include "html_detail.hpp"
#include "protein_detail.hpp"
#include "svg_detail.hpp"

namespace pulsatrix {

using namespace svg_detail;

namespace {

constexpr const char* kRight = "#2166ac";     // a predicted contact the structure has
constexpr const char* kWrong = "#d6604d";     // one it doesn't
constexpr const char* kContact = "#bdbdbd";   // a true contact
constexpr const char* kUnknown = "#eeeeee";   // a pair whose distance is unknown
constexpr int64_t kMaxMutationCells = 12000;  // above this, a mutation map is drawn as images

void Check(const std::string& problem, const char* who) {
    if (!problem.empty()) throw std::invalid_argument(std::string(who) + ": " + problem);
}

/** @brief A transform coefficient: four decimals, since glyph scales are small numbers. */
std::string Num4(double v) {
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::fixed, 4);
    std::string s(buf, r.ptr);
    s.erase(s.find_last_not_of('0') + 1);
    if (s.back() == '.') s.pop_back();
    return s == "-0" ? "0" : s;
}

RgbColor Mix(RgbColor a, RgbColor b, double t) {
    t = std::clamp(t, 0.0, 1.0);
    return {static_cast<float>(a.r + (b.r - a.r) * t), static_cast<float>(a.g + (b.g - a.g) * t), static_cast<float>(a.b + (b.b - a.b) * t)};
}

/** @brief White to dark blue, for predicted contact scores. */
RgbColor Blues(double t) { return Mix({1.0f, 1.0f, 1.0f}, {0.031f, 0.188f, 0.420f}, t); }

RgbColor Gray(const char* hex) {
    unsigned v = 0;
    std::from_chars(hex + 1, hex + 7, v, 16);
    return {static_cast<float>((v >> 16) & 255) / 255.0f, static_cast<float>((v >> 8) & 255) / 255.0f, static_cast<float>(v & 255) / 255.0f};
}

/** @brief The largest |v| over finite values, or 1 when there is none. */
double MaxAbs(const std::vector<float>& values) {
    double m = 0;
    for (float v : values) {
        if (std::isfinite(v)) m = std::max(m, std::abs(static_cast<double>(v)));
    }
    return m > 0 ? m : 1.0;
}

std::string Where(const std::string& sequence, int64_t i, int64_t first_position) {
    return std::string(1, sequence[static_cast<size_t>(i)]) + std::to_string(first_position + i);
}

/** @brief Positions per block: as many columns of @p cw as fit in @p avail, rounded down to tens. */
int64_t PerBlock(int64_t length, double avail, double cw) {
    int64_t n = std::max<int64_t>(1, static_cast<int64_t>(std::floor(avail / cw)));
    if (n >= 20) n -= n % 10;
    return std::min(n, length);
}

/** @brief Ticks and numbers above columns [from, from + count) at x0, every ten positions. */
void Ruler(Figure& f, double x0, double y, double cw, int64_t from, int64_t count, int64_t first_position) {
    const double fs = f.fs();
    for (int64_t k = 0; k < count; ++k) {
        const int64_t number = first_position + from + k;
        // Every tenth position, and each block's first unless a tenth is about to follow it.
        const bool first_of_block = k == 0 && ((number % 10 + 10) % 10 == 0 || (10 - (number % 10 + 10) % 10) > 3);
        if ((number % 10 + 10) % 10 != 0 && !first_of_block) continue;
        const double x = x0 + cw * (static_cast<double>(k) + 0.5);
        f.line("tick", x, y - fs * 0.3, x, y, kAxisColor);
        f.text("position", x, y - fs * 0.45, std::to_string(number), "middle", kMutedColor, " font-size=\"" + Num(fs * 0.8) + "\"");
    }
}

/** @brief The residues of [from, from + count), one letter per column, baseline at @p y. */
void SequenceRow(Figure& f, double x0, double y, double cw, const std::string& sequence, int64_t from, int64_t count,
                 const std::string& extra = "") {
    for (int64_t k = 0; k < count; ++k) {
        f.text("residue", x0 + cw * (static_cast<double>(k) + 0.5), y, std::string(1, sequence[static_cast<size_t>(from + k)]), "middle",
               kTextColor, " font-family=\"" + std::string(kMonoFont) + "\"" + extra);
    }
}

/** @brief A horizontal color scale @p w wide at (x, y), with @p lo, zero (when inside) and @p hi
 *         marked, and a caption to its left. */
template <typename ColorAt>
void Legend(Figure& f, double x, double y, double w, double lo, double hi, ColorAt color_at, const std::string& id,
            const std::string& caption) {
    const double fs = f.fs();
    std::string& b = f.body();
    if (!caption.empty()) {
        f.text("legend-label", x, y + fs * 0.75, caption, "start", kMutedColor, " font-size=\"" + Num(fs * 0.85) + "\"");
        x += TextWidth(caption, fs * 0.85) + fs * 0.6;
    }
    b += "<g class=\"legend\">\n<defs><linearGradient id=\"" + id + "\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"0\">\n";
    for (int i = 0; i <= 10; ++i) {
        const double t = i / 10.0;
        b += "<stop offset=\"" + Num(t) + "\" stop-color=\"" + Hex(color_at(t)) + "\"/>\n";
    }
    b += "</linearGradient></defs>\n<rect x=\"" + Num(x) + "\" y=\"" + Num(y) + "\" width=\"" + Num(w) + "\" height=\"" + Num(fs * 0.8) +
         "\" fill=\"url(#" + id + ")\" stroke=\"" + kAxisColor + "\" stroke-width=\"0.5\"/>\n";
    std::vector<double> marks = {lo, hi};
    if (lo < 0 && hi > 0) marks.insert(marks.begin() + 1, 0.0);
    for (double m : marks) {
        const double mx = x + (hi > lo ? (m - lo) / (hi - lo) : 0.0) * w;
        f.text("tick-label", mx, y + fs * 1.85, ValueText(m), "middle", kMutedColor, " font-size=\"" + Num(fs * 0.8) + "\"");
    }
    b += "</g>\n";
}

/** @brief The legend's width with its caption, for laying out a row of them. */
double LegendWidth(const std::string& caption, double bar_w, double fs) {
    return (caption.empty() ? 0.0 : TextWidth(caption, fs * 0.85) + fs * 0.6) + bar_w + fs * 2.0;
}

/** @brief @p text broken at spaces into lines at most @p width wide. */
std::vector<std::string> Wrap(const std::string& text, double width, double font_size) {
    std::vector<std::string> lines;
    std::string line;
    size_t start = 0;
    while (start <= text.size()) {
        size_t end = text.find(' ', start);
        if (end == std::string::npos) end = text.size();
        const std::string word = text.substr(start, end - start);
        const std::string joined = line.empty() ? word : line + " " + word;
        if (!line.empty() && TextWidth(joined, font_size) > width) {
            lines.push_back(line);
            line = word;
        } else {
            line = joined;
        }
        start = end + 1;
    }
    if (!line.empty()) lines.push_back(line);
    return lines;
}

std::string FigurePage(const std::string& heading, const std::string& svg, const std::string& note = "") {
    // The SVG file's XML declaration doesn't belong inside HTML.
    const size_t start = svg.find("<svg");
    std::string body = "<h1>" + Escape(heading) + "</h1>\n";
    if (!note.empty()) body += "<p class=\"muted\">" + Escape(note) + "</p>\n";
    body += "<figure>\n" + svg.substr(start) + "</figure>\n";
    return html_detail::Page(heading, "", body);
}

SvgOptions ToSvgOptions(const HtmlOptions& options) {
    SvgOptions o;
    o.width = options.width;
    return o;
}

}  // namespace

// ---- mutation map --------------------------------------------------------------------------

std::string RenderMutationMapSvg(const MutationMapDocument& doc, const SvgOptions& options) {
    CheckOptions(options);
    Check(protein_detail::MutationMapProblem(doc), "RenderMutationMapSvg");
    const auto L = static_cast<int64_t>(doc.sequence.size()), A = static_cast<int64_t>(doc.alphabet.size());
    const double max_abs = MaxAbs(doc.values);
    auto color_of = [&](float v) { return DivergingColormap(NormalizeSigned(v, static_cast<float>(max_abs))); };

    Figure f(options, "Mutation map" + (doc.method.empty() ? std::string() : " (" + doc.method + ")"));
    const double fs = f.fs(), cw = fs;
    const double x0 = fs * 2.4;
    Legend(f, fs, f.top(), fs * 12, -max_abs, max_abs, [&](double t) { return color_of(static_cast<float>((2 * t - 1) * max_abs)); },
           "mutation-scale", "score");
    double y = f.top() + fs * 3.4;
    const int64_t per = PerBlock(L, f.width() - x0 - fs, cw);
    const bool vector = L * A <= kMaxMutationCells;
    double right = 0;
    for (int64_t from = 0; from < L; from += per) {
        const int64_t n = std::min(per, L - from);
        Ruler(f, x0, y, cw, from, n, doc.first_position);
        SequenceRow(f, x0, y + fs * 1.1, cw, doc.sequence, from, n, " font-size=\"" + Num(fs * 0.85) + "\"");
        const double gy = y + fs * 1.5;
        for (int64_t a = 0; a < A; ++a) {
            f.text("letter", x0 - fs * 0.4, gy + cw * (static_cast<double>(a) + 0.5) + fs * 0.35, std::string(1, doc.alphabet[static_cast<size_t>(a)]),
                   "end", kTextColor, " font-family=\"" + std::string(kMonoFont) + "\" font-size=\"" + Num(fs * 0.85) + "\"");
        }
        if (vector) {
            for (int64_t k = 0; k < n; ++k) {
                const int64_t i = from + k;
                for (int64_t a = 0; a < A; ++a) {
                    const float v = doc.values[static_cast<size_t>(i * A + a)];
                    const char letter = doc.alphabet[static_cast<size_t>(a)];
                    const bool wild = letter == doc.sequence[static_cast<size_t>(i)];
                    const std::string name = Where(doc.sequence, i, doc.first_position) + (wild ? " (wild type)" : std::string(1, letter));
                    f.body() += "<rect class=\"cell\" x=\"" + Num(x0 + cw * static_cast<double>(k)) + "\" y=\"" + Num(gy + cw * static_cast<double>(a)) +
                                "\" width=\"" + Num(cw) + "\" height=\"" + Num(cw) + "\" shape-rendering=\"crispEdges\" fill=\"" +
                                (std::isfinite(v) ? Hex(color_of(v)) : std::string(kMissingColor)) + "\"><title>" + Escape(name) + ": " +
                                (std::isfinite(v) ? ValueText(v) : "not scored") + "</title></rect>\n";
                }
            }
        } else {
            std::vector<unsigned char> rgb(static_cast<size_t>(n * A * 3));
            auto byte = [](float c) { return static_cast<unsigned char>(std::lround(std::clamp(c, 0.0f, 1.0f) * 255.0f)); };
            for (int64_t a = 0; a < A; ++a) {
                for (int64_t k = 0; k < n; ++k) {
                    const float v = doc.values[static_cast<size_t>((from + k) * A + a)];
                    const RgbColor c = std::isfinite(v) ? color_of(v) : Gray(kMissingColor);
                    const size_t p = static_cast<size_t>((a * n + k) * 3);
                    rgb[p] = byte(c.r);
                    rgb[p + 1] = byte(c.g);
                    rgb[p + 2] = byte(c.b);
                }
            }
            f.body() += "<image class=\"mutation-image\" x=\"" + Num(x0) + "\" y=\"" + Num(gy) + "\" width=\"" + Num(cw * static_cast<double>(n)) +
                        "\" height=\"" + Num(cw * static_cast<double>(A)) +
                        "\" preserveAspectRatio=\"none\" style=\"image-rendering:pixelated\" href=\"" + PngDataUri(rgb, n, A) + "\"/>\n";
        }
        // The wild type's cell: a dot.
        for (int64_t k = 0; k < n; ++k) {
            const size_t a = doc.alphabet.find(doc.sequence[static_cast<size_t>(from + k)]);
            if (a == std::string::npos) continue;
            f.body() += "<circle class=\"wild-type\" cx=\"" + Num(x0 + cw * (static_cast<double>(k) + 0.5)) + "\" cy=\"" +
                        Num(gy + cw * (static_cast<double>(a) + 0.5)) + "\" r=\"" + Num(cw * 0.18) + "\" fill=\"" + kTextColor +
                        "\" pointer-events=\"none\"/>\n";
        }
        f.body() += "<rect class=\"frame\" x=\"" + Num(x0) + "\" y=\"" + Num(gy) + "\" width=\"" + Num(cw * static_cast<double>(n)) +
                    "\" height=\"" + Num(cw * static_cast<double>(A)) + "\" fill=\"none\" stroke=\"" + kAxisColor + "\" stroke-width=\"0.5\"/>\n";
        right = std::max(right, x0 + cw * static_cast<double>(n) + fs);
        y = gy + cw * static_cast<double>(A) + fs * 2.2;
    }
    return f.finish(y - fs * 0.8, std::max(right, fs * 22));
}

// ---- sequence logo -------------------------------------------------------------------------

namespace {

/** @brief WebLogo's "chemistry" colors. */
const char* ChemistryColor(char c) {
    switch (c) {
        case 'G': case 'S': case 'T': case 'Y': case 'C': return "#109648";  // polar
        case 'N': case 'Q': return "#7b2d8b";                                 // amide
        case 'K': case 'R': case 'H': return "#255c99";                       // basic
        case 'D': case 'E': return "#d62839";                                 // acidic
        default: return "#1a1a1a";                                            // hydrophobic and others
    }
}

/** @brief One letter stretched to the box [x, x + w] x [baseline - h, baseline]. At font size 100,
 *         bold sans-serif capitals are about 71.6 units tall (Helvetica, Arial, Liberation Sans). */
std::string Glyph(double x, double baseline, double w, double h, char letter, const char* color) {
    constexpr double kAdvance = 60.0, kCapHeight = 71.6;
    return "<text class=\"glyph\" x=\"0\" y=\"0\" font-size=\"100\" font-weight=\"bold\" textLength=\"" + Num(kAdvance) +
           "\" lengthAdjust=\"spacingAndGlyphs\" fill=\"" + color + "\" transform=\"matrix(" + Num4(w / kAdvance) + " 0 0 " +
           Num4(h / kCapHeight) + " " + Num(x) + " " + Num(baseline) + ")\">" + Escape(std::string(1, letter)) + "</text>\n";
}

}  // namespace

std::string RenderSequenceLogoSvg(const SequenceLogoDocument& doc, const SvgOptions& options) {
    CheckOptions(options);
    Check(protein_detail::SequenceLogoProblem(doc), "RenderSequenceLogoSvg");
    const int64_t P = doc.positions(), A = static_cast<int64_t>(doc.alphabet.size());
    const std::vector<float> bits = InformationContent(doc);
    const double max_bits = std::log2(static_cast<double>(A));

    Figure f(options, "Sequence logo" + (doc.method.empty() ? std::string() : " (" + doc.method + ")"));
    const double fs = f.fs(), cw = fs * 1.15, H = fs * 7.0;
    const double x0 = fs * 3.4;
    const int64_t per = PerBlock(P, f.width() - x0 - fs, cw);
    double y = f.top() + fs * 0.5;
    double right = 0;
    for (int64_t from = 0; from < P; from += per) {
        const int64_t n = std::min(per, P - from);
        const double base = y + H;
        f.y_axis(x0 - fs * 0.2, y, base, 0.0, max_bits, from == 0 ? "bits" : "", TextWidth("4", fs * 0.85));
        for (int64_t k = 0; k < n; ++k) {
            const int64_t i = from + k;
            const double x = x0 + cw * static_cast<double>(k);
            std::vector<std::pair<float, int64_t>> order;
            for (int64_t a = 0; a < A; ++a) order.emplace_back(doc.probabilities[static_cast<size_t>(i * A + a)], a);
            // Smallest at the bottom; ties in alphabet order.
            std::stable_sort(order.begin(), order.end(), [](const auto& p, const auto& q) { return p.first < q.first; });
            double top = base;
            const double column = H * bits[static_cast<size_t>(i)] / max_bits;
            for (const auto& [p, a] : order) {
                const double h = column * p;
                if (h < 0.25) continue;
                const char letter = doc.alphabet[static_cast<size_t>(a)];
                f.body() += Glyph(x + cw * 0.04, top, cw * 0.92, h, letter, ChemistryColor(letter));
                top -= h;
            }
            std::string tip = (doc.sequence.empty() ? std::string() : std::string(1, doc.sequence[static_cast<size_t>(i)])) +
                              std::to_string(doc.first_position + i) + ":";
            for (size_t r = 0; r < 3 && r < order.size(); ++r) {
                const auto& [p, a] = order[order.size() - 1 - r];
                tip += std::string(r ? "," : "") + " " + doc.alphabet[static_cast<size_t>(a)] + " " + ValueText(100.0 * p) + "%";
            }
            tip += " (" + ValueText(bits[static_cast<size_t>(i)]) + " bits)";
            f.body() += "<rect class=\"column\" x=\"" + Num(x) + "\" y=\"" + Num(y) + "\" width=\"" + Num(cw) + "\" height=\"" + Num(H) +
                        "\" fill=\"#ffffff\" fill-opacity=\"0\"><title>" + Escape(tip) + "</title></rect>\n";
        }
        f.line("axis", x0 - fs * 0.2, base, x0 + cw * static_cast<double>(n), base, kAxisColor);
        double below = base + fs * 1.15;
        if (!doc.sequence.empty()) {
            SequenceRow(f, x0, below, cw, doc.sequence, from, n, " font-size=\"" + Num(fs * 0.85) + "\"");
            below += fs * 1.0;
        }
        Ruler(f, x0, below + fs * 0.9, cw, from, n, doc.first_position);
        right = std::max(right, x0 + cw * static_cast<double>(n) + fs);
        y = below + fs * 2.4;
    }
    return f.finish(y - fs * 0.6, std::max(right, fs * 20));
}

// ---- contact map ---------------------------------------------------------------------------

std::string RenderContactMapSvg(const ContactMapDocument& doc, const ContactMapViewOptions& view, const SvgOptions& options) {
    CheckOptions(options);
    Check(protein_detail::ContactMapProblem(doc), "RenderContactMapSvg");
    if (view.min_separation < 1) throw std::invalid_argument("RenderContactMapSvg: min_separation must be at least 1");
    const int64_t L = doc.length();
    const bool has_truth = !doc.truth.empty();
    auto at = [&](const std::vector<float>& m, int64_t i, int64_t j) { return m[static_cast<size_t>(i * L + j)]; };

    // The predicted scores' range, over the pairs drawn (above the diagonal).
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (int64_t i = 0; i < L; ++i) {
        for (int64_t j = i + 1; j < L; ++j) {
            const float v = at(doc.predicted, i, j);
            if (std::isfinite(v)) {
                lo = std::min(lo, static_cast<double>(v));
                hi = std::max(hi, static_cast<double>(v));
            }
        }
    }
    if (!(lo < hi)) {
        lo = std::isfinite(lo) ? lo : 0.0;
        hi = lo + 1.0;
    }
    auto shade = [&](float v) { return std::isfinite(v) ? Blues((v - lo) / (hi - lo)) : Gray(kMissingColor); };
    auto lower = [&](int64_t i, int64_t j) {  // i > j: the structure
        if (!has_truth) return RgbColor{1.0f, 1.0f, 1.0f};
        const float t = at(doc.truth, i, j);
        return std::isnan(t) ? Gray(kUnknown) : (t > 0.5f ? Gray(kContact) : RgbColor{1.0f, 1.0f, 1.0f});
    };
    auto cell_color = [&](int64_t i, int64_t j) {
        if (i == j) return Gray("#d9d9d9");
        return j > i ? shade(at(doc.predicted, i, j)) : lower(i, j);
    };

    // The L best pairs at least min_separation apart, as ContactPrecision ranks them.
    std::vector<std::pair<int64_t, int64_t>> pairs;
    for (int64_t i = 0; i < L; ++i) {
        for (int64_t j = i + view.min_separation; j < L; ++j) pairs.emplace_back(i, j);
    }
    std::stable_sort(pairs.begin(), pairs.end(), [&](const auto& p, const auto& q) { return at(doc.predicted, p.first, p.second) > at(doc.predicted, q.first, q.second); });
    pairs.resize(std::min<size_t>(pairs.size(), static_cast<size_t>(L)));

    Figure f(options, "Contact map" + (doc.method.empty() ? std::string() : " (" + doc.method + ")"));
    const double fs = f.fs();
    const double x0 = fs * 3.6, y0 = f.top() + fs * 1.8;
    const double colorbar_w = fs * 5.5;
    const double S = std::max(fs * 8, std::min(f.width() - x0 - colorbar_w - fs * 1.5, fs * 60));
    const double cell = S / static_cast<double>(L);

    if (L * L <= kMaxVectorCells) {
        for (int64_t i = 0; i < L; ++i) {
            for (int64_t j = 0; j < L; ++j) {
                const RgbColor c = cell_color(i, j);
                if (c.r == 1.0f && c.g == 1.0f && c.b == 1.0f) continue;
                f.body() += "<rect class=\"cell\" x=\"" + Num(x0 + cell * static_cast<double>(j)) + "\" y=\"" + Num(y0 + cell * static_cast<double>(i)) +
                            "\" width=\"" + Num(cell) + "\" height=\"" + Num(cell) + "\" shape-rendering=\"crispEdges\" fill=\"" + Hex(c) + "\"/>\n";
            }
        }
    } else {
        std::vector<unsigned char> rgb(static_cast<size_t>(L * L * 3));
        auto byte = [](float c) { return static_cast<unsigned char>(std::lround(std::clamp(c, 0.0f, 1.0f) * 255.0f)); };
        for (int64_t i = 0; i < L; ++i) {
            for (int64_t j = 0; j < L; ++j) {
                const RgbColor c = cell_color(i, j);
                const size_t p = static_cast<size_t>((i * L + j) * 3);
                rgb[p] = byte(c.r);
                rgb[p + 1] = byte(c.g);
                rgb[p + 2] = byte(c.b);
            }
        }
        f.body() += "<image class=\"contact-image\" x=\"" + Num(x0) + "\" y=\"" + Num(y0) + "\" width=\"" + Num(S) + "\" height=\"" + Num(S) +
                    "\" preserveAspectRatio=\"none\" style=\"image-rendering:pixelated\" href=\"" + PngDataUri(rgb, L, L) + "\"/>\n";
    }

    // Dots below the diagonal, at (row j, column i) for the pair i < j.
    const double r = std::clamp(cell * 0.45, 1.2, fs * 0.4);
    for (size_t rank = 0; rank < pairs.size(); ++rank) {
        const auto [i, j] = pairs[rank];
        const float t = has_truth ? at(doc.truth, i, j) : std::numeric_limits<float>::quiet_NaN();
        const bool known = has_truth && !std::isnan(t);
        const std::string fill = !has_truth ? "#444444" : (!known ? "none" : (t > 0.5f ? kRight : kWrong));
        const std::string stroke = known || !has_truth ? "none" : "#555555";
        const std::string verdict = !has_truth ? "" : (!known ? ", distance unknown" : (t > 0.5f ? ", in contact" : ", not in contact"));
        f.body() += "<circle class=\"top-pair\" cx=\"" + Num(x0 + cell * (static_cast<double>(i) + 0.5)) + "\" cy=\"" +
                    Num(y0 + cell * (static_cast<double>(j) + 0.5)) + "\" r=\"" + Num(r) + "\" fill=\"" + fill + "\" stroke=\"" + stroke +
                    "\" stroke-width=\"0.8\"><title>" + Escape(Where(doc.sequence, i, doc.first_position) + "-" + Where(doc.sequence, j, doc.first_position) +
                                                              " (#" + std::to_string(rank + 1) + "): " + ValueText(at(doc.predicted, i, j)) + verdict) +
                    "</title></circle>\n";
    }
    f.body() += "<rect class=\"frame\" x=\"" + Num(x0) + "\" y=\"" + Num(y0) + "\" width=\"" + Num(S) + "\" height=\"" + Num(S) +
                "\" fill=\"none\" stroke=\"" + kAxisColor + "\" stroke-width=\"0.5\"/>\n";

    // Residue numbers along the top and the left.
    for (double t : NiceTicks(static_cast<double>(doc.first_position), static_cast<double>(doc.first_position + L - 1), 6)) {
        if (t != std::floor(t)) continue;
        const double p = x0 + cell * (t - static_cast<double>(doc.first_position) + 0.5);
        const double q = y0 + cell * (t - static_cast<double>(doc.first_position) + 0.5);
        f.line("tick", p, y0 - fs * 0.3, p, y0, kAxisColor);
        f.text("tick-label", p, y0 - fs * 0.5, ValueText(t), "middle", kMutedColor, " font-size=\"" + Num(fs * 0.85) + "\"");
        f.line("tick", x0 - fs * 0.3, q, x0, q, kAxisColor);
        f.text("tick-label", x0 - fs * 0.45, q + fs * 0.3, ValueText(t), "end", kMutedColor, " font-size=\"" + Num(fs * 0.85) + "\"");
    }
    ColorBar(f, x0 + S + fs, y0, std::min(S, fs * 14), lo, hi, [&](float t) { return Blues(t); }, "contact-scale");
    f.text("legend-label", x0 + S + fs, y0 + std::min(S, fs * 14) + fs * 1.3, "predicted", "start", kMutedColor,
           " font-size=\"" + Num(fs * 0.85) + "\"");

    double y = y0 + S + fs * 0.3;
    const std::string sep = std::to_string(view.min_separation);
    const std::string caption = "Above the diagonal: predicted. Below: the " + std::to_string(pairs.size()) + " best pairs " + sep + "+ apart" +
                                (has_truth ? " (blue in contact, red not), over true contacts (gray)." : ".");
    for (const std::string& line : Wrap(caption, S + colorbar_w, fs * 0.85)) {
        y += fs * 1.3;
        f.text("caption", x0, y, line, "start", kMutedColor, " font-size=\"" + Num(fs * 0.85) + "\"");
    }
    if (has_truth) {
        const ContactMap predicted{L, doc.predicted}, truth{L, doc.truth};
        y += fs * 1.3;
        f.text("precision", x0, y,
               "Precision at L: " + ValueText(ContactPrecision(predicted, truth, {view.min_separation, 0}, L)) + " (" + sep + "+ apart), " +
                   ValueText(ContactPrecision(predicted, truth, kLongRange, L)) + " (long range, 24+)",
               "start", kTextColor, " font-size=\"" + Num(fs * 0.85) + "\"");
    }
    return f.finish(y + fs, x0 + S + colorbar_w + fs);
}

// ---- residue tracks ------------------------------------------------------------------------

namespace {

const char* CategoryColor(size_t k) {
    static const char* kPalette[] = {"#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"};
    return kPalette[k % (sizeof kPalette / sizeof kPalette[0])];
}

/** @brief A track's color scale: diverging around zero when signed, Viridis from its minimum (or
 *         zero, when nothing is negative) otherwise. */
struct TrackScale {
    bool is_signed = false;
    double lo = 0, hi = 1;
    RgbColor operator()(float v) const {
        if (!std::isfinite(v)) return Gray(kMissingColor);
        if (is_signed) return DivergingColormap(NormalizeSigned(v, static_cast<float>(hi)));
        return ViridisColormap(static_cast<float>((v - lo) / (hi - lo)));
    }
};

TrackScale ScaleOf(const ResidueTracksDocument::Track& t) {
    TrackScale s;
    s.is_signed = t.is_signed;
    if (t.is_signed) {
        s.hi = MaxAbs(t.values);
        s.lo = -s.hi;
        return s;
    }
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (float v : t.values) {
        if (!std::isfinite(v)) continue;
        lo = std::min(lo, static_cast<double>(v));
        hi = std::max(hi, static_cast<double>(v));
    }
    if (!std::isfinite(lo)) return s;
    s.lo = std::min(0.0, lo);
    s.hi = hi > s.lo ? hi : s.lo + 1.0;
    return s;
}

}  // namespace

std::string RenderResidueTracksSvg(const ResidueTracksDocument& doc, const SvgOptions& options) {
    CheckOptions(options);
    Check(protein_detail::ResidueTracksProblem(doc), "RenderResidueTracksSvg");
    const auto L = static_cast<int64_t>(doc.sequence.size());
    std::vector<TrackScale> scales;
    for (const auto& t : doc.tracks) scales.push_back(ScaleOf(t));
    std::vector<std::string> categories;
    for (const auto& x : doc.features) {
        if (std::find(categories.begin(), categories.end(), x.category) == categories.end()) categories.push_back(x.category);
    }

    Figure f(options, "Residue tracks" + (doc.title.empty() ? std::string() : ": " + doc.title));
    const double fs = f.fs(), cw = fs * 0.8, row = fs * 1.15, frow = fs * 1.35;
    constexpr size_t kLabelColumns = 26;
    double label_w = fs * 3;
    for (const auto& t : doc.tracks) label_w = std::max(label_w, TextWidth(Truncate(t.name, kLabelColumns), fs * 0.9));
    for (const auto& c : categories) label_w = std::max(label_w, TextWidth(Truncate(c.empty() ? "features" : c, kLabelColumns), fs * 0.9));
    const double x0 = fs + label_w + fs * 0.6;
    const int64_t per = PerBlock(L, f.width() - x0 - fs, cw);
    double y = f.top() + fs * 1.2;
    double right = 0;
    for (int64_t from = 0; from < L; from += per) {
        const int64_t n = std::min(per, L - from);
        Ruler(f, x0, y, cw, from, n, doc.first_position);
        SequenceRow(f, x0, y + fs * 1.1, cw, doc.sequence, from, n, " font-size=\"" + Num(fs * 0.8) + "\"");
        double ty = y + fs * 1.55;
        for (size_t t = 0; t < doc.tracks.size(); ++t) {
            const auto& track = doc.tracks[t];
            f.text("track-label", x0 - fs * 0.5, ty + row * 0.72, Truncate(track.name, kLabelColumns), "end", kTextColor,
                   " font-size=\"" + Num(fs * 0.9) + "\"");
            for (int64_t k = 0; k < n; ++k) {
                const float v = track.values[static_cast<size_t>(from + k)];
                f.body() += "<rect class=\"track-cell\" x=\"" + Num(x0 + cw * static_cast<double>(k)) + "\" y=\"" + Num(ty) + "\" width=\"" + Num(cw) +
                            "\" height=\"" + Num(row - 2) + "\" shape-rendering=\"crispEdges\" fill=\"" + Hex(scales[t](v)) + "\"><title>" +
                            Escape(Where(doc.sequence, from + k, doc.first_position) + " " + track.name + ": " +
                                   (std::isfinite(v) ? ValueText(v) : std::string("none"))) +
                            "</title></rect>\n";
            }
            ty += row;
        }
        for (size_t c = 0; c < categories.size(); ++c) {
            f.text("feature-label", x0 - fs * 0.5, ty + frow * 0.68, Truncate(categories[c].empty() ? "features" : categories[c], kLabelColumns), "end",
                   kTextColor, " font-size=\"" + Num(fs * 0.9) + "\"");
            f.line("feature-axis", x0, ty + frow * 0.45, x0 + cw * static_cast<double>(n), ty + frow * 0.45, "#e0e0e0");
            for (const auto& x : doc.features) {
                if (x.category != categories[c]) continue;
                const int64_t s = std::max(x.start - doc.first_position, from), e = std::min(x.end - doc.first_position, from + n - 1);
                if (s > e) continue;
                const double fx = x0 + cw * static_cast<double>(s - from), fw = cw * static_cast<double>(e - s + 1);
                f.body() += "<rect class=\"feature\" x=\"" + Num(fx + 0.5) + "\" y=\"" + Num(ty + 1) + "\" width=\"" + Num(std::max(1.0, fw - 1)) +
                            "\" height=\"" + Num(frow - 3) + "\" rx=\"3\" fill=\"" + CategoryColor(c) + "\"><title>" +
                            Escape(x.name + " (" + std::to_string(x.start) + "-" + std::to_string(x.end) + ")") + "</title></rect>\n";
                if (TextWidth(x.name, fs * 0.8) + fs * 0.4 < fw) {
                    f.text("feature-name", fx + fw / 2, ty + frow * 0.68, x.name, "middle", "#ffffff",
                           " font-size=\"" + Num(fs * 0.8) + "\" pointer-events=\"none\"");
                }
            }
            ty += frow;
        }
        right = std::max(right, x0 + cw * static_cast<double>(n) + fs);
        y = ty + fs * 2.2;
    }
    // One scale per track.
    double lx = fs, ly = y - fs * 0.8;
    for (size_t t = 0; t < doc.tracks.size(); ++t) {
        const std::string caption = Truncate(doc.tracks[t].name, kLabelColumns);
        const double w = LegendWidth(caption, fs * 8, fs);
        if (lx > fs && lx + w > f.width()) {
            lx = fs;
            ly += fs * 2.6;
        }
        const TrackScale& s = scales[t];
        Legend(f, lx, ly, fs * 8, s.lo, s.hi, [&](double u) { return s(static_cast<float>(s.lo + u * (s.hi - s.lo))); },
               "track-scale-" + std::to_string(t), caption);
        lx += w;
        right = std::max(right, lx);
    }
    if (!doc.tracks.empty()) ly += fs * 2.6;
    return f.finish(ly, std::max(right, fs * 20));
}

// ---- pages ----------------------------------------------------------------------------------

std::string RenderMutationMapHtml(const MutationMapDocument& doc, const HtmlOptions& options) {
    const std::string heading = options.title.empty() ? (doc.title.empty() ? "Mutation map" : doc.title) : options.title;
    return FigurePage(heading, RenderMutationMapSvg(doc, ToSvgOptions(options)),
                      "Each cell scores a substitution" + (doc.method.empty() ? std::string() : " (" + doc.method + ")") +
                          "; dots mark the wild type. Hover a cell for its score.");
}

std::string RenderSequenceLogoHtml(const SequenceLogoDocument& doc, const HtmlOptions& options) {
    const std::string heading = options.title.empty() ? (doc.title.empty() ? "Sequence logo" : doc.title) : options.title;
    return FigurePage(heading, RenderSequenceLogoSvg(doc, ToSvgOptions(options)),
                      "Stack heights are information content in bits; hover a column for its likeliest letters.");
}

std::string RenderContactMapHtml(const ContactMapDocument& doc, const ContactMapViewOptions& view, const HtmlOptions& options) {
    const std::string heading = options.title.empty() ? (doc.title.empty() ? "Contact map" : doc.title) : options.title;
    return FigurePage(heading, RenderContactMapSvg(doc, view, ToSvgOptions(options)), "Hover a dot for the pair and its score.");
}

std::string RenderResidueTracksHtml(const ResidueTracksDocument& doc, const HtmlOptions& options) {
    const std::string heading = options.title.empty() ? (doc.title.empty() ? "Residue tracks" : doc.title) : options.title;
    return FigurePage(heading, RenderResidueTracksSvg(doc, ToSvgOptions(options)));
}

namespace {

// SRI hash of the pinned build (openssl dgst -sha384 -binary 3Dmol-min.js | openssl base64 -A).
constexpr const char* k3DmolSri = "sha384-OsczYbldvrHgslr9fFp/i4GiLSeuw9l+QIlv99ITw8soOwXcoGeflFMLg+CU/X1d";

constexpr const char* kStructureScript = R"JS(
(function (p) {
  var el = function (name) { return document.getElementById(p + '-' + name); };
  var d = JSON.parse(el('data').textContent);
  var viewer = $3Dmol.createViewer(el('viewer'), {backgroundColor: 'white'});
  viewer.addModel(d.structure, d.format);
  viewer.setStyle({}, {cartoon: {color: '#e3e3e3'}});
  viewer.setStyle({hetflag: true}, {stick: {colorscheme: 'grayCarbon', radius: 0.18}});
  viewer.setStyle({resn: ['HOH', 'WAT', 'DOD']}, {});
  var key = function (atom) { return String(atom.resi) + (atom.icode && atom.icode.trim() ? atom.icode.trim() : ''); };
  var current = 0;
  function paint(k) {
    current = k;
    var t = d.tracks[k];
    viewer.setStyle({chain: d.chain, hetflag: false}, {cartoon: {colorfunc: function (atom) {
      var c = t.colors[key(atom)];
      return c === undefined ? '#bdbdbd' : c;
    }}});
    viewer.render();
    el('legend-bar').style.background = 'linear-gradient(to right,' + t.stops.join(',') + ')';
    el('legend-lo').textContent = t.lo;
    el('legend-hi').textContent = t.hi;
  }
  viewer.setHoverable({chain: d.chain, hetflag: false}, true, function (atom, v) {
    if (atom.label) return;
    var k = key(atom), x = d.tracks[current].values[k];
    var text = atom.resn + ' ' + k + ': ' + (x === undefined || x === null ? 'none' : x);
    atom.label = v.addLabel(text, {position: atom, backgroundColor: '#222222', fontColor: 'white', fontSize: 12});
  }, function (atom, v) {
    if (atom.label) { v.removeLabel(atom.label); delete atom.label; }
  });
  var select = el('track');
  d.tracks.forEach(function (t, i) { var o = document.createElement('option'); o.value = i; o.textContent = t.name; select.appendChild(o); });
  select.addEventListener('change', function () { paint(Number(select.value)); });
  viewer.zoomTo({chain: d.chain});
  paint(0);
})
)JS";

}  // namespace

std::string StructurePanelHtml(const std::string& structure, const std::string& chain_id, const ResidueTracksDocument& tracks,
                               const HtmlOptions& options, const std::string& id) {
    if (id.empty() || id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-") != std::string::npos) {
        throw std::invalid_argument("StructurePanelHtml: the id must be lowercase letters, digits and hyphens");
    }
    Check(protein_detail::ResidueTracksProblem(tracks), "StructurePanelHtml");
    if (tracks.tracks.empty()) throw std::invalid_argument("StructurePanelHtml: the document has no track to color by");
    const size_t start = structure.find_first_not_of(" \t\r\n");
    const bool mmcif = start != std::string::npos && structure.compare(start, 5, "data_") == 0;
    const ProteinStructure parsed = mmcif ? ParseMmcif(structure) : ParsePdb(structure);
    const StructureChain& chain = parsed.chain(chain_id);
    const std::vector<std::string> chain_ids = ResidueIds(chain);
    std::vector<std::string> ids = tracks.residue_ids;
    if (ids.empty()) {
        if (chain.sequence() != tracks.sequence) {
            throw std::invalid_argument("StructurePanelHtml: chain " + chain_id + "'s residues (" + std::to_string(chain.residues.size()) +
                                        ") don't match the tracks' sequence; give residue_ids");
        }
        ids = chain_ids;
    } else {
        const std::set<std::string> known(chain_ids.begin(), chain_ids.end());
        for (const std::string& rid : ids) {
            if (known.count(rid) == 0) throw std::invalid_argument("StructurePanelHtml: chain " + chain_id + " has no residue " + rid);
        }
    }

    JsonValue data{JsonValue::Object{}};
    data.add("format", mmcif ? "cif" : "pdb");
    data.add("chain", chain_id);
    data.add("structure", structure);
    JsonValue list{JsonValue::Array{}};
    for (const auto& track : tracks.tracks) {
        const TrackScale scale = ScaleOf(track);
        JsonValue colors{JsonValue::Object{}}, values{JsonValue::Object{}}, stops{JsonValue::Array{}};
        for (size_t i = 0; i < ids.size(); ++i) {
            const float v = track.values[i];
            if (!std::isfinite(v)) continue;
            colors.add(ids[i], Hex(scale(v)));
            values.add(ids[i], ValueText(v));
        }
        for (int s = 0; s <= 10; ++s) stops.push_back(Hex(scale(static_cast<float>(scale.lo + (scale.hi - scale.lo) * s / 10.0))));
        JsonValue t{JsonValue::Object{}};
        t.add("name", track.name);
        t.add("colors", std::move(colors));
        t.add("values", std::move(values));
        t.add("stops", std::move(stops));
        t.add("lo", ValueText(scale.lo));
        t.add("hi", ValueText(scale.hi));
        list.push_back(std::move(t));
    }
    data.add("tracks", std::move(list));

    std::string scripts;
    if (options.scripts == HtmlScripts::Inline) {
        if (options.script_dir.empty()) throw std::invalid_argument("StructurePanelHtml: inline scripts need script_dir");
        scripts = "<script>" + html_detail::ReadScript(options.script_dir, "3Dmol-min.js") + "</script>\n";
    } else {
        scripts = std::string("<script src=\"https://cdn.jsdelivr.net/npm/3dmol@") + k3DmolVersion + "/build/3Dmol-min.js\" integrity=\"" + k3DmolSri +
                  "\" crossorigin=\"anonymous\"></script>\n";
    }
    const int height = std::max(320, options.width * 3 / 4);
    return "<div class=\"legend\"><label>Color by <select id=\"" + id + "-track\"></select></label><span id=\"" + id +
           "-legend-lo\"></span><div class=\"bar\" id=\"" + id + "-legend-bar\"></div><span id=\"" + id +
           "-legend-hi\"></span><span>gray: no value</span></div>\n<div id=\"" + id + "-viewer\" style=\"position:relative;width:" +
           std::to_string(options.width) + "px;max-width:100%;height:" + std::to_string(height) +
           "px;margin-top:12px;border:1px solid #ddd\"></div>\n"
           "<p class=\"muted\">Drag to rotate, scroll to zoom; hover a residue for its value.</p>\n"
           "<script type=\"application/json\" id=\"" + id + "-data\">" + html_detail::ScriptSafe(WriteJson(data)) + "</script>\n" + scripts +
           "<script>" + kStructureScript + "('" + id + "');</script>\n";
}

std::string RenderStructureHtml(const std::string& structure, const std::string& chain_id, const ResidueTracksDocument& tracks,
                                const HtmlOptions& options) {
    const std::string heading = options.title.empty() ? (tracks.title.empty() ? "Structure, chain " + chain_id : tracks.title) : options.title;
    return html_detail::Page(heading, "", "<h1>" + Escape(heading) + "</h1>\n" + StructurePanelHtml(structure, chain_id, tracks, options, "structure"));
}

std::string RenderFeatureDashboardHtml(const FeatureDashboardDocument& doc, const std::string& structure, const std::string& chain,
                                       size_t example, const HtmlOptions& options) {
    if (example >= doc.top_examples.size()) throw std::invalid_argument("RenderFeatureDashboardHtml: no such example");
    const auto& e = doc.top_examples[example];
    ResidueTracksDocument tracks;
    for (const std::string& t : e.tokens) {
        if (t.size() != 1) throw std::invalid_argument("RenderFeatureDashboardHtml: the example's tokens must be single residues");
        tracks.sequence += t;
    }
    tracks.tracks.push_back({"feature " + std::to_string(doc.feature_index) + " activation", e.activations, false});
    const std::string panel = "<h1>On the structure: " + Escape(e.label) + "</h1>\n" + StructurePanelHtml(structure, chain, tracks, options, "panel");
    return RenderFeatureDashboardHtml(doc, options, panel);
}

}  // namespace pulsatrix
