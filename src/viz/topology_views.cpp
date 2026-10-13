// The persistence diagram, barcode and Betti curve views of a persistence diagram document (TDA-9).

#include "pulsatrix/viz/topology_views.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/tda/vectorization.hpp"
#include "svg_detail.hpp"

namespace pulsatrix {

using namespace svg_detail;

namespace {

// Tableau 10, as the projection view uses; one color per dimension.
constexpr const char* kDimensionColors[] = {"#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2",
                                            "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"};
constexpr const char* kBandColor = "#ececec";

const char* ColorOf(int dimension) { return kDimensionColors[static_cast<size_t>(dimension) % 10]; }

// The entries to draw, in the order the view lists them (or the document's).
std::vector<const PersistenceDiagramEntry*> Selected(const PersistenceDiagramDocument& doc, const PersistenceViewOptions& view,
                                                     const char* who) {
    (void)ToJson(doc);  // every rule a document must follow
    std::vector<const PersistenceDiagramEntry*> out;
    if (view.dimensions.empty()) {
        for (const PersistenceDiagramEntry& e : doc.diagrams) out.push_back(&e);
        return out;
    }
    for (int d : view.dimensions) {
        const PersistenceDiagramEntry* e = doc.find(d);
        if (e == nullptr) throw std::invalid_argument(std::string(who) + ": the document has no H" + std::to_string(d));
        out.push_back(e);
    }
    return out;
}

bool HasTests(const PersistenceDiagramEntry& e) { return !e.p_values.empty() || e.bootstrap_band > 0; }

// The scale range: [lo, hi_finite] holds every birth and finite death; essential features are drawn
// at `infinity`, the document's threshold when it's above every finite value, else a little above.
struct Scale {
    double lo = 0, hi_finite = 0, infinity = 0, hi = 0;
    bool any_essential = false;
    bool any_feature = false;
};

Scale ScaleOf(const PersistenceDiagramDocument& doc, const std::vector<const PersistenceDiagramEntry*>& entries) {
    Scale s;
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (const PersistenceDiagramEntry* e : entries) {
        for (size_t i = 0; i < e->size(); ++i) {
            s.any_feature = true;
            lo = std::min(lo, e->births[i]);
            hi = std::max(hi, e->births[i]);
            if (std::isfinite(e->deaths[i])) {
                hi = std::max(hi, e->deaths[i]);
            } else {
                s.any_essential = true;
            }
        }
    }
    if (!s.any_feature) lo = 0, hi = 1;
    s.lo = std::min(0.0, lo);
    s.hi_finite = hi > s.lo ? hi : s.lo + 1;
    const double span = s.hi_finite - s.lo;
    s.infinity = doc.threshold > s.hi_finite + 0.05 * span ? doc.threshold : s.hi_finite + 0.1 * span;
    s.hi = (s.any_essential ? s.infinity : s.hi_finite) + 0.04 * span;
    return s;
}

std::string Tooltip(const PersistenceDiagramEntry& e, size_t i) {
    std::string tip = "H" + std::to_string(e.dimension) + ": born " + ValueText(e.births[i]) + ", ";
    if (std::isfinite(e.deaths[i])) {
        tip += "dies " + ValueText(e.deaths[i]) + " (persistence " + ValueText(e.deaths[i] - e.births[i]) + ")";
    } else {
        tip += "never dies";
    }
    if (!e.p_values.empty()) {
        tip += std::isnan(e.p_values[i]) ? ", not tested" : ", p = " + ValueText(e.p_values[i]);
    }
    return tip;
}

std::string Name(const PersistenceDiagramDocument& doc, const std::string& what) {
    if (!doc.title.empty()) return doc.title;
    return doc.source.empty() ? what : what + ": " + doc.source;
}

// The legend at (x, y): a swatch per dimension, then what the marks mean. Returns its right edge.
double Legend(Figure& f, double x, double y, const std::vector<const PersistenceDiagramEntry*>& entries, bool tests,
              bool band, bool essential) {
    const double fs = f.fs();
    double right = x;
    double ly = y;
    for (const PersistenceDiagramEntry* e : entries) {
        const std::string label = "H" + std::to_string(e->dimension) + " (" + std::to_string(e->size()) + ")";
        f.body() += "<circle class=\"legend-swatch\" cx=\"" + Num(x + fs * 0.4) + "\" cy=\"" + Num(ly + fs * 0.5) + "\" r=\"" +
                    Num(fs * 0.4) + "\" fill=\"" + ColorOf(e->dimension) + "\"/>\n";
        f.text("legend-label", x + fs * 1.2, ly + fs * 0.85, label);
        right = std::max(right, x + fs * 1.2 + TextWidth(label, fs));
        ly += fs * 1.4;
    }
    ly += fs * 0.4;
    auto note = [&](const std::string& s) {
        f.text("legend-note", x, ly + fs * 0.85, s, "start", kMutedColor, " font-size=\"" + Num(fs * 0.85) + "\"");
        right = std::max(right, x + TextWidth(s, fs * 0.85));
        ly += fs * 1.2;
    };
    if (tests) note("filled: significant; hollow: not");
    if (band) note("gray band: within the error bound");
    if (essential) note("\xE2\x88\x9E: never dies");
    return right;
}

}  // namespace

std::string RenderPersistenceDiagramSvg(const PersistenceDiagramDocument& doc, const PersistenceViewOptions& view,
                                        const SvgOptions& options) {
    CheckOptions(options);
    const std::vector<const PersistenceDiagramEntry*> entries = Selected(doc, view, "RenderPersistenceDiagramSvg");
    const Scale s = ScaleOf(doc, entries);
    Figure f(options, Name(doc, "Persistence diagram"));
    const double fs = f.fs();

    double label_w = 0;
    for (double t : NiceTicks(s.lo, s.hi)) label_w = std::max(label_w, TextWidth(ValueText(t), fs * 0.85));
    const double legend_w = fs * 13;
    const double x0 = fs * 2.2 + label_w + fs * 0.6;
    const double side = std::max(fs * 8, std::min(f.width() - x0 - fs * 1.5 - legend_w, 420.0));
    const double y_top = f.top() + fs;
    const double y_bottom = y_top + side;
    auto X = [&](double v) { return x0 + (v - s.lo) / (s.hi - s.lo) * side; };
    auto Y = [&](double v) { return y_bottom - (v - s.lo) / (s.hi - s.lo) * side; };

    // The band of persistence within twice the error bound: between the diagonal and y = x + 2e.
    double bound = 0;
    for (const PersistenceDiagramEntry* e : entries) bound = std::max(bound, e->error_bound);
    if (bound > 0) {
        const double w = std::min(2 * bound, s.hi - s.lo);
        f.body() += "<polygon class=\"error-band\" points=\"" + Num(X(s.lo)) + "," + Num(Y(s.lo)) + " " + Num(X(s.hi)) + "," +
                    Num(Y(s.hi)) + " " + Num(X(s.hi - w)) + "," + Num(Y(s.hi)) + " " + Num(X(s.lo)) + "," + Num(Y(s.lo + w)) +
                    "\" fill=\"" + kBandColor + "\"><title>" +
                    Escape("persistence up to " + ValueText(2 * bound) + ": twice the certified error bound") +
                    "</title></polygon>\n";
    }
    f.line("diagonal", X(s.lo), Y(s.lo), X(s.hi), Y(s.hi), kAxisColor);
    for (const PersistenceDiagramEntry* e : entries) {
        if (e->bootstrap_band > 0 && 2 * e->bootstrap_band < s.hi - s.lo) {
            const double b = 2 * e->bootstrap_band;
            f.line("bootstrap-band", X(s.lo), Y(s.lo + b), X(s.hi - b), Y(s.hi), ColorOf(e->dimension), 1.0, "2 3");
        }
    }
    if (s.any_essential) {
        f.line("infinity", X(s.lo), Y(s.infinity), X(s.hi), Y(s.infinity), kAxisColor, 1.0, "5 4");
        f.text("infinity-label", x0 - fs * 0.5, Y(s.infinity) + fs * 0.35, "\xE2\x88\x9E", "end", kMutedColor);
    }

    bool tests = false;
    const double r = fs * 0.32;
    for (const PersistenceDiagramEntry* e : entries) {
        tests = tests || HasTests(*e);
        for (size_t i = 0; i < e->size(); ++i) {
            const double death = std::isfinite(e->deaths[i]) ? e->deaths[i] : s.infinity;
            const bool filled = IsSignificant(*e, i);
            f.body() += std::string("<circle class=\"feature h") + std::to_string(e->dimension) + "\" cx=\"" + Num(X(e->births[i])) +
                        "\" cy=\"" + Num(Y(death)) + "\" r=\"" + Num(r) + "\" fill=\"" + (filled ? ColorOf(e->dimension) : "#ffffff") +
                        "\" fill-opacity=\"0.85\" stroke=\"" + ColorOf(e->dimension) + "\" stroke-width=\"1.2\"><title>" +
                        Escape(Tooltip(*e, i)) + "</title></circle>\n";
        }
    }
    if (!s.any_feature) f.text("empty", x0 + side / 2, y_top + side / 2, "no features", "middle", kMutedColor);
    // The axes cover the finite values; the ∞ line sits above them.
    f.y_axis(x0, Y(s.hi_finite), y_bottom, s.lo, s.hi_finite, "death", label_w);
    f.x_axis(y_bottom, x0, X(s.hi_finite), s.lo, s.hi_finite, "birth");
    const double right = Legend(f, x0 + side + fs * 1.5, y_top, entries, tests, bound > 0, s.any_essential);
    return f.finish(y_bottom + fs * 4.0, right + fs);
}

std::string RenderBarcodeSvg(const PersistenceDiagramDocument& doc, const PersistenceViewOptions& view, const SvgOptions& options) {
    CheckOptions(options);
    if (view.max_bars < 1) throw std::invalid_argument("RenderBarcodeSvg: max_bars must be at least 1");
    const std::vector<const PersistenceDiagramEntry*> entries = Selected(doc, view, "RenderBarcodeSvg");
    const Scale s = ScaleOf(doc, entries);
    Figure f(options, Name(doc, "Barcode"));
    const double fs = f.fs();

    const double x0 = fs * 3.5;
    const double x1 = std::max(x0 + fs * 10, f.width() - fs * 1.5);
    auto X = [&](double v) { return x0 + (v - s.lo) / (s.hi - s.lo) * (x1 - x0); };
    const double bar_h = fs * 0.55, step = fs * 0.8;
    double y = f.top() + fs * 0.5;

    for (const PersistenceDiagramEntry* e : entries) {
        const char* color = ColorOf(e->dimension);
        f.text("dimension-label", fs, y + fs * 0.9, "H" + std::to_string(e->dimension), "start", color, " font-weight=\"bold\"");
        std::vector<size_t> order(e->size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return e->deaths[a] - e->births[a] > e->deaths[b] - e->births[b];
        });
        const size_t shown = std::min(order.size(), static_cast<size_t>(view.max_bars));
        if (order.empty()) {
            f.text("empty", x0, y + fs * 0.9, "no features", "start", kMutedColor);
            y += fs * 1.6;
            continue;
        }
        for (size_t k = 0; k < shown; ++k) {
            const size_t i = order[k];
            const bool essential = !std::isfinite(e->deaths[i]);
            const double b = X(e->births[i]), d = X(essential ? s.infinity : e->deaths[i]);
            const double mid = y + bar_h / 2;
            const std::string opacity = IsSignificant(*e, i) ? "0.9" : "0.3";
            f.body() += "<rect class=\"bar h" + std::to_string(e->dimension) + "\" x=\"" + Num(b) + "\" y=\"" + Num(y) + "\" width=\"" +
                        Num(std::max(d - b, 0.5)) + "\" height=\"" + Num(bar_h) + "\" fill=\"" + color + "\" fill-opacity=\"" + opacity +
                        "\"><title>" + Escape(Tooltip(*e, i)) + "</title></rect>\n";
            if (essential) {
                f.body() += "<polygon class=\"essential\" points=\"" + Num(d) + "," + Num(y - bar_h * 0.3) + " " + Num(d + bar_h) + "," +
                            Num(mid) + " " + Num(d) + "," + Num(y + bar_h * 1.3) + "\" fill=\"" + color + "\"/>\n";
            }
            if (e->error_bound > 0) {
                const double w = X(s.lo + e->error_bound) - X(s.lo);
                f.line("whisker", b - w, mid, b + w, mid, kTextColor, 0.8);
                if (!essential) f.line("whisker", d - w, mid, d + w, mid, kTextColor, 0.8);
            }
            y += step;
        }
        if (order.size() > shown) {
            f.text("more-bars", x0, y + fs * 0.8, "+" + std::to_string(order.size() - shown) + " shorter bars not shown", "start",
                   kMutedColor, " font-size=\"" + Num(fs * 0.85) + "\"");
            y += fs * 1.2;
        }
        y += fs * 0.8;
    }
    if (s.any_essential) {
        f.line("infinity", X(s.infinity), f.top(), X(s.infinity), y, kAxisColor, 1.0, "5 4");
        f.text("infinity-label", X(s.infinity), f.top(), "\xE2\x88\x9E", "middle", kMutedColor);
    }
    f.x_axis(y, x0, X(s.hi_finite), s.lo, s.hi_finite, "scale");
    double bottom = y + fs * 4.4;
    double error_bound = 0;
    for (const PersistenceDiagramEntry* e : entries) error_bound = std::max(error_bound, e->error_bound);
    if (error_bound > 0) {
        f.text("barcode-note", x0, bottom, "Whiskers: \xC2\xB1" + ValueText(error_bound) + ", the certified error bound", "start",
               kMutedColor, " font-size=\"" + Num(fs * 0.85) + "\"");
        bottom += fs * 1.2;
    }
    return f.finish(bottom + fs * 0.4);
}

std::string RenderBettiCurveSvg(const PersistenceDiagramDocument& doc, const PersistenceViewOptions& view, const SvgOptions& options) {
    CheckOptions(options);
    if (view.grid_points < 2) throw std::invalid_argument("RenderBettiCurveSvg: grid_points must be at least 2");
    const std::vector<const PersistenceDiagramEntry*> entries = Selected(doc, view, "RenderBettiCurveSvg");
    const Scale s = ScaleOf(doc, entries);
    const std::vector<double> grid = LinearGrid(s.lo, s.hi_finite + 0.05 * (s.hi_finite - s.lo), view.grid_points);

    std::vector<std::vector<int64_t>> curves;
    int64_t top = 1;
    for (const PersistenceDiagramEntry* e : entries) {
        PersistenceDiagram d;
        d.dimension = e->dimension;
        for (size_t i = 0; i < e->size(); ++i) {
            PersistencePair p;
            p.birth = e->births[i];
            p.death = e->deaths[i];
            d.pairs.push_back(p);
        }
        curves.push_back(BettiCurve(d, grid));
        for (int64_t v : curves.back()) top = std::max(top, v);
    }

    Figure f(options, Name(doc, "Betti curves"));
    const double fs = f.fs();
    double label_w = 0;
    for (double t : NiceTicks(0, static_cast<double>(top))) label_w = std::max(label_w, TextWidth(ValueText(t), fs * 0.85));
    const double legend_w = fs * 8;
    const double x0 = fs * 2.2 + label_w + fs * 0.6;
    const double x1 = std::max(x0 + fs * 10, f.width() - fs * 1.5 - legend_w);
    const double y_top = f.top() + fs;
    const double y_bottom = y_top + std::min(fs * 18, (x1 - x0) * 0.6);
    const double xlo = grid.front(), xhi = grid.back();
    const double yhi = static_cast<double>(top) * 1.05;
    auto X = [&](double v) { return x0 + (v - xlo) / (xhi - xlo) * (x1 - x0); };
    auto Y = [&](double v) { return y_bottom - v / yhi * (y_bottom - y_top); };

    for (size_t c = 0; c < curves.size(); ++c) {
        std::string points;
        for (size_t g = 0; g < grid.size(); ++g) {
            const double v = static_cast<double>(curves[c][g]);
            if (g > 0) points += Num(X(grid[g])) + "," + Num(Y(static_cast<double>(curves[c][g - 1]))) + " ";
            points += Num(X(grid[g])) + "," + Num(Y(v)) + " ";
        }
        points.pop_back();
        f.body() += "<polyline class=\"betti h" + std::to_string(entries[c]->dimension) + "\" points=\"" + points +
                    "\" fill=\"none\" stroke=\"" + ColorOf(entries[c]->dimension) + "\" stroke-width=\"1.6\"><title>Betti " +
                    std::to_string(entries[c]->dimension) + "</title></polyline>\n";
    }
    f.y_axis(x0, y_top, y_bottom, 0, yhi, "features alive", label_w);
    f.x_axis(y_bottom, x0, x1, xlo, xhi, "scale");
    double right = x1;
    const double lx = x1 + fs * 1.5;
    for (size_t c = 0; c < entries.size(); ++c) {
        const double ly = y_top + fs * 1.4 * static_cast<double>(c);
        const std::string label = "\xCE\xB2" + std::to_string(entries[c]->dimension);
        f.line("legend-swatch", lx, ly + fs * 0.5, lx + fs, ly + fs * 0.5, ColorOf(entries[c]->dimension), 2.0);
        f.text("legend-label", lx + fs * 1.4, ly + fs * 0.85, label);
        right = std::max(right, lx + fs * 1.4 + TextWidth(label, fs));
    }
    const char* note = "Not covered by the error bound: short-lived features can come and go";
    f.text("betti-note", x0, y_bottom + fs * 4.2, note, "start", kMutedColor, " font-size=\"" + Num(fs * 0.85) + "\"");
    return f.finish(y_bottom + fs * 5.4, std::max(right + fs, x0 + TextWidth(note, fs * 0.85)));
}

}  // namespace pulsatrix
