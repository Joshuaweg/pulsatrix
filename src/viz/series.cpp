#include "pulsatrix/viz/series.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "document_io.hpp"
#include "svg_detail.hpp"

namespace pulsatrix {

using namespace document_io;
using namespace svg_detail;

namespace {

// Tableau 10, as the projection and topology views use. A series keeps its color in every panel.
constexpr const char* kSeriesColors[] = {"#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2",
                                         "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"};

AxisScale ParseScale(const std::string& name, const std::string& pointer, const DocReader& r) {
    if (name == "linear") return AxisScale::Linear;
    if (name == "log") return AxisScale::Log;
    r.fail(pointer, "\"" + name + "\" is not an axis scale (linear or log)");
}

// Every structural rule; `fail(pointer, what)` reports the first broken one and must not return.
template <typename Fail>
void CheckStructure(const SeriesDocument& doc, Fail fail) {
    if (doc.panels.empty()) fail("/panels", "a series document needs at least one panel");
    for (size_t p = 0; p < doc.panels.size(); ++p) {
        const std::string pp = Index("/panels", p);
        const SeriesPanel& panel = doc.panels[p];
        if (panel.series.empty()) fail(pp + "/series", "a panel needs at least one series");
        std::set<std::string> names;
        for (size_t s = 0; s < panel.series.size(); ++s) {
            const std::string sp = Index(pp + "/series", s);
            const Series& series = panel.series[s];
            if (series.name.empty()) fail(sp + "/name", "a series needs a name");
            if (!names.insert(series.name).second) fail(sp + "/name", "\"" + series.name + "\" appears twice in the panel");
            if (series.x.empty()) fail(sp + "/x", "a series needs at least one point");
            if (series.y.size() != series.x.size()) fail(sp + "/y", "needs one value per x");
            if (series.lo.size() != series.hi.size()) fail(sp + "/hi", "lo and hi need the same length");
            if (!series.lo.empty() && series.lo.size() != series.x.size()) fail(sp + "/lo", "needs one value per x, or none");
            for (size_t i = 0; i < series.lo.size(); ++i) {
                if (series.lo[i] > series.hi[i]) fail(Index(sp + "/lo", i), "lo is above hi");
            }
        }
    }
}

JsonValue Doubles(DocWriter& w, const std::vector<double>& values, const std::string& pointer) {
    JsonValue a{JsonValue::Array{}};
    for (size_t i = 0; i < values.size(); ++i) a.push_back(w.num_double(values[i], Index(pointer, i)));
    return a;
}

std::vector<double> ReadDoubles(DocReader& r, const JsonValue& v, const std::string& pointer) {
    const JsonValue::Array& a = r.array(v, pointer);
    std::vector<double> out;
    out.reserve(a.size());
    for (size_t i = 0; i < a.size(); ++i) out.push_back(r.number_double(a[i], Index(pointer, i)));
    return out;
}

}  // namespace

const char* AxisScaleName(AxisScale scale) { return scale == AxisScale::Log ? "log" : "linear"; }

std::string ToJson(const SeriesDocument& doc) {
    CheckStructure(doc, [](const std::string& pointer, const std::string& what) {
        throw std::invalid_argument("ToJson(SeriesDocument): " + pointer + ": " + what);
    });
    DocWriter w("series");
    w.root().add("title", doc.title);
    w.root().add("x_label", doc.x_label);
    w.root().add("y_label", doc.y_label);
    w.root().add("x_scale", AxisScaleName(doc.x_scale));
    w.root().add("y_scale", AxisScaleName(doc.y_scale));
    w.root().add("interval", doc.interval);
    JsonValue panels{JsonValue::Array{}};
    for (size_t p = 0; p < doc.panels.size(); ++p) {
        const std::string pp = Index("/panels", p);
        JsonValue panel{JsonValue::Object{}};
        panel.add("title", doc.panels[p].title);
        JsonValue series{JsonValue::Array{}};
        for (size_t s = 0; s < doc.panels[p].series.size(); ++s) {
            const Series& line = doc.panels[p].series[s];
            const std::string sp = Index(pp + "/series", s);
            JsonValue o{JsonValue::Object{}};
            o.add("name", line.name);
            o.add("x", Doubles(w, line.x, sp + "/x"));
            o.add("y", Doubles(w, line.y, sp + "/y"));
            if (line.has_intervals()) {
                o.add("lo", Doubles(w, line.lo, sp + "/lo"));
                o.add("hi", Doubles(w, line.hi, sp + "/hi"));
            }
            series.push_back(std::move(o));
        }
        panel.add("series", std::move(series));
        panels.push_back(std::move(panel));
    }
    w.root().add("panels", std::move(panels));
    return w.finish();
}

SeriesDocument ParseSeriesDocument(std::string_view json) {
    DocReader r(json, "series");
    const JsonValue& root = r.root();
    SeriesDocument doc;
    doc.title = r.string(r.member(root, "", "title"), "/title");
    doc.x_label = r.string(r.member(root, "", "x_label"), "/x_label");
    doc.y_label = r.string(r.member(root, "", "y_label"), "/y_label");
    doc.x_scale = ParseScale(r.string(r.member(root, "", "x_scale"), "/x_scale"), "/x_scale", r);
    doc.y_scale = ParseScale(r.string(r.member(root, "", "y_scale"), "/y_scale"), "/y_scale", r);
    doc.interval = r.string(r.member(root, "", "interval"), "/interval");
    const JsonValue::Array& panels = r.array(r.member(root, "", "panels"), "/panels");
    for (size_t p = 0; p < panels.size(); ++p) {
        const std::string pp = Index("/panels", p);
        r.object(panels[p], pp);
        SeriesPanel panel;
        panel.title = r.string(r.member(panels[p], pp, "title"), pp + "/title");
        const JsonValue::Array& series = r.array(r.member(panels[p], pp, "series"), pp + "/series");
        for (size_t s = 0; s < series.size(); ++s) {
            const std::string sp = Index(pp + "/series", s);
            r.object(series[s], sp);
            Series line;
            line.name = r.string(r.member(series[s], sp, "name"), sp + "/name");
            line.x = ReadDoubles(r, r.member(series[s], sp, "x"), sp + "/x");
            line.y = ReadDoubles(r, r.member(series[s], sp, "y"), sp + "/y");
            const JsonValue* lo = series[s].find("lo");
            const JsonValue* hi = series[s].find("hi");
            if ((lo == nullptr) != (hi == nullptr)) r.fail(sp, "lo and hi come together");
            if (lo != nullptr) {
                line.lo = ReadDoubles(r, *lo, sp + "/lo");
                line.hi = ReadDoubles(r, *hi, sp + "/hi");
            }
            panel.series.push_back(std::move(line));
        }
        doc.panels.push_back(std::move(panel));
    }
    r.finish();
    CheckStructure(doc, [&](const std::string& pointer, const std::string& what) { r.fail(pointer, what); });
    return doc;
}

// ---- SVG -------------------------------------------------------------------------------------

namespace {

struct Axis {
    AxisScale scale = AxisScale::Linear;
    double lo = 0;
    double hi = 1;

    double unit(double v) const {  // 0 at lo, 1 at hi
        if (scale == AxisScale::Log) return (std::log10(v) - std::log10(lo)) / (std::log10(hi) - std::log10(lo));
        return (v - lo) / (hi - lo);
    }

    std::vector<double> ticks() const {
        if (scale == AxisScale::Linear) return NiceTicks(lo, hi);
        std::vector<double> out;
        const int first = static_cast<int>(std::ceil(std::log10(lo) - 1e-9));
        const int last = static_cast<int>(std::floor(std::log10(hi) + 1e-9));
        for (int k = first; k <= last; ++k) out.push_back(std::pow(10.0, k));
        if (out.size() < 3) {  // under two decades: add 2 and 5 times each power of ten
            std::vector<double> more;
            for (int k = first - 1; k <= last; ++k) {
                for (double m : {1.0, 2.0, 5.0}) {
                    const double t = m * std::pow(10.0, k);
                    if (t >= lo * (1 - 1e-9) && t <= hi * (1 + 1e-9)) more.push_back(t);
                }
            }
            out = more;
        }
        return out;
    }
};

// The range an axis needs for these values, padded so marks don't touch the frame.
Axis AxisFor(AxisScale scale, double min_v, double max_v, bool zero_anchored) {
    Axis a;
    a.scale = scale;
    if (scale == AxisScale::Log) {
        double lo = std::log10(min_v), hi = std::log10(max_v);
        if (hi - lo < 1e-12) {
            lo -= 0.5;
            hi += 0.5;
        }
        const double pad = (hi - lo) * 0.05;
        a.lo = std::pow(10.0, lo - pad);
        a.hi = std::pow(10.0, hi + pad);
        return a;
    }
    if (zero_anchored) {
        const auto d = ZeroAnchoredDomain(min_v, max_v);
        a.lo = d.first;
        a.hi = d.second;
        return a;
    }
    double span = max_v - min_v;
    if (span == 0) span = std::max(1.0, std::abs(max_v));
    a.lo = min_v - span * 0.05;
    a.hi = max_v + span * 0.05;
    return a;
}

// A tick label: powers of ten from 0.0001 to 100000 written out ("1000", not "1e+03").
std::string TickText(double v, AxisScale scale) {
    if (scale == AxisScale::Log) {
        const double k = std::round(std::log10(v));
        if (std::abs(v - std::pow(10.0, k)) <= 1e-9 * v && k >= -4 && k <= 5) {
            if (k >= 0) return std::to_string(static_cast<long long>(std::llround(v)));
            return "0." + std::string(static_cast<size_t>(-k - 1), '0') + "1";
        }
    }
    return ValueText(v);
}

void CheckValue(double v, AxisScale scale, const char* what) {
    if (!std::isfinite(v)) throw std::invalid_argument(std::string("RenderSeriesSvg: ") + what + " is not finite");
    if (scale == AxisScale::Log && v <= 0) {
        throw std::invalid_argument(std::string("RenderSeriesSvg: ") + what + " is not positive, so it can't go on a log axis");
    }
}

}  // namespace

std::string RenderSeriesSvg(const SeriesDocument& doc, const SeriesViewOptions& view, const SvgOptions& options) {
    CheckOptions(options);
    (void)ToJson(doc);  // every structural rule

    // Colors follow series names in order of first appearance, across panels.
    std::vector<std::string> names;
    for (const SeriesPanel& panel : doc.panels) {
        for (const Series& s : panel.series) {
            if (std::find(names.begin(), names.end(), s.name) == names.end()) names.push_back(s.name);
        }
    }
    auto color = [&](const std::string& name) {
        const size_t i = static_cast<size_t>(std::find(names.begin(), names.end(), name) - names.begin());
        return kSeriesColors[i % 10];
    };

    // Value ranges; a lower bound at or below zero on a log axis is drawn to the axis floor.
    double x_min = INFINITY, x_max = -INFINITY;
    std::vector<std::pair<double, double>> y_range(doc.panels.size(), {INFINITY, -INFINITY});
    for (size_t p = 0; p < doc.panels.size(); ++p) {
        for (const Series& s : doc.panels[p].series) {
            for (size_t i = 0; i < s.size(); ++i) {
                CheckValue(s.x[i], doc.x_scale, "an x value");
                CheckValue(s.y[i], doc.y_scale, "a y value");
                x_min = std::min(x_min, s.x[i]);
                x_max = std::max(x_max, s.x[i]);
                double lo = s.y[i], hi = s.y[i];
                if (s.has_intervals()) {
                    CheckValue(s.hi[i], doc.y_scale, "an interval's upper bound");
                    if (!std::isfinite(s.lo[i])) throw std::invalid_argument("RenderSeriesSvg: an interval's lower bound is not finite");
                    hi = std::max(hi, s.hi[i]);
                    if (doc.y_scale == AxisScale::Linear || s.lo[i] > 0) lo = std::min(lo, s.lo[i]);
                }
                y_range[p].first = std::min(y_range[p].first, lo);
                y_range[p].second = std::max(y_range[p].second, hi);
            }
        }
    }
    if (view.shared_y) {
        std::pair<double, double> all{INFINITY, -INFINITY};
        for (const auto& r : y_range) all = {std::min(all.first, r.first), std::max(all.second, r.second)};
        std::fill(y_range.begin(), y_range.end(), all);
    }
    const Axis x_axis = AxisFor(doc.x_scale, x_min, x_max, false);
    std::vector<Axis> y_axes;
    for (const auto& r : y_range) y_axes.push_back(AxisFor(doc.y_scale, r.first, r.second, r.first >= 0));

    Figure f(options, doc.title.empty() ? "Series" : doc.title);
    const double fs = f.fs();
    double label_w = 0;
    for (const Axis& a : y_axes) {
        for (double t : a.ticks()) label_w = std::max(label_w, TextWidth(TickText(t, a.scale), fs * 0.85));
    }
    const double x0 = fs * 2.2 + label_w + fs * 0.6;
    const double x1 = f.width() - fs * 1.5;
    const double plot_h = std::min(fs * 16, (x1 - x0) * 0.55);
    auto X = [&](double v) { return x0 + x_axis.unit(v) * (x1 - x0); };

    // The legend, once, above the panels.
    double y = f.top();
    double lx = x0;
    double right = x1;
    for (const std::string& name : names) {
        const double w = fs * 1.6 + TextWidth(name, fs) + fs * 1.2;
        if (lx + w > x1 && lx > x0) {
            lx = x0;
            y += fs * 1.4;
        }
        f.line("legend-swatch", lx, y + fs * 0.5, lx + fs * 1.2, y + fs * 0.5, color(name), 2.0);
        f.body() += "<circle class=\"legend-marker\" cx=\"" + Num(lx + fs * 0.6) + "\" cy=\"" + Num(y + fs * 0.5) + "\" r=\"" +
                    Num(fs * 0.3) + "\" fill=\"" + color(name) + "\"/>\n";
        f.text("legend-label", lx + fs * 1.6, y + fs * 0.85, name);
        right = std::max(right, lx + w);
        lx += w;
    }
    y += fs * 1.8;

    for (size_t p = 0; p < doc.panels.size(); ++p) {
        const SeriesPanel& panel = doc.panels[p];
        if (!panel.title.empty()) {
            f.text("panel-title", x0, y + fs, panel.title, "start", kTextColor, " font-weight=\"bold\"");
            y += fs * 1.6;
        }
        const double y_top = y;
        const double y_bottom = y + plot_h;
        const Axis& ya = y_axes[p];
        auto Y = [&](double v) { return y_bottom - ya.unit(v) * (y_bottom - y_top); };

        for (double t : ya.ticks()) f.line("grid", x0, Y(t), x1, Y(t), "#eeeeee");
        f.line("axis", x0, y_top, x0, y_bottom, kAxisColor);
        for (double t : ya.ticks()) {
            f.line("tick", x0 - fs * 0.35, Y(t), x0, Y(t), kAxisColor);
            f.text("tick-label", x0 - fs * 0.5, Y(t) + fs * 0.3, TickText(t, ya.scale), "end", kMutedColor,
                   " font-size=\"" + Num(fs * 0.85) + "\"");
        }
        f.line("axis", x0, y_bottom, x1, y_bottom, kAxisColor);
        for (double t : x_axis.ticks()) {
            f.line("tick", X(t), y_bottom, X(t), y_bottom + fs * 0.35, kAxisColor);
            f.text("tick-label", X(t), y_bottom + fs * 1.35, TickText(t, x_axis.scale), "middle", kMutedColor,
                   " font-size=\"" + Num(fs * 0.85) + "\"");
        }
        if (!doc.y_label.empty()) {
            // Rotated, the label runs along the panel: cut it to the panel's height.
            const auto fit = static_cast<size_t>(std::max(4.0, plot_h / (kCharWidthEm * fs)));
            const double cx = x0 - label_w - fs * 1.2;
            const double cy = (y_top + y_bottom) / 2;
            f.text("axis-label", cx, cy, Truncate(doc.y_label, fit), "middle", kMutedColor,
                   " transform=\"rotate(-90 " + Num(cx) + " " + Num(cy) + ")\"");
        }

        for (const Series& s : panel.series) {
            const char* c = color(s.name);
            if (s.has_intervals()) {
                for (size_t i = 0; i < s.size(); ++i) {
                    const bool floor = doc.y_scale == AxisScale::Log && s.lo[i] <= 0;
                    const double ylo = floor ? y_bottom : Y(std::max(s.lo[i], ya.lo));
                    const double yhi = Y(std::min(s.hi[i], ya.hi));
                    const double xp = X(s.x[i]);
                    f.line("whisker", xp, ylo, xp, yhi, c, 1.25);
                    f.line("whisker-cap", xp - fs * 0.3, yhi, xp + fs * 0.3, yhi, c, 1.25);
                    if (!floor) f.line("whisker-cap", xp - fs * 0.3, ylo, xp + fs * 0.3, ylo, c, 1.25);
                }
            }
            std::vector<size_t> order(s.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return s.x[a] < s.x[b]; });
            if (s.size() > 1) {
                std::string points;
                for (size_t i : order) points += Num(X(s.x[i])) + "," + Num(Y(s.y[i])) + " ";
                points.pop_back();
                f.body() += "<polyline class=\"series\" points=\"" + points + "\" fill=\"none\" stroke=\"" + c +
                            "\" stroke-width=\"2\" stroke-linejoin=\"round\"/>\n";
            }
            for (size_t i : order) {
                std::string tip = s.name + ": " + (doc.x_label.empty() ? "x" : doc.x_label) + " " + ValueText(s.x[i]) + ", " +
                                  (doc.y_label.empty() ? "y" : doc.y_label) + " " + ValueText(s.y[i]);
                if (s.has_intervals()) tip += " [" + ValueText(s.lo[i]) + ", " + ValueText(s.hi[i]) + "]";
                f.body() += "<circle class=\"point\" cx=\"" + Num(X(s.x[i])) + "\" cy=\"" + Num(Y(s.y[i])) + "\" r=\"" +
                            Num(fs * 0.33) + "\" fill=\"" + c + "\" stroke=\"#ffffff\" stroke-width=\"1.5\"><title>" +
                            Escape(tip) + "</title></circle>\n";
            }
        }
        y = y_bottom + fs * 2.2;
    }
    if (!doc.x_label.empty()) {
        f.text("axis-label", (x0 + x1) / 2, y + fs * 0.4, doc.x_label, "middle", kMutedColor);
        y += fs * 1.4;
    }
    if (!doc.interval.empty()) {
        const std::string note = "Whiskers: " + doc.interval;
        f.text("interval-note", x0, y + fs * 0.6, note, "start", kMutedColor, " font-size=\"" + Num(fs * 0.85) + "\"");
        right = std::max(right, x0 + TextWidth(note, fs * 0.85));
        y += fs * 1.4;
    }
    return f.finish(y + fs * 0.6, right + fs);
}

}  // namespace pulsatrix
