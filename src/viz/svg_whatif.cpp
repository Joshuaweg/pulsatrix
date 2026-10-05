// SVG views for the counterfactual and sensitivity tools (CFS epic).

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/viz/svg.hpp"
#include "svg_detail.hpp"

namespace pulsatrix {

using namespace svg_detail;

namespace {

constexpr const char* kIceColor = "#7f9fc6";
constexpr const char* kAverageColor = "#c0392b";

// (lo, hi) of every value, padded by 5% of the span, or +-1 around a constant.
std::pair<double, double> PaddedDomain(const std::vector<const std::vector<float>*>& series) {
    double lo = std::numeric_limits<double>::infinity();
    double hi = -lo;
    for (const std::vector<float>* s : series) {
        for (float v : *s) {
            lo = std::min(lo, static_cast<double>(v));
            hi = std::max(hi, static_cast<double>(v));
        }
    }
    if (hi - lo < 1e-12 * std::max(1.0, std::fabs(lo))) {
        return {lo - 1.0, hi + 1.0};
    }
    const double pad = (hi - lo) * 0.05;
    return {lo - pad, hi + pad};
}

std::string Points(const std::vector<float>& grid, const float* ys, double x0, double x1,
                   double y_top, double y_bottom, double xlo, double xhi, double ylo, double yhi) {
    std::string pts;
    for (size_t k = 0; k < grid.size(); ++k) {
        const double x = x0 + (grid[k] - xlo) / (xhi - xlo) * (x1 - x0);
        const double y = y_bottom - (ys[k] - ylo) / (yhi - ylo) * (y_bottom - y_top);
        if (k > 0) pts += ' ';
        pts += Num(x) + "," + Num(y);
    }
    return pts;
}

}  // namespace

// ---- partial dependence ---------------------------------------------------------------------

std::string RenderPartialDependenceSvg(const PartialDependenceDocument& doc, const PartialDependenceSvgOptions& pd,
                                       const SvgOptions& options) {
    CheckOptions(options);
    if (pd.max_curves < 0) {
        throw std::invalid_argument("RenderPartialDependenceSvg: max_curves must not be negative");
    }
    (void)ToJson(doc);  // the document's own consistency checks
    CheckFinite(doc.grid, "RenderPartialDependenceSvg: grid");
    CheckFinite(doc.partial_dependence, "RenderPartialDependenceSvg: partial dependence");
    CheckFinite(doc.ice, "RenderPartialDependenceSvg: ICE");
    CheckFinite(doc.feature_values, "RenderPartialDependenceSvg: feature values");
    const bool has_ice = doc.num_instances > 0;
    if (pd.style != IceStyle::Raw && !has_ice) {
        throw std::invalid_argument("RenderPartialDependenceSvg: centered and derivative views need ICE curves");
    }
    if (pd.style == IceStyle::Derivative && doc.grid.size() < 2) {
        throw std::invalid_argument("RenderPartialDependenceSvg: a derivative needs at least 2 grid points");
    }

    // The curves in the chosen style, and their average.
    const size_t g = doc.grid.size();
    std::vector<float> curves = doc.ice;
    if (has_ice && pd.style != IceStyle::Raw) {
        IceResult r = ToIceResult(doc);
        curves = pd.style == IceStyle::Centered ? r.centered(0) : r.derivative();
    }
    std::vector<float> average = doc.partial_dependence;
    if (pd.style != IceStyle::Raw) {
        IceResult r;
        r.grid = doc.grid;
        r.num_instances = doc.num_instances;
        r.curves = curves;
        average = r.partial_dependence();
    }

    // Which curves to draw: evenly spaced through the instances.
    const auto n = static_cast<size_t>(doc.num_instances);
    std::vector<size_t> shown;
    const size_t limit = std::min(n, static_cast<size_t>(pd.max_curves));
    for (size_t j = 0; j < limit; ++j) {
        shown.push_back(j * n / limit);
    }
    std::vector<std::vector<float>> shown_curves;
    for (size_t i : shown) {
        shown_curves.emplace_back(curves.begin() + static_cast<std::ptrdiff_t>(i * g),
                                  curves.begin() + static_cast<std::ptrdiff_t>((i + 1) * g));
    }
    std::vector<const std::vector<float>*> series{&average};
    for (const auto& c : shown_curves) series.push_back(&c);
    auto [ylo, yhi] = PaddedDomain(series);
    if (pd.style != IceStyle::Raw) {
        ylo = std::min(ylo, 0.0);
        yhi = std::max(yhi, 0.0);
    }
    double xlo = doc.grid.front();
    double xhi = doc.grid.back();
    if (g == 1) {
        xlo -= 1.0;
        xhi += 1.0;
    }

    const std::string feature = doc.feature.empty() ? "feature" : doc.feature;
    const std::string target = doc.target.empty() ? "prediction" : doc.target;
    std::string y_caption = target;
    if (pd.style == IceStyle::Centered) {
        y_caption = target + " - " + target + " at " + ValueText(doc.grid.front());
    } else if (pd.style == IceStyle::Derivative) {
        y_caption = "d " + target + " / d " + feature;
    }
    const char* style_name = pd.style == IceStyle::Raw ? "" : pd.style == IceStyle::Centered ? "centered " : "derivative ";
    Figure f(options, std::string("Partial dependence of ") + target + " on " + feature + " (" + style_name +
                          (has_ice ? "ICE)" : "average)"));
    const double fs = f.fs();
    double label_w = 0;
    for (double t : NiceTicks(ylo, yhi)) {
        label_w = std::max(label_w, TextWidth(ValueText(t), fs * 0.85));
    }
    const double x0 = fs * 2.2 + label_w + fs * 0.6;
    const double x1 = f.width() - fs * 1.5;
    const double y_top = f.top() + fs * 2.0;  // room for the legend
    const double y_bottom = y_top + std::max(120.0, (x1 - x0) * 0.6);

    // Legend.
    {
        double lx = x0;
        const double ly = f.top() + fs * 0.6;
        if (!shown.empty()) {
            f.line("legend-ice", lx, ly, lx + fs * 1.5, ly, kIceColor, 1.0);
            const std::string t = "ICE (" + std::to_string(shown.size()) + " of " + std::to_string(n) + " shown)";
            f.text("legend-label", lx + fs * 2.0, ly + fs * 0.35, t);
            lx += fs * 2.5 + TextWidth(t, fs) + fs;
        }
        f.line("legend-average", lx, ly, lx + fs * 1.5, ly, kAverageColor, 2.5);
        f.text("legend-label", lx + fs * 2.0, ly + fs * 0.35,
               has_ice ? "average of " + std::to_string(n) + " curves" : "partial dependence");
    }

    if (pd.style != IceStyle::Raw && ylo < 0.0 && yhi > 0.0) {
        const double zy = y_bottom - (0.0 - ylo) / (yhi - ylo) * (y_bottom - y_top);
        f.line("zero-line", x0, zy, x1, zy, kMutedColor, 1.0, "3 3");
    }
    f.body() += "<g class=\"ice-curves\" fill=\"none\" stroke=\"" + std::string(kIceColor) +
                "\" stroke-width=\"1\" stroke-opacity=\"0.45\">\n";
    for (const auto& c : shown_curves) {
        f.body() += "<polyline class=\"ice\" points=\"" +
                    Points(doc.grid, c.data(), x0, x1, y_top, y_bottom, xlo, xhi, ylo, yhi) + "\"/>\n";
    }
    f.body() += "</g>\n";
    f.body() += "<polyline class=\"average\" fill=\"none\" stroke=\"" + std::string(kAverageColor) +
                "\" stroke-width=\"2.5\" points=\"" +
                Points(doc.grid, average.data(), x0, x1, y_top, y_bottom, xlo, xhi, ylo, yhi) + "\"/>\n";

    // Rug: where the instances' own values fall, inside the grid's range.
    if (!doc.feature_values.empty()) {
        f.body() += "<g class=\"rug\" stroke=\"" + std::string(kTextColor) + "\" stroke-opacity=\"0.5\">\n";
        for (float v : doc.feature_values) {
            if (v < xlo || v > xhi) continue;
            const double x = x0 + (v - xlo) / (xhi - xlo) * (x1 - x0);
            f.body() += "<line x1=\"" + Num(x) + "\" y1=\"" + Num(y_bottom) + "\" x2=\"" + Num(x) + "\" y2=\"" +
                        Num(y_bottom - fs * 0.6) + "\"/>\n";
        }
        f.body() += "</g>\n";
    }

    f.y_axis(x0, y_top, y_bottom, ylo, yhi, y_caption, label_w);
    f.x_axis(y_bottom, x0, x1, xlo, xhi, feature);
    return f.finish(y_bottom + fs * 4.0);
}

// ---- tornado ------------------------------------------------------------------------------------

std::string RenderTornadoSvg(const SensitivityDocument& doc, int top_k, const SvgOptions& options) {
    CheckOptions(options);
    if (doc.features.empty()) {
        throw std::invalid_argument("RenderTornadoSvg: no features");
    }
    if (top_k < 1) {
        throw std::invalid_argument("RenderTornadoSvg: top_k must be at least 1");
    }
    CheckFinite({doc.output}, "RenderTornadoSvg: output");
    for (const auto& f : doc.features) {
        CheckFinite({f.value, f.low, f.high, f.output_low, f.output_high}, "RenderTornadoSvg: feature values");
    }
    // Largest swing first; ties keep document order.
    std::vector<size_t> order(doc.features.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    auto swing = [&](size_t i) { return std::fabs(doc.features[i].output_high - doc.features[i].output_low); };
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return swing(a) > swing(b); });
    order.resize(std::min(order.size(), static_cast<size_t>(top_k)));

    double lo = doc.output, hi = doc.output;
    for (size_t i : order) {
        lo = std::min({lo, static_cast<double>(doc.features[i].output_low), static_cast<double>(doc.features[i].output_high)});
        hi = std::max({hi, static_cast<double>(doc.features[i].output_low), static_cast<double>(doc.features[i].output_high)});
    }
    if (hi - lo < 1e-12 * std::max(1.0, std::fabs(lo))) {
        lo -= 1.0;
        hi += 1.0;
    } else {
        const double pad = (hi - lo) * 0.12;  // room for the value labels at the bar ends
        lo -= pad;
        hi += pad;
    }

    const std::string target = doc.target.empty() ? "output" : doc.target;
    Figure f(options, "Sensitivity of " + target + " to " + std::to_string(order.size()) + " features");
    const double fs = f.fs();
    const size_t max_label_chars = static_cast<size_t>(std::max(4.0, f.width() * 0.3 / (kCharWidthEm * fs)));
    std::vector<std::string> labels;
    double label_w = 0;
    for (size_t i : order) {
        labels.push_back(Truncate(doc.features[i].name + " = " + ValueText(doc.features[i].value), max_label_chars));
        label_w = std::max(label_w, TextWidth(labels.back(), fs));
    }
    const double x0 = fs + label_w + fs * 0.75;
    const double x1 = f.width() - fs * 1.5;
    auto X = [&](double v) { return x0 + (v - lo) / (hi - lo) * (x1 - x0); };
    const std::string high_color = Hex(DivergingColormap(0.8f));
    const std::string low_color = Hex(DivergingColormap(-0.8f));

    // Legend.
    const double ly = f.top() + fs * 0.6;
    f.body() += "<rect class=\"legend-high\" x=\"" + Num(x0) + "\" y=\"" + Num(ly - fs * 0.4) + "\" width=\"" +
                Num(fs) + "\" height=\"" + Num(fs * 0.8) + "\" fill=\"" + high_color + "\"/>\n";
    f.text("legend-label", x0 + fs * 1.4, ly + fs * 0.35, "high value");
    const double lx = x0 + fs * 1.4 + TextWidth("high value", fs) + fs;
    f.body() += "<rect class=\"legend-low\" x=\"" + Num(lx) + "\" y=\"" + Num(ly - fs * 0.4) + "\" width=\"" + Num(fs) +
                "\" height=\"" + Num(fs * 0.8) + "\" fill=\"" + low_color + "\"/>\n";
    f.text("legend-label", lx + fs * 1.4, ly + fs * 0.35, "low value");

    const double row_h = fs * 2.6;
    const double bar_h = fs * 0.9;
    const double y0 = f.top() + fs * 1.8;
    const double base_x = X(doc.output);
    const std::string small = " font-size=\"" + Num(fs * 0.8) + "\"";
    for (size_t r = 0; r < order.size(); ++r) {
        const auto& ft = doc.features[order[r]];
        const double cy = y0 + row_h * (static_cast<double>(r) + 0.5);
        f.body() += "<g class=\"feature-row\">\n";
        f.text("feature-label", x0 - fs * 0.75, cy + fs * 0.35, labels[r], "end");
        auto bar = [&](const char* cls, double out, double input, double top, const std::string& color) {
            const double a = std::min(base_x, X(out)), b = std::max(base_x, X(out));
            f.body() += std::string("<rect class=\"") + cls + "\" x=\"" + Num(a) + "\" y=\"" + Num(top) + "\" width=\"" +
                        Num(std::max(b - a, 0.5)) + "\" height=\"" + Num(bar_h) + "\" fill=\"" + color + "\">" +
                        "<title>" + Escape(ft.name + " = " + ValueText(input) + ": " + target + " " + ValueText(out)) +
                        "</title></rect>\n";
            const bool right = X(out) >= base_x;
            f.text("bar-label", right ? b + fs * 0.3 : a - fs * 0.3, top + bar_h * 0.8, ValueText(input),
                   right ? "start" : "end", kMutedColor, small);
        };
        bar("bar-high", ft.output_high, ft.high, cy - bar_h, high_color);
        bar("bar-low", ft.output_low, ft.low, cy, low_color);
        f.body() += "</g>\n";
    }
    const double plot_bottom = y0 + row_h * static_cast<double>(order.size());
    f.line("base-line", base_x, y0, base_x, plot_bottom, kMutedColor, 1.0, "3 3");
    f.x_axis(plot_bottom + fs * 0.25, x0, x1, lo, hi, target + " (unchanged: " + ValueText(doc.output) + ")");
    return f.finish(plot_bottom + fs * 4.0);
}

// ---- counterfactual -------------------------------------------------------------------------

std::string RenderCounterfactualSvg(const CounterfactualDocument& doc, int max_rows, const SvgOptions& options) {
    CheckOptions(options);
    if (max_rows < 1) {
        throw std::invalid_argument("RenderCounterfactualSvg: max_rows must be at least 1");
    }
    CheckFinite({doc.output_before, doc.output_after}, "RenderCounterfactualSvg: outputs");
    std::vector<size_t> changed;
    for (size_t i = 0; i < doc.features.size(); ++i) {
        const auto& f = doc.features[i];
        CheckFinite({f.original, f.counterfactual, f.scale}, "RenderCounterfactualSvg: feature values");
        if (!(f.scale > 0.0f)) {
            throw std::invalid_argument("RenderCounterfactualSvg: every scale must be positive");
        }
        if (std::fabs(f.counterfactual - f.original) > 1e-3f * f.scale) {
            changed.push_back(i);
        }
    }
    auto cost = [&](size_t i) {
        const auto& f = doc.features[i];
        return (f.counterfactual - f.original) / f.scale;
    };
    std::stable_sort(changed.begin(), changed.end(),
                     [&](size_t a, size_t b) { return std::fabs(cost(a)) > std::fabs(cost(b)); });
    double total = 0.0;
    for (size_t i : changed) total += std::fabs(cost(i));
    const size_t shown = std::min(changed.size(), static_cast<size_t>(max_rows));

    const std::string target = doc.target.empty() ? "the target" : doc.target;
    Figure f(options, std::string("Counterfactual ") + (doc.valid ? "reaching " : "not reaching ") + target);
    const double fs = f.fs();
    double y = f.top() + fs * 0.8;
    f.text(doc.valid ? "verdict valid" : "verdict invalid", fs, y,
           std::string(doc.valid ? "Reaches " : "Does not reach ") + target, "start", doc.valid ? "#2e7d32" : "#c62828",
           " font-weight=\"bold\"");
    y += fs * 1.5;
    f.text("outputs", fs, y, "output " + ValueText(doc.output_before) + " \xE2\x86\x92 " + ValueText(doc.output_after), "start",
           kMutedColor);
    y += fs * 1.2;

    // Rows: name | old -> new | bar of the change in scale units, on an axis through zero.
    const size_t max_label_chars = static_cast<size_t>(std::max(4.0, f.width() * 0.25 / (kCharWidthEm * fs)));
    std::vector<std::string> names, values;
    double name_w = 0, value_w = 0, max_abs = 0;
    for (size_t r = 0; r < shown; ++r) {
        const auto& ft = doc.features[changed[r]];
        names.push_back(Truncate(ft.name, max_label_chars));
        values.push_back(ValueText(ft.original) + " \xE2\x86\x92 " + ValueText(ft.counterfactual));
        name_w = std::max(name_w, TextWidth(names.back(), fs));
        value_w = std::max(value_w, TextWidth(values.back(), fs));
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(cost(changed[r]))));
    }
    const double x_name = fs + name_w;
    const double x_value = x_name + fs * 1.0;
    const double x0 = x_value + value_w + fs * 1.5;
    const double x1 = f.width() - fs * 1.5;
    const double lim = max_abs > 0 ? max_abs * 1.05 : 1.0;
    auto X = [&](double v) { return x0 + (v + lim) / (2 * lim) * (x1 - x0); };
    const double row_h = fs * 1.9;
    const double y0 = y + fs * 0.6;
    for (size_t r = 0; r < shown; ++r) {
        const size_t i = changed[r];
        const double cy = y0 + row_h * (static_cast<double>(r) + 0.5);
        const double c = cost(i);
        f.body() += "<g class=\"changed-feature\">\n";
        f.text("feature-label", x_name, cy + fs * 0.35, names[r], "end");
        f.text("feature-change", x_value, cy + fs * 0.35, values[r], "start", kTextColor,
               std::string(" font-family=\"") + kMonoFont + "\"");
        const double a = std::min(X(0), X(c)), b = std::max(X(0), X(c));
        f.body() += "<rect class=\"bar\" x=\"" + Num(a) + "\" y=\"" + Num(cy - fs * 0.45) + "\" width=\"" +
                    Num(std::max(b - a, 0.5)) + "\" height=\"" + Num(fs * 0.9) + "\" fill=\"" +
                    Hex(DivergingColormap(NormalizeSigned(static_cast<float>(c), static_cast<float>(lim)))) +
                    "\" stroke=\"#555555\" stroke-width=\"0.3\"/>\n";
        f.body() += "</g>\n";
    }
    double bottom = y0 + row_h * static_cast<double>(shown);
    if (shown > 0) {
        f.line("zero-line", X(0), y0, X(0), bottom, kMutedColor, 1.0, "3 3");
        f.x_axis(bottom + fs * 0.25, x0, x1, -lim, lim, "change / scale");
        bottom += fs * 3.2;
    }
    std::string summary = std::to_string(changed.size()) + " of " + std::to_string(doc.features.size()) +
                          " features changed, distance " + ValueText(total);
    if (changed.empty()) summary = "No feature changed";
    if (changed.size() > shown) summary += " (" + std::to_string(changed.size() - shown) + " more not shown)";
    f.text("summary", fs, bottom + fs * 0.8, summary, "start", kMutedColor);
    return f.finish(bottom + fs * 1.8);
}

}  // namespace pulsatrix
