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

}  // namespace pulsatrix
