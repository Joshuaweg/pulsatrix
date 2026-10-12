// The SVG scatter plot of a projection document (INT-5).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/viz/colormap.hpp"
#include "pulsatrix/viz/svg.hpp"
#include "svg_detail.hpp"

namespace pulsatrix {

using namespace svg_detail;

namespace {

// Tableau 10, the palette the residue tracks use too, then Tableau 20's light shades, so twelve
// months or twenty classes each get their own color. Groups past twenty reuse colors.
constexpr const char* kGroupColors[] = {"#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1",
                                        "#ff9da7", "#9c755f", "#bab0ac", "#a0cbe8", "#ffbe7d", "#8cd17d", "#ff9d9a",
                                        "#86bcb6", "#f1ce63", "#d4a6c8", "#fabfd2", "#d7b5a6", "#79706e"};
constexpr size_t kNumGroupColors = sizeof kGroupColors / sizeof kGroupColors[0];
constexpr const char* kPointColor = "#4e79a7";

std::string AxisCaption(const ProjectionDocument& doc, int axis) {
    std::string caption = doc.axes[static_cast<size_t>(axis)];
    if (!doc.explained_variance_ratio.empty()) {
        char share[32];
        std::snprintf(share, sizeof share, " (%.1f%% of variance)", 100.0 * doc.explained_variance_ratio[static_cast<size_t>(axis)]);
        caption += share;
    }
    return caption;
}

}  // namespace

std::string RenderProjectionSvg(const ProjectionDocument& doc, int x_axis, int y_axis, const SvgOptions& options) {
    CheckOptions(options);
    // ToJson checks everything a document must hold; reuse it rather than repeat the rules.
    (void)ToJson(doc);
    const auto axes = static_cast<int>(doc.axes.size());
    if (x_axis < 0 || y_axis < 0 || x_axis >= axes || y_axis >= axes || x_axis == y_axis) {
        throw std::invalid_argument("RenderProjectionSvg: x_axis and y_axis must be two different axes of the document");
    }
    const int64_t n = doc.num_points();
    if (n == 0) throw std::invalid_argument("RenderProjectionSvg: no points");
    auto at = [&](int64_t i, int axis) { return static_cast<double>(doc.coordinates[static_cast<size_t>(i * axes + axis)]); };

    // One scale for both axes, so distances on the page are distances in the projection.
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (int64_t i = 0; i < n; ++i) {
        for (int axis : {x_axis, y_axis}) {
            lo = std::min(lo, at(i, axis));
            hi = std::max(hi, at(i, axis));
        }
    }
    const double pad = hi > lo ? 0.05 * (hi - lo) : 1.0;
    lo -= pad;
    hi += pad;

    // Colors: by group, else by value, else one color.
    std::vector<std::string> group_order;
    std::map<std::string, size_t> group_index;
    for (const std::string& g : doc.groups) {
        if (group_index.emplace(g, group_order.size()).second) group_order.push_back(g);
    }
    double vlo = std::numeric_limits<double>::infinity(), vhi = -vlo;
    const bool by_value = doc.groups.empty() && !doc.values.empty();
    if (by_value) {
        for (float v : doc.values) {
            if (std::isfinite(v)) {
                vlo = std::min(vlo, static_cast<double>(v));
                vhi = std::max(vhi, static_cast<double>(v));
            }
        }
        if (!(vhi >= vlo)) vlo = 0, vhi = 1;  // no finite value
        if (vhi == vlo) vhi = vlo + 1;
    }
    auto color_of = [&](int64_t i) -> std::string {
        if (!doc.groups.empty()) return kGroupColors[group_index.at(doc.groups[static_cast<size_t>(i)]) % kNumGroupColors];
        if (by_value) {
            const float v = doc.values[static_cast<size_t>(i)];
            if (!std::isfinite(v)) return kMissingColor;
            return Hex(ViridisColormap(static_cast<float>((v - vlo) / (vhi - vlo))));
        }
        return kPointColor;
    };

    const std::string name = doc.title.empty() ? "Projection of " + std::to_string(n) + " points" : doc.title;
    Figure f(options, name);
    const double fs = f.fs();
    double label_w = 0;
    for (double t : NiceTicks(lo, hi)) label_w = std::max(label_w, TextWidth(ValueText(t), fs * 0.85));
    double legend_w = 0;
    if (!group_order.empty()) {
        for (size_t g = 0; g < std::min<size_t>(group_order.size(), 12); ++g) {
            legend_w = std::max(legend_w, TextWidth(Truncate(group_order[g], 24), fs * 0.9));
        }
        legend_w += fs * 2.5;
    } else if (by_value) {
        legend_w = fs * 2.0 + TextWidth(ValueText(vhi), fs * 0.85) + fs;
    }
    const double x0 = fs * 2.2 + label_w + fs * 0.6;
    const double side = std::max(fs * 8, std::min(f.width() - x0 - fs * 1.5 - legend_w, 520.0));
    const double y_top = f.top() + fs;
    const double y_bottom = y_top + side;
    const double xr = x0 + side;
    auto X = [&](double v) { return x0 + (v - lo) / (hi - lo) * side; };
    auto Y = [&](double v) { return y_bottom - (v - lo) / (hi - lo) * side; };

    if (lo < 0 && hi > 0) {
        f.line("zero", X(0), y_top, X(0), y_bottom, "#e0e0e0");
        f.line("zero", x0, Y(0), xr, Y(0), "#e0e0e0");
    }
    const double r = std::clamp(fs * 0.3 * std::sqrt(200.0 / static_cast<double>(std::max<int64_t>(n, 200))), fs * 0.12, fs * 0.3);
    for (int64_t i = 0; i < n; ++i) {
        std::string tip = doc.labels.empty() ? "point " + std::to_string(i) : doc.labels[static_cast<size_t>(i)];
        if (!doc.groups.empty()) tip += " [" + doc.groups[static_cast<size_t>(i)] + "]";
        if (!doc.values.empty()) tip += ", value " + ValueText(doc.values[static_cast<size_t>(i)]);
        tip += ": " + doc.axes[static_cast<size_t>(x_axis)] + " " + ValueText(at(i, x_axis)) + ", " + doc.axes[static_cast<size_t>(y_axis)] + " " +
               ValueText(at(i, y_axis));
        f.body() += "<circle class=\"point\" cx=\"" + Num(X(at(i, x_axis))) + "\" cy=\"" + Num(Y(at(i, y_axis))) + "\" r=\"" + Num(r) +
                    "\" fill=\"" + color_of(i) + "\" fill-opacity=\"0.8\" stroke=\"#333333\" stroke-width=\"0.3\"><title>" + Escape(tip) +
                    "</title></circle>\n";
    }
    f.y_axis(x0, y_top, y_bottom, lo, hi, AxisCaption(doc, y_axis), label_w);
    f.x_axis(y_bottom, x0, xr, lo, hi, AxisCaption(doc, x_axis));

    double right = xr + fs * 1.5;
    const double lx = xr + fs * 1.5;
    if (!group_order.empty()) {
        const size_t shown = std::min<size_t>(group_order.size(), 12);
        for (size_t g = 0; g < shown; ++g) {
            const double ly = y_top + fs * 1.4 * static_cast<double>(g);
            f.body() += "<circle class=\"legend-swatch\" cx=\"" + Num(lx + fs * 0.4) + "\" cy=\"" + Num(ly + fs * 0.5) + "\" r=\"" +
                        Num(fs * 0.4) + "\" fill=\"" + kGroupColors[g % kNumGroupColors] + "\"/>\n";
            f.text("legend-label", lx + fs * 1.2, ly + fs * 0.85, Truncate(group_order[g], 24));
        }
        if (group_order.size() > shown) {
            f.text("legend-more", lx, y_top + fs * 1.4 * static_cast<double>(shown) + fs * 0.85,
                   std::to_string(group_order.size() - shown) + " more groups", "start", kMutedColor);
        }
        right = lx + legend_w;
    } else if (by_value) {
        ColorBar(f, lx, y_top, std::min(side, fs * 14), vlo, vhi, [](float t) { return ViridisColormap(t); }, "projection-values");
        right = lx + legend_w;
    }
    if (IsNeighborEmbedding(doc)) {
        const char* note = "Neighbors are kept; distances, cluster sizes and densities aren't";
        f.text("projection-note", x0, y_bottom + fs * 4.2, note, "start", kMutedColor);
        return f.finish(y_bottom + fs * 5.4, std::max(right, x0 + TextWidth(note, fs)));
    }
    return f.finish(y_bottom + fs * 4.0, right);
}

}  // namespace pulsatrix
