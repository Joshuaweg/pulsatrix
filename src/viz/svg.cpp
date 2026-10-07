#include "pulsatrix/viz/svg.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/viz/colormap.hpp"
#include "pulsatrix/viz/plot_data.hpp"
#include "svg_detail.hpp"

// Static, so these definitions never clash with a program that also compiles stb_image_write.
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace pulsatrix {

using namespace svg_detail;

// ---- bar chart ------------------------------------------------------------------------------

std::string RenderBarChartSvg(const AttributionDocument& doc, int top_k, const SvgOptions& options) {
    CheckOptions(options);
    CheckFinite(doc.values, "RenderBarChartSvg");
    BarSeries bars = ToFeatureImportanceBars(ToHostAttribution(doc), top_k);

    Figure f(options, "Feature attribution (" + doc.method + ")");
    const double fs = f.fs();
    double min_v = 0, max_v = 0;
    size_t label_chars = 0, value_chars = 0;
    for (size_t i = 0; i < bars.values.size(); ++i) {
        min_v = std::min(min_v, static_cast<double>(bars.values[i]));
        max_v = std::max(max_v, static_cast<double>(bars.values[i]));
        label_chars = std::max(label_chars, Columns(std::string_view(bars.labels[i])));
        value_chars = std::max(value_chars, ValueText(bars.values[i]).size());
    }
    const size_t max_label_chars = static_cast<size_t>(std::max(4.0, f.width() * 0.35 / (kCharWidthEm * fs)));
    label_chars = std::min(label_chars, max_label_chars);
    const double label_w = static_cast<double>(label_chars) * kCharWidthEm * fs + fs;
    const double value_w = static_cast<double>(value_chars) * kCharWidthEm * fs + fs * 0.5;
    const double x0 = label_w + (min_v < 0 ? value_w : 0.0);
    const double x1 = f.width() - fs - (max_v > 0 ? value_w : 0.0);
    auto [lo, hi] = ZeroAnchoredDomain(min_v, max_v, 0.0);
    auto X = [&](double v) { return x0 + (v - lo) / (hi - lo) * (x1 - x0); };

    const double row_h = fs * 2.0;
    const double bar_h = fs * 1.3;
    const double y0 = f.top() + fs * 0.5;
    const float max_abs = static_cast<float>(std::max(std::abs(min_v), std::abs(max_v)));
    for (size_t i = 0; i < bars.values.size(); ++i) {
        const double v = bars.values[i];
        const double cy = y0 + row_h * (static_cast<double>(i) + 0.5);
        const double left = std::min(X(0), X(v));
        const RgbColor fill = DivergingColormap(NormalizeSigned(bars.values[i], max_abs));
        const RgbColor edge = DivergingColormap(v < 0 ? -1.0f : 1.0f);
        f.body() += "<rect class=\"bar\" x=\"" + Num(left) + "\" y=\"" + Num(cy - bar_h / 2) + "\" width=\"" +
                    Num(std::abs(X(v) - X(0))) + "\" height=\"" + Num(bar_h) + "\" fill=\"" + Hex(fill) + "\" stroke=\"" +
                    Hex(edge) + "\" stroke-width=\"0.75\"><title>" + Escape(bars.labels[i]) + ": " + ValueText(v) +
                    "</title></rect>\n";
        f.text("feature-label", label_w - fs * 0.5, cy + fs * 0.35, Truncate(bars.labels[i], max_label_chars), "end");
        const bool negative = v < 0;
        f.text("value-label", negative ? X(v) - fs * 0.3 : X(v) + fs * 0.3, cy + fs * 0.35, ValueText(v),
               negative ? "end" : "start", kMutedColor);
    }
    const double plot_bottom = y0 + row_h * static_cast<double>(bars.values.size());
    f.line("zero-line", X(0), y0, X(0), plot_bottom, kTextColor);
    f.x_axis(plot_bottom + fs * 0.25, x0, x1, lo, hi, "attribution (" + doc.method + ")");
    return f.finish(plot_bottom + fs * 4.0);
}

// ---- waterfall ------------------------------------------------------------------------------

std::string RenderWaterfallSvg(const AttributionDocument& doc, float baseline_value, const SvgOptions& options) {
    CheckOptions(options);
    CheckFinite(doc.values, "RenderWaterfallSvg");
    if (!std::isfinite(baseline_value)) {
        throw std::invalid_argument("RenderWaterfallSvg: the baseline is not finite");
    }
    std::vector<WaterfallStep> steps = ToWaterfallSteps(ToHostAttribution(doc), baseline_value);
    std::vector<WaterfallBar> bars = ToWaterfallBars(steps, baseline_value);
    std::vector<std::string> names = FeatureNames(doc, steps.size());

    Figure f(options, "Waterfall from baseline to prediction (" + doc.method + ")");
    const double fs = f.fs();
    double lo = baseline_value, hi = baseline_value;
    for (const WaterfallBar& b : bars) {
        lo = std::min(lo, static_cast<double>(b.bottom));
        hi = std::max(hi, static_cast<double>(b.top));
    }
    if (hi - lo == 0.0) {
        lo -= 1.0;
        hi += 1.0;
    }
    const double pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;

    const size_t n = steps.size();
    const double axis_w = fs * 4.5;
    const double x0 = axis_w;
    const double final_value = steps.back().cumulative;
    const double right_label_w = std::max(TextWidth("base " + ValueText(baseline_value), fs),
                                          TextWidth("= " + ValueText(final_value), fs) * 1.1);
    const double x1 = f.width() - fs * 0.6 - right_label_w;
    const double slot = (x1 - x0) / static_cast<double>(n);
    const double col_w = slot * 0.7;
    const size_t max_label_chars = 24;
    double longest = 0;
    for (const std::string& name : names) {
        longest = std::max(longest, TextWidth(Truncate(name, max_label_chars), fs));
    }
    const bool rotate = longest > slot * 0.95;
    const double label_h = rotate ? longest * 0.75 + fs : fs * 1.5;

    const double y_top = f.top() + fs;
    const double plot_h = std::max(fs * 14.0, f.width() * 0.45);
    auto Y = [&](double v) { return y_top + (hi - v) / (hi - lo) * plot_h; };
    const double y_bottom = y_top + plot_h;

    // Value axis on the left.
    f.line("axis", x0, y_top, x0, y_bottom, kAxisColor);
    for (double t : NiceTicks(lo, hi)) {
        f.line("tick", x0 - fs * 0.35, Y(t), x0, Y(t), kAxisColor);
        f.line("grid", x0, Y(t), x1, Y(t), "#eeeeee");
        f.text("tick-label", x0 - fs * 0.5, Y(t) + fs * 0.35, ValueText(t), "end", kMutedColor,
               " font-size=\"" + Num(fs * 0.85) + "\"");
    }
    f.line("baseline", x0, Y(baseline_value), x1, Y(baseline_value), kMutedColor, 1.0, "4 3");
    f.text("baseline-label", x1 + fs * 0.3, Y(baseline_value) + fs * 0.35, "base " + ValueText(baseline_value), "start",
           kMutedColor);

    const RgbColor up = DivergingColormap(1.0f);
    const RgbColor down = DivergingColormap(-1.0f);
    for (size_t i = 0; i < n; ++i) {
        const double cx = x0 + slot * (static_cast<double>(i) + 0.5);
        const WaterfallBar& b = bars[i];
        f.body() += "<rect class=\"step\" x=\"" + Num(cx - col_w / 2) + "\" y=\"" + Num(Y(b.top)) + "\" width=\"" +
                    Num(col_w) + "\" height=\"" + Num(Y(b.bottom) - Y(b.top)) + "\" fill=\"" + Hex(b.increase ? up : down) +
                    "\"><title>" + Escape(names[i]) + ": " + (steps[i].delta >= 0 ? "+" : "") + ValueText(steps[i].delta) +
                    " (running total " + ValueText(steps[i].cumulative) + ")</title></rect>\n";
        if (i + 1 < n) {
            f.line("connector", cx + col_w / 2, Y(steps[i].cumulative), cx + slot - col_w / 2, Y(steps[i].cumulative),
                   kMutedColor, 0.75);
        }
        const std::string label = Truncate(names[i], max_label_chars);
        if (rotate) {
            const double ly = y_bottom + fs * 0.8;
            f.text("feature-label", cx, ly, label, "end", kTextColor,
                   " transform=\"rotate(-45 " + Num(cx) + " " + Num(ly) + ")\"");
        } else {
            f.text("feature-label", cx, y_bottom + fs * 1.3, label, "middle");
        }
    }
    f.line("total", x1 - slot * 0.15, Y(final_value), x1, Y(final_value), kTextColor, 1.5);
    // Keep the total's label clear of the baseline's when the two values are close.
    double total_y = Y(final_value) + fs * 0.35;
    const double base_y = Y(baseline_value) + fs * 0.35;
    if (std::abs(total_y - base_y) < fs * 1.1) {
        total_y = final_value < baseline_value ? base_y + fs * 1.1 : base_y - fs * 1.1;
    }
    f.text("total-label", x1 + fs * 0.3, total_y, "= " + ValueText(final_value), "start", kTextColor,
           " font-weight=\"bold\"");
    return f.finish(y_bottom + label_h + fs);
}

// ---- heatmap --------------------------------------------------------------------------------

std::string RenderHeatmapSvg(const HeatmapDocument& doc, const SvgOptions& options) {
    CheckOptions(options);
    if (doc.rows <= 0 || doc.cols <= 0) {
        throw std::invalid_argument("RenderHeatmapSvg: the grid is empty");
    }
    if (doc.cols > std::numeric_limits<int64_t>::max() / doc.rows ||
        static_cast<size_t>(doc.rows * doc.cols) != doc.values.size()) {
        throw std::invalid_argument("RenderHeatmapSvg: values.size() is not rows * cols");
    }
    if ((!doc.row_labels.empty() && static_cast<int64_t>(doc.row_labels.size()) != doc.rows) ||
        (!doc.col_labels.empty() && static_cast<int64_t>(doc.col_labels.size()) != doc.cols)) {
        throw std::invalid_argument("RenderHeatmapSvg: a label list doesn't match the grid");
    }

    HeatmapGrid finite{{}, 1, 0};
    for (float v : doc.values) {
        if (std::isfinite(v)) finite.values.push_back(v);
    }
    finite.cols = static_cast<int64_t>(finite.values.size());
    const HeatmapColorScale scale = ComputeHeatmapColorScale(finite);
    auto color_of = [&](float v) {
        return scale.is_signed ? DivergingColormap(NormalizeSigned(v, scale.scale_max))
                               : ViridisColormap(NormalizeUnsigned(v, scale.scale_max));
    };

    Figure f(options, "Heatmap" + (doc.title.empty() ? std::string() : ": " + doc.title));
    const double fs = f.fs();
    const size_t max_label_chars = 16;
    double row_label_w = 0, col_label_h = 0;
    for (const auto& l : doc.row_labels) row_label_w = std::max(row_label_w, TextWidth(Truncate(l, max_label_chars), fs));
    for (const auto& l : doc.col_labels) col_label_h = std::max(col_label_h, TextWidth(Truncate(l, max_label_chars), fs));
    if (row_label_w > 0) row_label_w += fs * 0.5;
    if (col_label_h > 0) col_label_h = col_label_h * 0.75 + fs;

    const double colorbar_w = fs * 5.5;
    const double x0 = fs + row_label_w;
    const double avail = f.width() - x0 - colorbar_w - fs;
    const double cell = std::min(fs * 4.0, avail / static_cast<double>(doc.cols));
    const double y0 = f.top() + col_label_h + fs * 0.5;
    const double grid_w = cell * static_cast<double>(doc.cols);
    const double grid_h = cell * static_cast<double>(doc.rows);

    for (int64_t c = 0; c < static_cast<int64_t>(doc.col_labels.size()); ++c) {
        const double cx = x0 + cell * (static_cast<double>(c) + 0.5);
        const double ly = y0 - fs * 0.4;
        f.text("col-label", cx, ly, Truncate(doc.col_labels[static_cast<size_t>(c)], max_label_chars), "start", kTextColor,
               " transform=\"rotate(-45 " + Num(cx) + " " + Num(ly) + ")\"");
    }
    for (int64_t r = 0; r < static_cast<int64_t>(doc.row_labels.size()); ++r) {
        f.text("row-label", x0 - fs * 0.4, y0 + cell * (static_cast<double>(r) + 0.5) + fs * 0.35,
               Truncate(doc.row_labels[static_cast<size_t>(r)], max_label_chars), "end");
    }

    const int64_t cells = doc.rows * doc.cols;
    if (cells <= kMaxVectorCells) {
        for (int64_t r = 0; r < doc.rows; ++r) {
            for (int64_t c = 0; c < doc.cols; ++c) {
                const float v = doc.values[static_cast<size_t>(r * doc.cols + c)];
                const bool missing = !std::isfinite(v);
                std::string where = doc.row_labels.empty() ? "row " + std::to_string(r) : doc.row_labels[static_cast<size_t>(r)];
                where += ", " + (doc.col_labels.empty() ? "col " + std::to_string(c) : doc.col_labels[static_cast<size_t>(c)]);
                f.body() += std::string("<rect class=\"cell") + (missing ? " missing" : "") + "\" x=\"" +
                            Num(x0 + cell * static_cast<double>(c)) + "\" y=\"" + Num(y0 + cell * static_cast<double>(r)) +
                            "\" width=\"" + Num(cell) + "\" height=\"" + Num(cell) + "\" shape-rendering=\"crispEdges\" fill=\"" +
                            (missing ? std::string(kMissingColor) : Hex(color_of(v))) + "\"><title>" + Escape(where) + ": " +
                            (missing ? (std::isnan(v) ? "NaN" : (v > 0 ? "inf" : "-inf")) : ValueText(v)) +
                            "</title></rect>\n";
            }
        }
    } else {
        std::vector<unsigned char> rgb(static_cast<size_t>(cells) * 3);
        auto byte = [](float v) { return static_cast<unsigned char>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
        for (int64_t i = 0; i < cells; ++i) {
            const float v = doc.values[static_cast<size_t>(i)];
            RgbColor c = std::isfinite(v) ? color_of(v) : RgbColor{189.0f / 255.0f, 189.0f / 255.0f, 189.0f / 255.0f};
            rgb[static_cast<size_t>(i) * 3] = byte(c.r);
            rgb[static_cast<size_t>(i) * 3 + 1] = byte(c.g);
            rgb[static_cast<size_t>(i) * 3 + 2] = byte(c.b);
        }
        std::vector<unsigned char> png;
        auto sink = [](void* ctx, void* data, int size) {
            auto* out = static_cast<std::vector<unsigned char>*>(ctx);
            out->insert(out->end(), static_cast<unsigned char*>(data), static_cast<unsigned char*>(data) + size);
        };
        if (doc.cols > std::numeric_limits<int>::max() / 3 || doc.rows > std::numeric_limits<int>::max() ||
            stbi_write_png_to_func(sink, &png, static_cast<int>(doc.cols), static_cast<int>(doc.rows), 3, rgb.data(),
                                   static_cast<int>(doc.cols) * 3) == 0) {
            throw std::invalid_argument("RenderHeatmapSvg: the grid is too large to encode");
        }
        f.body() += "<image class=\"heatmap-image\" x=\"" + Num(x0) + "\" y=\"" + Num(y0) + "\" width=\"" + Num(grid_w) +
                    "\" height=\"" + Num(grid_h) +
                    "\" preserveAspectRatio=\"none\" style=\"image-rendering:pixelated\" href=\"data:image/png;base64," +
                    Base64(png) + "\"/>\n";
    }
    f.body() += "<rect class=\"frame\" x=\"" + Num(x0) + "\" y=\"" + Num(y0) + "\" width=\"" + Num(grid_w) + "\" height=\"" +
                Num(grid_h) + "\" fill=\"none\" stroke=\"" + kAxisColor + "\" stroke-width=\"0.5\"/>\n";

    const double bar_h = std::max(fs * 6.0, std::min(grid_h, fs * 16.0));
    const double lo = scale.is_signed ? -static_cast<double>(scale.scale_max) : 0.0;
    const double hi = scale.scale_max;
    ColorBar(f, x0 + grid_w + fs, y0, bar_h, lo, hi,
             [&](float t) { return color_of(static_cast<float>(lo + t * (hi - lo))); }, "heatmap-scale");
    // A small grid doesn't need the full width: end the figure after the color bar's labels.
    const double used = std::max(x0 + grid_w + fs * 2.0 + TextWidth("-" + ValueText(hi), fs * 0.85) + fs * 1.5,
                                 options.title.empty() ? 0.0 : TextWidth(options.title, fs * 1.25) + fs * 2.0);
    return f.finish(y0 + std::max(grid_h, bar_h) + fs, used);
}

// ---- token strip ----------------------------------------------------------------------------

std::string RenderTokenStripSvg(const TokenRelevanceDocument& doc, const SvgOptions& options) {
    CheckOptions(options);
    if (doc.tokens.size() != doc.relevance.size()) {
        throw std::invalid_argument("RenderTokenStripSvg: needs one relevance score per token");
    }
    if (doc.tokens.empty()) {
        throw std::invalid_argument("RenderTokenStripSvg: there are no tokens");
    }
    if (!doc.scored.empty() && doc.scored.size() != doc.tokens.size()) {
        throw std::invalid_argument("RenderTokenStripSvg: needs one scored flag per token");
    }
    float max_abs = 0.0f;
    for (size_t i = 0; i < doc.relevance.size(); ++i) {
        if (doc.is_scored(i) && std::isfinite(doc.relevance[i])) max_abs = std::max(max_abs, std::abs(doc.relevance[i]));
    }

    Figure f(options, std::string(doc.granularity == "word" ? "Word" : "Token") + " relevance (" + doc.method + ")");
    const double fs = f.fs();
    const double char_w = kCharWidthEm * fs;
    const double line_h = fs * 1.9;
    const double box_h = fs * 1.45;
    const double left = fs;
    const double right = f.width() - fs;
    const size_t max_chars = static_cast<size_t>(std::max(1.0, std::floor((right - left) / char_w)));

    double x = left;
    double y = f.top() + fs * 0.25;
    for (size_t i = 0; i < doc.tokens.size(); ++i) {
        // A newline inside a token is shown as a symbol and ends the line after the token.
        std::string shown;
        bool breaks_line = false;
        std::string_view token = doc.tokens[i];
        while (true) {
            size_t nl = token.find('\n');
            for (const std::string& ch : Characters(token.substr(0, nl))) {
                shown += ch;
            }
            if (nl == std::string_view::npos) break;
            shown += "\xE2\x86\xB5";  // a visible return symbol
            breaks_line = true;
            token.remove_prefix(nl + 1);
        }
        shown = Truncate(shown, max_chars);
        const double w = static_cast<double>(std::max<size_t>(1, Columns(std::string_view(shown)))) * char_w;
        if (x + w > right + 1e-9 && x > left) {
            x = left;
            y += line_h;
        }
        if (!doc.is_scored(i)) {
            // Context, such as the spaces between words: plain text, no box and no score.
            f.body() += std::string("<text class=\"context-text\" xml:space=\"preserve\" x=\"") + Num(x) + "\" y=\"" +
                        Num(y + box_h * 0.72) + "\" font-family=\"" + kMonoFont + "\" fill=\"" + kTextColor + "\">" +
                        Escape(shown) + "</text>\n";
            x += w;
            if (breaks_line) {
                x = left;
                y += line_h;
            }
            continue;
        }
        const float r = doc.relevance[i];
        const bool missing = !std::isfinite(r);
        const RgbColor c = missing ? RgbColor{0.741f, 0.741f, 0.741f} : DivergingColormap(NormalizeSigned(r, max_abs));
        f.body() += std::string("<rect class=\"token") + (missing ? " missing" : "") + "\" x=\"" + Num(x) + "\" y=\"" + Num(y) +
                    "\" width=\"" + Num(w) + "\" height=\"" + Num(box_h) + "\" fill=\"" +
                    (missing ? std::string(kMissingColor) : Hex(c)) + "\"><title>" + Escape(doc.tokens[i]) + ": " +
                    (missing ? std::string("not finite") : ValueText(r)) + "</title></rect>\n";
        f.body() += std::string("<text class=\"token-text\" xml:space=\"preserve\" x=\"") + Num(x) + "\" y=\"" +
                    Num(y + box_h * 0.72) + "\" font-family=\"" + kMonoFont + "\" fill=\"" + TextOn(c) + "\">" +
                    Escape(shown) + "</text>\n";
        x += w;
        if (breaks_line) {
            x = left;
            y += line_h;
        }
    }
    y += box_h + fs * 1.6;
    if (!doc.target.empty()) {
        f.body() += std::string("<text class=\"target\" x=\"") + Num(left) + "\" y=\"" + Num(y) + "\" fill=\"" + kMutedColor +
                    "\">explaining <tspan class=\"target-token\" xml:space=\"preserve\" font-family=\"" + kMonoFont +
                    "\" fill=\"" + kTextColor + "\">" + Escape(doc.target) + "</tspan></text>\n";
        y += line_h;
    }
    if (doc.unassigned && std::isfinite(*doc.unassigned)) {
        f.body() += std::string("<text class=\"unassigned\" x=\"") + Num(left) + "\" y=\"" + Num(y) + "\" fill=\"" + kMutedColor +
                    "\">relevance outside the pieces shown: " + ValueText(*doc.unassigned) + "</text>\n";
        y += line_h;
    }
    // Legend: the scale's ends.
    const double legend_w = std::min(f.width() * 0.4, fs * 16.0);
    f.body() += "<g class=\"legend\">\n<defs><linearGradient id=\"token-scale\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"0\">\n";
    for (int i = 0; i <= 10; ++i) {
        f.body() += "<stop offset=\"" + Num(i / 10.0) + "\" stop-color=\"" + Hex(DivergingColormap(i / 5.0f - 1.0f)) + "\"/>\n";
    }
    f.body() += "</linearGradient></defs>\n<rect x=\"" + Num(left + fs * 3) + "\" y=\"" + Num(y - fs * 0.8) + "\" width=\"" +
                Num(legend_w) + "\" height=\"" + Num(fs * 0.8) + "\" fill=\"url(#token-scale)\" stroke=\"" + kAxisColor +
                "\" stroke-width=\"0.5\"/>\n";
    f.text("tick-label", left + fs * 2.6, y, ValueText(-max_abs), "end", kMutedColor);
    f.text("tick-label", left + fs * 3 + legend_w + fs * 0.4, y, ValueText(max_abs), "start", kMutedColor);
    f.body() += "</g>\n";
    return f.finish(y + fs);
}

// ---- beeswarm -------------------------------------------------------------------------------

std::string RenderBeeswarmSvg(const std::vector<AttributionDocument>& docs, const std::vector<int64_t>& feature_indices,
                              const SvgOptions& options) {
    CheckOptions(options);
    if (docs.empty()) {
        throw std::invalid_argument("RenderBeeswarmSvg: no attributions");
    }
    if (feature_indices.empty()) {
        throw std::invalid_argument("RenderBeeswarmSvg: no features");
    }
    for (size_t d = 0; d < docs.size(); ++d) {
        for (int64_t fi : feature_indices) {
            if (fi < 0 || static_cast<size_t>(fi) >= docs[d].values.size()) {
                throw std::invalid_argument("RenderBeeswarmSvg: feature " + std::to_string(fi) + " is out of range for attribution " +
                                            std::to_string(d));
            }
            if (!std::isfinite(docs[d].values[static_cast<size_t>(fi)])) {
                throw std::invalid_argument("RenderBeeswarmSvg: attribution " + std::to_string(d) + " has a non-finite value for feature " +
                                            std::to_string(fi));
            }
        }
    }
    std::vector<Attribution> attrs;
    attrs.reserve(docs.size());
    for (const AttributionDocument& d : docs) {
        attrs.push_back(ToHostAttribution(d));
    }
    std::vector<std::string> names = FeatureNames(docs.front(), docs.front().values.size());

    double min_v = 0, max_v = 0;
    std::vector<std::vector<BeeswarmPoint>> rows;
    for (int64_t fi : feature_indices) {
        rows.push_back(ToBeeswarmPoints(attrs, fi));
        for (const BeeswarmPoint& p : rows.back()) {
            min_v = std::min(min_v, static_cast<double>(p.x));
            max_v = std::max(max_v, static_cast<double>(p.x));
        }
    }
    const float max_abs = static_cast<float>(std::max(std::abs(min_v), std::abs(max_v)));

    Figure f(options, "Attribution beeswarm across " + std::to_string(docs.size()) + " inputs");
    const double fs = f.fs();
    const size_t max_label_chars = static_cast<size_t>(std::max(4.0, f.width() * 0.3 / (kCharWidthEm * fs)));
    double label_w = 0;
    for (int64_t fi : feature_indices) {
        label_w = std::max(label_w, TextWidth(Truncate(names[static_cast<size_t>(fi)], max_label_chars), fs));
    }
    const double x0 = fs + label_w + fs * 0.75;
    const double x1 = f.width() - fs * 1.5;
    auto [lo, hi] = ZeroAnchoredDomain(min_v, max_v);
    auto X = [&](double v) { return x0 + (v - lo) / (hi - lo) * (x1 - x0); };

    const double radius = std::max(1.5, fs * 0.28);
    const double row_h = std::max(fs * 3.0, radius * 10.0);
    const double y0 = f.top() + fs * 0.5;
    for (size_t r = 0; r < rows.size(); ++r) {
        const double cy = y0 + row_h * (static_cast<double>(r) + 0.5);
        double max_offset = 0;
        for (const BeeswarmPoint& p : rows[r]) max_offset = std::max(max_offset, static_cast<double>(std::abs(p.y)));
        const double half = row_h / 2 - radius - 1.0;
        f.body() += "<g class=\"feature-row\">\n";
        if (r % 2 == 1) {
            f.body() += "<rect class=\"row-band\" x=\"" + Num(x0) + "\" y=\"" + Num(cy - row_h / 2) + "\" width=\"" +
                        Num(x1 - x0) + "\" height=\"" + Num(row_h) + "\" fill=\"#f6f6f6\"/>\n";
        }
        f.text("feature-label", x0 - fs * 0.75, cy + fs * 0.35, Truncate(names[static_cast<size_t>(feature_indices[r])], max_label_chars),
               "end");
        for (const BeeswarmPoint& p : rows[r]) {
            const double py = cy + (max_offset > 0 ? static_cast<double>(p.y) / max_offset * half : 0.0);
            f.body() += "<circle class=\"point\" cx=\"" + Num(X(p.x)) + "\" cy=\"" + Num(py) + "\" r=\"" + Num(radius) +
                        "\" fill=\"" + Hex(DivergingColormap(NormalizeSigned(p.x, max_abs))) + "\" stroke=\"#555555\" stroke-width=\"0.3\"/>\n";
        }
        f.body() += "</g>\n";
    }
    const double plot_bottom = y0 + row_h * static_cast<double>(rows.size());
    f.line("zero-line", X(0), y0, X(0), plot_bottom, kMutedColor, 1.0, "3 3");
    f.x_axis(plot_bottom + fs * 0.25, x0, x1, lo, hi, "attribution (" + docs.front().method + ")");
    return f.finish(plot_bottom + fs * 4.0);
}

}  // namespace pulsatrix
