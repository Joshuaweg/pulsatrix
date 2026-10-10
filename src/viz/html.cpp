#include "pulsatrix/viz/html.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "pulsatrix/ice.hpp"
#include "pulsatrix/viz/colormap.hpp"
#include "pulsatrix/viz/plot_data.hpp"
#include "html_detail.hpp"
#include "svg_detail.hpp"

namespace pulsatrix {

namespace html_detail {

/** @brief JSON for a <script> element: "<" escaped so a label holding "</script>" can't end it. */
std::string ScriptSafe(const std::string& json) {
    std::string out;
    out.reserve(json.size());
    for (char c : json) {
        if (c == '<') {
            out += "\\u003c";
        } else {
            out += c;
        }
    }
    return out;
}

std::string ReadScript(const std::string& dir, const char* name) {
    const std::string path = dir + "/" + name;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(std::string("HTML views: can't read ") + path +
                                 " for inline scripts (tools/render/fetch_vega.sh downloads it)");
    }
    std::stringstream ss;
    ss << in.rdbuf();
    std::string js = ss.str();
    // "</script" inside the library would end the element early.
    for (size_t at = js.find("</script"); at != std::string::npos; at = js.find("</script", at + 2)) js.replace(at, 2, "<\\/");
    return js;
}

constexpr const char* kStyle =
    "body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;margin:24px;color:#222;background:#fff;"
    "max-width:1100px}"
    "h1{font-size:18px;font-weight:600;margin:0 0 12px}"
    ".muted{color:#666}.mono{font-family:ui-monospace,Menlo,Consolas,monospace}"
    ".tokens{line-height:2.1;white-space:pre-wrap;font-size:16px;margin:12px 0}"
    ".tokens span.s{border-radius:3px;padding:2px 1px}"
    ".legend{display:flex;align-items:center;gap:8px;font-size:13px;color:#666;margin-top:12px}"
    ".legend .bar{width:220px;height:10px;border:1px solid #999}"
    "table{border-collapse:collapse;font-size:14px}td,th{padding:4px 10px;text-align:left}"
    "th{color:#666;font-weight:500;border-bottom:1px solid #ddd}"
    "figure{margin:12px 0;overflow-x:auto}figure svg{display:block}"
    ".tokens.residues{word-break:break-all;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;line-height:1.9}";

std::string Page(const std::string& heading, const std::string& head_extra, const std::string& body) {
    return "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
           "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>" +
           svg_detail::Escape(heading) + "</title>\n<style>" + kStyle + "</style>\n" + head_extra + "</head>\n<body>\n" + body +
           "</body>\n</html>\n";
}

}  // namespace html_detail

namespace {

using Obj = JsonValue::Object;
using Arr = JsonValue::Array;

constexpr const char* kSchema = "https://vega.github.io/schema/vega-lite/v6.json";
constexpr const char* kIncrease = "#d6604d";  // the SVG renderer's red and blue ends
constexpr const char* kDecrease = "#4393c3";
constexpr const char* kNeutral = "#bdbdbd";

// SRI hashes of the pinned builds (openssl dgst -sha384 -binary FILE | openssl base64 -A).
constexpr const char* kVegaSri = "sha384-VKdcJr3ZaBIJMbVcopTAI/JEuUkSY6qnwVu9iLuw0DnQ9gQ1JjsfZJhXFQAgNi43";
constexpr const char* kVegaLiteSri = "sha384-9/70gNCfOu6G7xXvkdreMfuqAEsoaGJVXV2BN/JLRXkSmcGvnMqtsRx8HZtUWAvI";
constexpr const char* kVegaEmbedSri = "sha384-Muy1QRxYFeNrA1zShc1KtN4OfipkR/61gt+HWqN57c1Zxu3Oe16TsilsruVVjfKL";

JsonValue Num(double v) { return JsonValue(v); }
JsonValue NumF(float v) { return std::isfinite(v) ? JsonValue::Float(v) : JsonValue(); }

std::string Hex(float signed_normalized) { return svg_detail::Hex(DivergingColormap(signed_normalized)); }

/** @brief The diverging color scale the SVG renderer uses: blue below zero, red above, symmetric. */
JsonValue DivergingScale(double max_abs) {
    const double m = max_abs > 0 ? max_abs : 1.0;
    return Obj{{"domain", Arr{Num(-m), Num(0), Num(m)}}, {"range", Arr{Hex(-1.0f), Hex(0.0f), Hex(1.0f)}}};
}

JsonValue Title(const std::string& text, const std::string& subtitle = "") {
    Obj t{{"text", text}, {"anchor", "start"}, {"fontSize", 14}};
    if (!subtitle.empty()) t.emplace_back("subtitle", subtitle);
    return t;
}

/** @brief A spec's common members: schema, title, width, data values and the shared config. */
Obj Base(const std::string& title, const HtmlOptions& options, Arr values, const std::string& subtitle = "") {
    if (options.width < 120) throw std::invalid_argument("HTML views: width must be at least 120 pixels");
    return Obj{{"$schema", kSchema},
               {"title", Title(options.title.empty() ? title : options.title, subtitle)},
               {"width", options.width},
               {"data", Obj{{"values", std::move(values)}}},
               {"config", Obj{{"view", Obj{{"stroke", nullptr}}}, {"axis", Obj{{"labelLimit", 260}}}}}};
}

void Set(Obj& spec, const std::string& key, JsonValue value) {
    for (auto& kv : spec) {
        if (kv.first == key) {
            kv.second = std::move(value);
            return;
        }
    }
    spec.emplace_back(key, std::move(value));
}

JsonValue Field(const std::string& field, const std::string& type, Obj extra = {}) {
    Obj f{{"field", field}, {"type", type}};
    for (auto& kv : extra) f.push_back(std::move(kv));
    return f;
}

JsonValue Tooltip(std::initializer_list<std::pair<const char*, const char*>> fields) {
    Arr t;
    for (const auto& [field, type] : fields) {
        Obj f{{"field", field}, {"type", type}};
        if (std::string(type) == "quantitative") f.emplace_back("format", ".4~g");
        t.push_back(f);
    }
    return t;
}

/** @brief Zoom with the mouse wheel and pan by dragging, on both axes of a continuous chart, or
 *         on x only when y is categorical. */
JsonValue ZoomParams(bool x_only = false) {
    Obj select{{"type", "interval"}};
    if (x_only) select.emplace_back("encodings", Arr{"x"});
    return Arr{Obj{{"name", "zoom"}, {"select", select}, {"bind", "scales"}}};
}

void CheckTopK(int top_k, const char* what) {
    if (top_k < 1) throw std::invalid_argument(std::string(what) + ": top_k must be at least 1");
}

std::string Escape(std::string_view s) { return svg_detail::Escape(s); }
using html_detail::Page;
using html_detail::ReadScript;
using html_detail::ScriptSafe;

std::string Scripts(const HtmlOptions& options) {
    if (options.scripts == HtmlScripts::Inline) {
        if (options.script_dir.empty()) throw std::invalid_argument("HTML views: inline scripts need script_dir");
        return "<script>" + ReadScript(options.script_dir, "vega.min.js") + "</script>\n<script>" +
               ReadScript(options.script_dir, "vega-lite.min.js") + "</script>\n<script>" +
               ReadScript(options.script_dir, "vega-embed.min.js") + "</script>\n";
    }
    auto tag = [](const std::string& package, const char* version, const char* file, const char* sri) {
        return "<script src=\"https://cdn.jsdelivr.net/npm/" + package + "@" + version + "/build/" + file +
               "\" integrity=\"" + sri + "\" crossorigin=\"anonymous\"></script>\n";
    };
    return tag("vega", kVegaVersion, "vega.min.js", kVegaSri) + tag("vega-lite", kVegaLiteVersion, "vega-lite.min.js", kVegaLiteSri) +
           tag("vega-embed", kVegaEmbedVersion, "vega-embed.min.js", kVegaEmbedSri);
}

}  // namespace

std::string html_detail::VegaScripts(const HtmlOptions& options) { return Scripts(options); }

namespace {

/** @brief A legend for a diverging relevance scale, as HTML. */
std::string DivergingLegend(double max_abs) {
    std::string stops;
    for (int i = 0; i <= 10; ++i) stops += (i ? "," : "") + Hex(static_cast<float>(i) / 5.0f - 1.0f);
    return "<div class=\"legend\"><span>" + svg_detail::ValueText(-max_abs) + "</span><div class=\"bar\" style=\"background:linear-gradient(to right," +
           stops + ")\"></div><span>" + svg_detail::ValueText(max_abs) + "</span></div>\n";
}

/** @brief One span per piece, colored on a diverging scale (or an unsigned one for activations). */
std::string TokenSpans(const std::vector<std::string>& tokens, const std::vector<float>& values, const std::vector<bool>& scored,
                       double scale, bool is_signed) {
    std::string out;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const bool s = scored.empty() || scored[i];
        if (!s) {
            out += "<span>" + Escape(tokens[i]) + "</span>";
            continue;
        }
        const float v = values[i];
        RgbColor c{0.741f, 0.741f, 0.741f};
        if (std::isfinite(v)) {
            c = is_signed ? DivergingColormap(NormalizeSigned(v, static_cast<float>(scale)))
                          : DivergingColormap(std::clamp(static_cast<float>(v / (scale > 0 ? scale : 1.0)), 0.0f, 1.0f));
        }
        const std::string label = std::isfinite(v) ? svg_detail::ValueText(v) : "not finite";
        out += "<span class=\"s\" style=\"background:" + svg_detail::Hex(c) + ";color:" + svg_detail::TextOn(c) + "\" title=\"" +
               Escape(tokens[i]) + ": " + label + "\">" + Escape(tokens[i]) + "</span>";
    }
    return out;
}

}  // namespace

// ---- Specs ---------------------------------------------------------------------------------

JsonValue ToVegaLiteBarChart(const AttributionDocument& doc, int top_k, const HtmlOptions& options) {
    CheckTopK(top_k, "ToVegaLiteBarChart");
    svg_detail::CheckFinite(doc.values, "ToVegaLiteBarChart");
    const BarSeries bars = ToFeatureImportanceBars(svg_detail::ToHostAttribution(doc), top_k);
    Arr values;
    double max_abs = 0;
    for (size_t i = 0; i < bars.values.size(); ++i) {
        values.push_back(Obj{{"feature", bars.labels[i]}, {"value", NumF(bars.values[i])}, {"rank", static_cast<int64_t>(i)}});
        max_abs = std::max(max_abs, std::abs(static_cast<double>(bars.values[i])));
    }
    Obj spec = Base("Feature attribution (" + doc.method + ")", options, std::move(values));
    spec.emplace_back("height", Obj{{"step", 22}});
    spec.emplace_back("mark", Obj{{"type", "bar"}, {"tooltip", true}});
    spec.emplace_back("encoding", Obj{{"y", Field("feature", "nominal", Obj{{"sort", Obj{{"field", "rank"}}}, {"title", nullptr}})},
                                      {"x", Field("value", "quantitative", Obj{{"title", "attribution"}})},
                                      {"color", Field("value", "quantitative", Obj{{"scale", DivergingScale(max_abs)}, {"legend", nullptr}})},
                                      {"tooltip", Tooltip({{"feature", "nominal"}, {"value", "quantitative"}})}});
    return spec;
}

JsonValue ToVegaLiteWaterfall(const AttributionDocument& doc, float baseline_value, const HtmlOptions& options) {
    svg_detail::CheckFinite(doc.values, "ToVegaLiteWaterfall");
    if (!std::isfinite(baseline_value)) throw std::invalid_argument("ToVegaLiteWaterfall: the baseline must be finite");
    const std::vector<WaterfallStep> steps = ToWaterfallSteps(svg_detail::ToHostAttribution(doc), baseline_value);
    const std::vector<WaterfallBar> bars = ToWaterfallBars(steps, baseline_value);
    Arr values;
    values.push_back(Obj{{"step", "baseline"}, {"order", static_cast<int64_t>(0)}, {"bottom", Num(0)}, {"top", NumF(baseline_value)},
                         {"delta", NumF(baseline_value)}, {"total", NumF(baseline_value)}, {"kind", "baseline"}});
    for (size_t i = 0; i < steps.size(); ++i) {
        values.push_back(Obj{{"step", steps[i].label},
                             {"order", static_cast<int64_t>(i + 1)},
                             {"bottom", NumF(bars[i].bottom)},
                             {"top", NumF(bars[i].top)},
                             {"delta", NumF(steps[i].delta)},
                             {"total", NumF(steps[i].cumulative)},
                             {"kind", bars[i].increase ? "increase" : "decrease"}});
    }
    const float final_value = steps.empty() ? baseline_value : steps.back().cumulative;
    values.push_back(Obj{{"step", "prediction"}, {"order", static_cast<int64_t>(steps.size() + 1)}, {"bottom", Num(0)},
                         {"top", NumF(final_value)}, {"delta", NumF(final_value)}, {"total", NumF(final_value)}, {"kind", "baseline"}});
    Obj spec = Base("Waterfall (" + doc.method + ")", options, std::move(values),
                    "from " + svg_detail::ValueText(baseline_value) + " to " + svg_detail::ValueText(final_value));
    spec.emplace_back("height", 320);
    spec.emplace_back("mark", Obj{{"type", "bar"}, {"tooltip", true}});
    spec.emplace_back("encoding",
                      Obj{{"x", Field("step", "nominal", Obj{{"sort", Obj{{"field", "order"}}}, {"title", nullptr}, {"axis", Obj{{"labelAngle", -40}}}})},
                          {"y", Field("bottom", "quantitative", Obj{{"title", "output"}})},
                          {"y2", Obj{{"field", "top"}}},
                          {"color", Field("kind", "nominal",
                                          Obj{{"scale", Obj{{"domain", Arr{"increase", "decrease", "baseline"}},
                                                            {"range", Arr{kIncrease, kDecrease, kNeutral}}}},
                                              {"legend", Obj{{"title", nullptr}, {"orient", "top"}}}})},
                          {"tooltip", Tooltip({{"step", "nominal"}, {"delta", "quantitative"}, {"total", "quantitative"}})}});
    return spec;
}

JsonValue ToVegaLiteHeatmap(const HeatmapDocument& doc, const HtmlOptions& options) {
    // RenderHeatmapSvg's checks; non-finite cells are allowed and drawn gray.
    if (doc.rows <= 0 || doc.cols <= 0) throw std::invalid_argument("ToVegaLiteHeatmap: the grid is empty");
    if (doc.cols > std::numeric_limits<int64_t>::max() / doc.rows || static_cast<size_t>(doc.rows * doc.cols) != doc.values.size()) {
        throw std::invalid_argument("ToVegaLiteHeatmap: values.size() is not rows * cols");
    }
    if ((!doc.row_labels.empty() && static_cast<int64_t>(doc.row_labels.size()) != doc.rows) ||
        (!doc.col_labels.empty() && static_cast<int64_t>(doc.col_labels.size()) != doc.cols)) {
        throw std::invalid_argument("ToVegaLiteHeatmap: a label list doesn't match the grid");
    }
    HeatmapGrid finite{{}, 1, 0};
    for (float v : doc.values) {
        if (std::isfinite(v)) finite.values.push_back(v);
    }
    finite.cols = static_cast<int64_t>(finite.values.size());
    const HeatmapColorScale scale = ComputeHeatmapColorScale(finite);
    Arr values;
    values.reserve(doc.values.size());
    for (int64_t r = 0; r < doc.rows; ++r) {
        for (int64_t c = 0; c < doc.cols; ++c) {
            const float v = doc.values[static_cast<size_t>(r * doc.cols + c)];
            Obj cell{{"r", r}, {"c", c}, {"v", NumF(v)}};
            if (!doc.row_labels.empty()) cell.emplace_back("row", doc.row_labels[static_cast<size_t>(r)]);
            if (!doc.col_labels.empty()) cell.emplace_back("col", doc.col_labels[static_cast<size_t>(c)]);
            values.push_back(std::move(cell));
        }
    }
    const bool row_labels = !doc.row_labels.empty(), col_labels = !doc.col_labels.empty();
    const int64_t cell = std::clamp<int64_t>(options.width / std::max<int64_t>(doc.cols, 1), 1, 40);
    Obj spec = Base(doc.title.empty() ? "Heatmap" : doc.title, options, std::move(values));
    Set(spec, "width", static_cast<int64_t>(cell * doc.cols));  // square cells
    spec.emplace_back("height", static_cast<int64_t>(cell * doc.rows));
    spec.emplace_back("mark", Obj{{"type", "rect"}, {"tooltip", true}});
    JsonValue color = scale.is_signed
                          ? Field("v", "quantitative", Obj{{"scale", DivergingScale(scale.scale_max)}, {"title", "value"}})
                          : Field("v", "quantitative", Obj{{"scale", Obj{{"scheme", "viridis"}, {"domain", Arr{Num(0), Num(scale.scale_max)}}}},
                                                           {"title", "value"}});
    Obj color_obj = color.as_object();
    color_obj.emplace_back("condition", Obj{{"test", "datum.v === null"}, {"value", kNeutral}});
    const bool dense = doc.cols > 48 || doc.rows > 48;
    spec.emplace_back("encoding",
                      Obj{{"x", Field(col_labels ? "col" : "c", "ordinal",
                                      Obj{{"title", nullptr}, {"sort", Obj{{"field", "c"}}}, {"axis", dense ? JsonValue() : Obj{{"labelAngle", -45}}}})},
                          {"y", Field(row_labels ? "row" : "r", "ordinal", Obj{{"title", nullptr}, {"sort", Obj{{"field", "r"}}},
                                                                             {"axis", dense ? JsonValue() : Obj{}}})},
                          {"color", color_obj},
                          {"tooltip", Tooltip({{row_labels ? "row" : "r", "nominal"}, {col_labels ? "col" : "c", "nominal"}, {"v", "quantitative"}})}});
    return spec;
}

JsonValue ToVegaLiteBeeswarm(const std::vector<AttributionDocument>& docs, const std::vector<int64_t>& feature_indices,
                             const HtmlOptions& options) {
    if (docs.empty() || feature_indices.empty()) throw std::invalid_argument("ToVegaLiteBeeswarm: needs documents and features");
    std::vector<Attribution> runs;
    for (const auto& d : docs) {
        svg_detail::CheckFinite(d.values, "ToVegaLiteBeeswarm");
        runs.push_back(svg_detail::ToHostAttribution(d));
    }
    const std::vector<std::string> names = svg_detail::FeatureNames(docs[0], docs[0].values.size());
    Arr values;
    double max_abs = 0;
    for (size_t row = 0; row < feature_indices.size(); ++row) {
        const int64_t f = feature_indices[row];
        for (const auto& d : docs) {
            if (f < 0 || static_cast<size_t>(f) >= d.values.size()) throw std::invalid_argument("ToVegaLiteBeeswarm: a feature index is out of range");
        }
        const std::vector<BeeswarmPoint> points = ToBeeswarmPoints(runs, f);
        for (size_t i = 0; i < points.size(); ++i) {
            values.push_back(Obj{{"feature", names[static_cast<size_t>(f)]}, {"row", static_cast<int64_t>(row)}, {"value", NumF(points[i].x)},
                                 {"offset", NumF(points[i].y)}, {"input", static_cast<int64_t>(i)}});
            max_abs = std::max(max_abs, std::abs(static_cast<double>(points[i].x)));
        }
    }
    Obj spec = Base("Attribution across inputs (" + docs[0].method + ")", options, std::move(values),
                    std::to_string(docs.size()) + " inputs");
    spec.emplace_back("height", Obj{{"step", 46}});
    spec.emplace_back("mark", Obj{{"type", "circle"}, {"size", 36}, {"opacity", 0.85}, {"tooltip", true}});
    spec.emplace_back("encoding",
                      Obj{{"y", Field("feature", "nominal", Obj{{"sort", Obj{{"field", "row"}}}, {"title", nullptr}})},
                          {"yOffset", Field("offset", "quantitative", Obj{{"scale", Obj{{"domain", Arr{Num(-1), Num(1)}}}}})},
                          {"x", Field("value", "quantitative", Obj{{"title", "attribution"}})},
                          {"color", Field("value", "quantitative", Obj{{"scale", DivergingScale(max_abs)}, {"legend", nullptr}})},
                          {"tooltip", Tooltip({{"feature", "nominal"}, {"input", "ordinal"}, {"value", "quantitative"}})}});
    spec.emplace_back("params", ZoomParams(/*x_only=*/true));
    return spec;
}

JsonValue ToVegaLitePartialDependence(const PartialDependenceDocument& doc, const PartialDependenceSvgOptions& pd,
                                      const HtmlOptions& options) {
    (void)ToJson(doc);
    svg_detail::CheckFinite(doc.grid, "ToVegaLitePartialDependence");
    svg_detail::CheckFinite(doc.partial_dependence, "ToVegaLitePartialDependence");
    svg_detail::CheckFinite(doc.ice, "ToVegaLitePartialDependence");
    if (pd.max_curves < 0) throw std::invalid_argument("ToVegaLitePartialDependence: max_curves must not be negative");
    if (pd.style != IceStyle::Raw && doc.num_instances == 0) {
        throw std::invalid_argument("ToVegaLitePartialDependence: centered and derivative views need ICE curves");
    }
    const size_t g = doc.grid.size();
    std::vector<float> curves = doc.ice;
    std::vector<float> grid = doc.grid;
    if (doc.num_instances > 0) {
        const IceResult ice = ToIceResult(doc);
        if (pd.style == IceStyle::Centered) curves = ice.centered();
        if (pd.style == IceStyle::Derivative) {
            curves = ice.derivative();
            grid.assign(doc.grid.begin(), doc.grid.end());  // derivative curves keep the grid's length
        }
    }
    const size_t n = static_cast<size_t>(doc.num_instances);
    const size_t points = n > 0 ? curves.size() / n : g;
    std::vector<float> average(points, 0.0f);
    if (pd.style == IceStyle::Raw || n == 0) {
        average = doc.partial_dependence;
    } else {
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < points; ++k) average[k] += curves[i * points + k] / static_cast<float>(n);
    }
    Arr ice_values, average_values, rug_values;
    const size_t shown = std::min<size_t>(n, static_cast<size_t>(pd.max_curves));
    for (size_t s = 0; s < shown; ++s) {
        const size_t i = shown == n ? s : s * n / shown;  // evenly spaced through the instances
        for (size_t k = 0; k < points; ++k) {
            ice_values.push_back(Obj{{"x", NumF(grid[k])}, {"y", NumF(curves[i * points + k])}, {"instance", static_cast<int64_t>(i)}});
        }
    }
    for (size_t k = 0; k < points; ++k) average_values.push_back(Obj{{"x", NumF(grid[k])}, {"y", NumF(average[k])}});
    for (float v : doc.feature_values) {
        if (std::isfinite(v)) rug_values.push_back(Obj{{"x", NumF(v)}});
    }
    const std::string feature = doc.feature.empty() ? "feature" : doc.feature;
    const char* style = pd.style == IceStyle::Centered ? "centered ICE" : pd.style == IceStyle::Derivative ? "derivative ICE" : "";
    const std::string title = std::string(doc.method == "ale" ? "Accumulated local effects" : "Partial dependence") + ": " + feature;
    std::string subtitle = doc.target.empty() ? "" : "target " + doc.target;
    if (*style) subtitle += std::string(subtitle.empty() ? "" : ", ") + style;
    Obj spec = Base(title, options, Arr{}, subtitle);
    spec.erase(std::remove_if(spec.begin(), spec.end(), [](const auto& kv) { return kv.first == "data"; }), spec.end());
    spec.emplace_back("height", 340);
    const JsonValue x = Field("x", "quantitative", Obj{{"title", feature}});
    const std::string ylabel = pd.style == IceStyle::Derivative ? "slope of the prediction" : "prediction";
    Arr layers;
    if (!ice_values.empty()) {
        layers.push_back(Obj{{"data", Obj{{"values", ice_values}}},
                             {"mark", Obj{{"type", "line"}, {"strokeWidth", 1}, {"opacity", 0.55}, {"color", "#6b8fb3"}}},
                             {"encoding", Obj{{"x", x}, {"y", Field("y", "quantitative", Obj{{"title", ylabel}})}, {"detail", Field("instance", "nominal")}}}});
    }
    layers.push_back(Obj{{"data", Obj{{"values", average_values}}},
                         {"mark", Obj{{"type", "line"}, {"strokeWidth", 3}, {"color", "#b2182b"}, {"point", true}, {"tooltip", true}}},
                         {"encoding", Obj{{"x", x}, {"y", Field("y", "quantitative", Obj{{"title", ylabel}})},
                                          {"tooltip", Tooltip({{"x", "quantitative"}, {"y", "quantitative"}})}}},
                         {"params", ZoomParams()}});
    if (!rug_values.empty()) {
        layers.push_back(Obj{{"data", Obj{{"values", rug_values}}},
                             {"mark", Obj{{"type", "tick"}, {"color", "#333"}, {"opacity", 0.8}, {"size", 10}, {"thickness", 1.5}}},
                             {"encoding", Obj{{"x", x}, {"y", Obj{{"value", 340}}}}}});
    }
    if (pd.style != IceStyle::Raw) {
        layers.push_back(Obj{{"mark", Obj{{"type", "rule"}, {"strokeDash", Arr{4, 4}}, {"color", "#888"}}},
                             {"encoding", Obj{{"y", Obj{{"datum", 0}}}}}});
    }
    spec.emplace_back("layer", std::move(layers));
    return spec;
}

JsonValue ToVegaLiteTornado(const SensitivityDocument& doc, int top_k, const HtmlOptions& options) {
    CheckTopK(top_k, "ToVegaLiteTornado");
    if (doc.features.empty()) throw std::invalid_argument("ToVegaLiteTornado: the document has no features");
    std::vector<const SensitivityDocument::Feature*> order;
    for (const auto& f : doc.features) {
        for (float v : {f.value, f.low, f.high, f.output_low, f.output_high}) {
            if (!std::isfinite(v)) throw std::invalid_argument("ToVegaLiteTornado: a value is not finite");
        }
        order.push_back(&f);
    }
    if (!std::isfinite(doc.output)) throw std::invalid_argument("ToVegaLiteTornado: the output is not finite");
    std::stable_sort(order.begin(), order.end(), [](const auto* a, const auto* b) {
        return std::abs(a->output_high - a->output_low) > std::abs(b->output_high - b->output_low);
    });
    order.resize(std::min<size_t>(order.size(), static_cast<size_t>(top_k)));
    Arr values;
    for (size_t r = 0; r < order.size(); ++r) {
        const auto& f = *order[r];
        for (const auto& [end, input, side] : {std::tuple{f.output_low, f.low, "low"}, std::tuple{f.output_high, f.high, "high"}}) {
            values.push_back(Obj{{"feature", f.name}, {"rank", static_cast<int64_t>(r)}, {"from", NumF(doc.output)}, {"to", NumF(end)},
                                 {"side", side}, {"input", NumF(input)}, {"value", NumF(f.value)}});
        }
    }
    Obj spec = Base("Local sensitivity", options, std::move(values),
                    (doc.target.empty() ? std::string() : "target " + doc.target + ", ") + "output " + svg_detail::ValueText(doc.output));
    spec.emplace_back("height", Obj{{"step", 26}});
    spec.emplace_back("layer",
                      Arr{Obj{{"mark", Obj{{"type", "bar"}, {"tooltip", true}, {"opacity", 0.9}}},
                              {"encoding", Obj{{"y", Field("feature", "nominal", Obj{{"sort", Obj{{"field", "rank"}}}, {"title", nullptr}})},
                                               {"x", Field("from", "quantitative", Obj{{"title", "output"}, {"scale", Obj{{"zero", false}}}})},
                                               {"x2", Obj{{"field", "to"}}},
                                               {"color", Field("side", "nominal", Obj{{"scale", Obj{{"domain", Arr{"low", "high"}}, {"range", Arr{kDecrease, kIncrease}}}},
                                                                                     {"legend", Obj{{"title", "feature at its"}, {"orient", "top"}}}})},
                                               {"tooltip", Tooltip({{"feature", "nominal"}, {"side", "nominal"}, {"input", "quantitative"},
                                                                    {"to", "quantitative"}, {"value", "quantitative"}})}}}},
                          Obj{{"mark", Obj{{"type", "rule"}, {"strokeDash", Arr{4, 4}}, {"color", "#555"}}},
                              {"encoding", Obj{{"x", Obj{{"datum", NumF(doc.output)}}}}}}});
    return spec;
}

JsonValue ToVegaLiteCounterfactual(const CounterfactualDocument& doc, int max_rows, const HtmlOptions& options) {
    if (max_rows < 1) throw std::invalid_argument("ToVegaLiteCounterfactual: max_rows must be at least 1");
    struct Row {
        const CounterfactualDocument::Feature* f;
        float units;
    };
    std::vector<Row> rows;
    for (const auto& f : doc.features) {
        for (float v : {f.original, f.counterfactual, f.scale}) {
            if (!std::isfinite(v)) throw std::invalid_argument("ToVegaLiteCounterfactual: a value is not finite");
        }
        const float units = (f.counterfactual - f.original) / (f.scale > 0 ? f.scale : 1.0f);
        if (std::abs(units) > 1e-3f) rows.push_back({&f, units});
    }
    std::stable_sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return std::abs(a.units) > std::abs(b.units); });
    const size_t hidden = rows.size() > static_cast<size_t>(max_rows) ? rows.size() - static_cast<size_t>(max_rows) : 0;
    rows.resize(rows.size() - hidden);
    Arr values;
    double max_abs = 0;
    for (size_t r = 0; r < rows.size(); ++r) {
        const auto& f = *rows[r].f;
        values.push_back(Obj{{"feature", f.name + ": " + svg_detail::ValueText(f.original) + " \xE2\x86\x92 " + svg_detail::ValueText(f.counterfactual)},
                             {"name", f.name}, {"rank", static_cast<int64_t>(r)}, {"original", NumF(f.original)},
                             {"counterfactual", NumF(f.counterfactual)}, {"change", NumF(rows[r].units)}});
        max_abs = std::max(max_abs, std::abs(static_cast<double>(rows[r].units)));
    }
    std::string verdict = std::string(doc.valid ? "reaches the target" : "does not reach the target") +
                          (doc.target.empty() ? "" : " (" + doc.target + ")") + "; output " + svg_detail::ValueText(doc.output_before) +
                          " \xE2\x86\x92 " + svg_detail::ValueText(doc.output_after);
    if (hidden > 0) verdict += "; " + std::to_string(hidden) + " smaller changes not shown";
    Obj spec = Base("Counterfactual", options, std::move(values), verdict);
    spec.emplace_back("height", Obj{{"step", 26}});
    spec.emplace_back("mark", Obj{{"type", "bar"}, {"tooltip", true}});
    spec.emplace_back("encoding",
                      Obj{{"y", Field("feature", "nominal", Obj{{"sort", Obj{{"field", "rank"}}}, {"title", nullptr}})},
                          {"x", Field("change", "quantitative", Obj{{"title", "change, in units of the feature's scale"}})},
                          {"color", Field("change", "quantitative", Obj{{"scale", DivergingScale(max_abs)}, {"legend", nullptr}})},
                          {"tooltip", Tooltip({{"name", "nominal"}, {"original", "quantitative"}, {"counterfactual", "quantitative"},
                                               {"change", "quantitative"}})}});
    return spec;
}

JsonValue ToVegaLiteCounterfactualSet(const std::vector<CounterfactualDocument>& docs, const HtmlOptions& options) {
    if (docs.empty()) throw std::invalid_argument("ToVegaLiteCounterfactualSet: no documents");
    const auto& first = docs[0].features;
    for (const auto& d : docs) {
        if (d.features.size() != first.size()) throw std::invalid_argument("ToVegaLiteCounterfactualSet: the documents describe different inputs");
        for (size_t i = 0; i < first.size(); ++i) {
            if (d.features[i].name != first[i].name || d.features[i].original != first[i].original) {
                throw std::invalid_argument("ToVegaLiteCounterfactualSet: the documents describe different inputs");
            }
            for (float v : {d.features[i].original, d.features[i].counterfactual, d.features[i].scale}) {
                if (!std::isfinite(v)) throw std::invalid_argument("ToVegaLiteCounterfactualSet: a value is not finite");
            }
        }
    }
    Arr values;
    double max_abs = 0;
    int64_t row = 0;
    for (size_t i = 0; i < first.size(); ++i) {
        bool changed = false;
        for (const auto& d : docs) {
            const auto& f = d.features[i];
            changed = changed || std::abs((f.counterfactual - f.original) / (f.scale > 0 ? f.scale : 1.0f)) > 1e-3f;
        }
        if (!changed) continue;
        values.push_back(Obj{{"feature", first[i].name}, {"row", row}, {"column", "input"}, {"col", static_cast<int64_t>(0)},
                             {"value", NumF(first[i].original)}, {"change", NumF(0.0f)}, {"label", svg_detail::ValueText(first[i].original)}});
        for (size_t k = 0; k < docs.size(); ++k) {
            const auto& f = docs[k].features[i];
            const float units = (f.counterfactual - f.original) / (f.scale > 0 ? f.scale : 1.0f);
            const bool moved = std::abs(units) > 1e-3f;
            max_abs = std::max(max_abs, std::abs(static_cast<double>(units)));
            values.push_back(Obj{{"feature", first[i].name},
                                 {"row", row},
                                 {"column", "#" + std::to_string(k + 1) + (docs[k].valid ? " \xE2\x9C\x93" : " \xE2\x9C\x97")},
                                 {"col", static_cast<int64_t>(k + 1)},
                                 {"value", NumF(f.counterfactual)},
                                 {"change", moved ? NumF(units) : NumF(0.0f)},
                                 {"label", moved ? svg_detail::ValueText(f.counterfactual) : std::string()}});
        }
        ++row;
    }
    Obj spec = Base("Counterfactuals", options, std::move(values), std::to_string(docs.size()) + " counterfactuals of one input");
    spec.emplace_back("height", Obj{{"step", 28}});
    const JsonValue x = Field("column", "ordinal", Obj{{"sort", Obj{{"field", "col"}}}, {"title", nullptr}, {"axis", Obj{{"orient", "top"}, {"labelAngle", 0}}}});
    const JsonValue y = Field("feature", "nominal", Obj{{"sort", Obj{{"field", "row"}}}, {"title", nullptr}});
    spec.emplace_back("layer", Arr{Obj{{"mark", Obj{{"type", "rect"}, {"tooltip", true}}},
                                       {"encoding", Obj{{"x", x}, {"y", y},
                                                        {"color", Field("change", "quantitative", Obj{{"scale", DivergingScale(max_abs)}, {"legend", Obj{{"title", "change (scale units)"}}}})},
                                                        {"tooltip", Tooltip({{"feature", "nominal"}, {"column", "nominal"}, {"value", "quantitative"},
                                                                             {"change", "quantitative"}})}}}},
                                   Obj{{"mark", Obj{{"type", "text"}, {"fontSize", 12}}},
                                       {"encoding", Obj{{"x", x}, {"y", y}, {"text", Field("label", "nominal")}}}}});
    return spec;
}

JsonValue ToVegaLiteMorris(const MorrisDocument& doc, const HtmlOptions& options) {
    if (doc.features.empty()) throw std::invalid_argument("ToVegaLiteMorris: the document has no features");
    Arr values;
    double top = 0;
    for (const auto& f : doc.features) {
        for (float v : {f.mu_star, f.sigma, f.mu_star_conf}) {
            if (!std::isfinite(v) || v < 0) throw std::invalid_argument("ToVegaLiteMorris: mu*, sigma and the confidence must be finite and non-negative");
        }
        if (!std::isfinite(f.mu)) throw std::invalid_argument("ToVegaLiteMorris: mu is not finite");
        values.push_back(Obj{{"feature", f.name}, {"mu_star", NumF(f.mu_star)}, {"sigma", NumF(f.sigma)}, {"mu", NumF(f.mu)},
                             {"lo", NumF(std::max(0.0f, f.mu_star - f.mu_star_conf))}, {"hi", NumF(f.mu_star + f.mu_star_conf)}});
        top = std::max({top, static_cast<double>(f.mu_star + f.mu_star_conf), static_cast<double>(f.sigma)});
    }
    Obj spec = Base("Morris screening", options, std::move(values),
                    (doc.target.empty() ? std::string() : "target " + doc.target + ", ") + std::to_string(doc.num_trajectories) +
                        " trajectories; above the dashed line, effects vary more than their average");
    spec.emplace_back("height", 360);
    const JsonValue x = Field("mu_star", "quantitative", Obj{{"title", "\xCE\xBC* (importance)"}});
    const JsonValue y = Field("sigma", "quantitative", Obj{{"title", "\xCF\x83 (how much the effect varies)"}});
    spec.emplace_back("layer",
                      Arr{Obj{{"data", Obj{{"values", Arr{Obj{{"a", Num(0)}}, Obj{{"a", Num(top * 1.05)}}}}}},
                              {"mark", Obj{{"type", "line"}, {"strokeDash", Arr{4, 4}}, {"color", "#999"}}},
                              {"encoding", Obj{{"x", Field("a", "quantitative")}, {"y", Field("a", "quantitative")}}}},
                          Obj{{"mark", Obj{{"type", "rule"}, {"color", "#888"}}},
                              {"encoding", Obj{{"x", Field("lo", "quantitative")}, {"x2", Obj{{"field", "hi"}}}, {"y", y}}}},
                          Obj{{"mark", Obj{{"type", "circle"}, {"size", 70}, {"color", "#b2182b"}, {"tooltip", true}}},
                              {"encoding", Obj{{"x", x}, {"y", y},
                                               {"tooltip", Tooltip({{"feature", "nominal"}, {"mu_star", "quantitative"}, {"sigma", "quantitative"},
                                                                    {"mu", "quantitative"}})}}},
                              {"params", ZoomParams()}},
                          Obj{{"mark", Obj{{"type", "text"}, {"align", "left"}, {"dx", 7}, {"fontSize", 11}}},
                              {"encoding", Obj{{"x", x}, {"y", y}, {"text", Field("feature", "nominal")}}}}});
    return spec;
}

JsonValue ToVegaLiteSobol(const SobolDocument& doc, int top_k, const HtmlOptions& options) {
    CheckTopK(top_k, "ToVegaLiteSobol");
    if (doc.features.empty()) throw std::invalid_argument("ToVegaLiteSobol: the document has no features");
    std::vector<const SobolDocument::Feature*> order;
    for (const auto& f : doc.features) {
        for (float v : {f.first_order, f.total_order, f.first_order_conf, f.total_order_conf}) {
            if (!std::isfinite(v)) throw std::invalid_argument("ToVegaLiteSobol: a value is not finite");
        }
        order.push_back(&f);
    }
    std::stable_sort(order.begin(), order.end(), [](const auto* a, const auto* b) { return a->total_order > b->total_order; });
    order.resize(std::min<size_t>(order.size(), static_cast<size_t>(top_k)));
    Arr values;
    for (size_t r = 0; r < order.size(); ++r) {
        const auto& f = *order[r];
        values.push_back(Obj{{"feature", f.name}, {"rank", static_cast<int64_t>(r)}, {"index", "total order"}, {"value", NumF(f.total_order)},
                             {"lo", NumF(f.total_order - f.total_order_conf)}, {"hi", NumF(f.total_order + f.total_order_conf)}});
        values.push_back(Obj{{"feature", f.name}, {"rank", static_cast<int64_t>(r)}, {"index", "first order (alone)"}, {"value", NumF(f.first_order)},
                             {"lo", NumF(f.first_order - f.first_order_conf)}, {"hi", NumF(f.first_order + f.first_order_conf)}});
    }
    Obj spec = Base("Sobol indices", options, std::move(values),
                    (doc.target.empty() ? std::string() : "target " + doc.target + ", ") + std::to_string(doc.num_samples) +
                        " samples; the gap between the bars is variance explained only through interactions");
    spec.emplace_back("height", Obj{{"step", 30}});
    Arr order_names;  // one explicit order for both layers (a field sort can't be unioned across them)
    for (const auto* f : order) order_names.push_back(f->name);
    const JsonValue y = Field("feature", "nominal", Obj{{"sort", order_names}, {"title", nullptr}});
    const JsonValue color = Field("index", "nominal", Obj{{"scale", Obj{{"domain", Arr{"total order", "first order (alone)"}}, {"range", Arr{"#f4a582", "#b2182b"}}}},
                                                          {"legend", Obj{{"title", nullptr}, {"orient", "top"}}}});
    spec.emplace_back("layer", Arr{Obj{{"mark", Obj{{"type", "bar"}, {"tooltip", true}}},
                                       {"encoding", Obj{{"y", y}, {"yOffset", Field("index", "nominal", Obj{{"sort", Arr{"total order", "first order (alone)"}}})},
                                                        {"x", Field("value", "quantitative", Obj{{"title", "share of the output's variance"}})},
                                                        {"color", color},
                                                        {"tooltip", Tooltip({{"feature", "nominal"}, {"index", "nominal"}, {"value", "quantitative"}, {"lo", "quantitative"}, {"hi", "quantitative"}})}}}},
                                   Obj{{"mark", Obj{{"type", "rule"}, {"color", "#333"}}},
                                       {"encoding", Obj{{"y", y}, {"yOffset", Field("index", "nominal", Obj{{"sort", Arr{"total order", "first order (alone)"}}})},
                                                        {"x", Field("lo", "quantitative")}, {"x2", Obj{{"field", "hi"}}}}}}});
    return spec;
}

JsonValue ToVegaLiteTrainingLog(const TrainingLogDocument& doc, const HtmlOptions& options) {
    if (doc.scalars.empty() && doc.histograms.empty()) throw std::invalid_argument("ToVegaLiteTrainingLog: the log is empty");
    for (const auto& s : doc.scalars) {  // non-finite values (a diverged loss) are left out, not rejected
        if (s.steps.size() != s.values.size()) throw std::invalid_argument("ToVegaLiteTrainingLog: " + s.tag + " needs one value per step");
    }
    Arr charts;
    const int width = options.width;
    for (const auto& s : doc.scalars) {
        Arr values;
        for (size_t i = 0; i < s.steps.size(); ++i) {
            if (std::isfinite(s.values[i])) values.push_back(Obj{{"step", s.steps[i]}, {"value", Num(s.values[i])}});
        }
        charts.push_back(Obj{{"title", Obj{{"text", s.tag}, {"anchor", "start"}, {"fontSize", 13}}},
                             {"width", width},
                             {"height", 180},
                             {"data", Obj{{"values", std::move(values)}}},
                             {"mark", Obj{{"type", "line"}, {"tooltip", true}, {"color", "#2166ac"}}},
                             {"encoding", Obj{{"x", Field("step", "quantitative", Obj{{"axis", Obj{{"tickMinStep", 1}, {"format", "d"}}}})},
                                              {"y", Field("value", "quantitative", Obj{{"scale", Obj{{"zero", false}}}, {"title", nullptr}})},
                                              {"tooltip", Tooltip({{"step", "quantitative"}, {"value", "quantitative"}})}}},
                             {"params", ZoomParams()}});
    }
    for (const auto& h : doc.histograms) {
        Arr values;
        for (float v : h.values) {
            if (std::isfinite(v)) values.push_back(Obj{{"value", NumF(v)}});
        }
        charts.push_back(Obj{{"title", Obj{{"text", h.tag + " (latest histogram)"}, {"anchor", "start"}, {"fontSize", 13}}},
                             {"width", width},
                             {"height", 140},
                             {"data", Obj{{"values", std::move(values)}}},
                             {"mark", Obj{{"type", "bar"}, {"tooltip", true}, {"color", "#4393c3"}}},
                             {"encoding", Obj{{"x", Field("value", "quantitative", Obj{{"bin", Obj{{"maxbins", 40}}}, {"title", nullptr}})},
                                              {"y", Obj{{"aggregate", "count"}, {"type", "quantitative"}, {"title", "count"}}}}}});
    }
    return Obj{{"$schema", kSchema},
               {"title", Title(options.title.empty() ? "Training log" : options.title)},
               {"vconcat", std::move(charts)},
               {"config", Obj{{"view", Obj{{"stroke", nullptr}}}}}};
}

JsonValue ToVegaLiteFeatureHistogram(const FeatureDashboardDocument& doc, const HtmlOptions& options) {
    (void)ToJson(doc);
    Arr values;
    for (size_t i = 0; i < doc.histogram_counts.size(); ++i) {
        values.push_back(Obj{{"lo", NumF(doc.histogram_edges[i])}, {"hi", NumF(doc.histogram_edges[i + 1])}, {"count", doc.histogram_counts[i]}});
    }
    Obj spec = Base("Activations", options, std::move(values));
    spec.emplace_back("height", 160);
    spec.emplace_back("mark", Obj{{"type", "bar"}, {"tooltip", true}, {"color", "#d6604d"}});
    spec.emplace_back("encoding", Obj{{"x", Field("lo", "quantitative", Obj{{"title", "activation"}})},
                                      {"x2", Obj{{"field", "hi"}}},
                                      {"y", Field("count", "quantitative", Obj{{"title", "count"}})},
                                      {"y2", Obj{{"datum", 0}}},  // fill each bin down to zero
                                      {"tooltip", Tooltip({{"lo", "quantitative"}, {"hi", "quantitative"}, {"count", "quantitative"}})}});
    return spec;
}

// ---- Pages ---------------------------------------------------------------------------------

namespace {

std::string ChartPage(const JsonValue& spec, const std::string& heading, const HtmlOptions& options, const std::string& before,
                      const std::string& after) {
    const std::string spec_json = WriteJson(spec);
    const std::string body = before + "<div id=\"vis\"></div>\n" + after +
                             "<script type=\"application/json\" id=\"spec\">" + ScriptSafe(spec_json) +
                             "</script>\n<script>vegaEmbed('#vis', JSON.parse(document.getElementById('spec').textContent), "
                             "{actions: {export: true, source: true, compiled: false, editor: false}})"
                             ".catch(function (e) { document.getElementById('vis').textContent = 'Could not draw the chart: ' + e; });"
                             "</script>\n";
    return Page(heading, Scripts(options), body);
}

}  // namespace

std::string VegaLitePage(const JsonValue& spec, const std::string& heading, const HtmlOptions& options, const std::string& extra_html) {
    return ChartPage(spec, heading, options, "", extra_html);
}

namespace {
std::string Heading(const HtmlOptions& options, const std::string& fallback) { return options.title.empty() ? fallback : options.title; }
}  // namespace

std::string RenderBarChartHtml(const AttributionDocument& doc, int top_k, const HtmlOptions& options) {
    return VegaLitePage(ToVegaLiteBarChart(doc, top_k, options), Heading(options, "Feature attribution"), options);
}
std::string RenderWaterfallHtml(const AttributionDocument& doc, float baseline_value, const HtmlOptions& options) {
    return VegaLitePage(ToVegaLiteWaterfall(doc, baseline_value, options), Heading(options, "Waterfall"), options);
}
std::string RenderHeatmapHtml(const HeatmapDocument& doc, const HtmlOptions& options) {
    return VegaLitePage(ToVegaLiteHeatmap(doc, options), Heading(options, doc.title.empty() ? "Heatmap" : doc.title), options);
}
std::string RenderBeeswarmHtml(const std::vector<AttributionDocument>& docs, const std::vector<int64_t>& feature_indices,
                               const HtmlOptions& options) {
    return VegaLitePage(ToVegaLiteBeeswarm(docs, feature_indices, options), Heading(options, "Attribution across inputs"), options);
}
std::string RenderPartialDependenceHtml(const PartialDependenceDocument& doc, const PartialDependenceSvgOptions& pd,
                                        const HtmlOptions& options) {
    return VegaLitePage(ToVegaLitePartialDependence(doc, pd, options), Heading(options, "Partial dependence"), options);
}
std::string RenderTornadoHtml(const SensitivityDocument& doc, int top_k, const HtmlOptions& options) {
    return VegaLitePage(ToVegaLiteTornado(doc, top_k, options), Heading(options, "Local sensitivity"), options);
}
std::string RenderCounterfactualHtml(const CounterfactualDocument& doc, int max_rows, const HtmlOptions& options) {
    return VegaLitePage(ToVegaLiteCounterfactual(doc, max_rows, options), Heading(options, "Counterfactual"), options);
}
std::string RenderCounterfactualSetHtml(const std::vector<CounterfactualDocument>& docs, const HtmlOptions& options) {
    return VegaLitePage(ToVegaLiteCounterfactualSet(docs, options), Heading(options, "Counterfactuals"), options);
}
std::string RenderMorrisHtml(const MorrisDocument& doc, const HtmlOptions& options) {
    return VegaLitePage(ToVegaLiteMorris(doc, options), Heading(options, "Morris screening"), options);
}
std::string RenderSobolHtml(const SobolDocument& doc, int top_k, const HtmlOptions& options) {
    return VegaLitePage(ToVegaLiteSobol(doc, top_k, options), Heading(options, "Sobol indices"), options);
}
std::string RenderTrainingLogHtml(const TrainingLogDocument& doc, const HtmlOptions& options) {
    return VegaLitePage(ToVegaLiteTrainingLog(doc, options), Heading(options, "Training log"), options);
}

std::string RenderFeatureDashboardHtml(const FeatureDashboardDocument& doc, const HtmlOptions& options, const std::string& extra_html) {
    (void)ToJson(doc);
    const std::string heading = Heading(options, doc.source + " feature " + std::to_string(doc.feature_index));
    std::string stats = "<h1>" + Escape(heading) + "</h1>\n<p class=\"muted\">active on " +
                        svg_detail::ValueText(100.0 * doc.activation_density) + "% of inputs; largest activation " +
                        svg_detail::ValueText(doc.max_activation) + "</p>\n";
    std::string examples;
    if (!doc.top_examples.empty()) {
        examples = "<h1>Top examples</h1>\n";
        const double scale = doc.max_activation > 0 ? doc.max_activation : 1.0;
        for (const auto& e : doc.top_examples) {
            examples += "<p class=\"muted\">" + Escape(e.label) + "</p>\n";
            if (!e.tokens.empty()) {
                // A protein's residues have no spaces between them: let its line break anywhere.
                const bool residues = std::all_of(e.tokens.begin(), e.tokens.end(), [](const std::string& t) { return t.size() == 1; });
                examples += std::string("<div class=\"tokens") + (residues ? " residues" : "") + "\" dir=\"auto\">" +
                            TokenSpans(e.tokens, e.activations, {}, scale, false) + "</div>\n";
            } else if (!e.activations.empty()) {
                examples += "<p class=\"mono\">activation " + svg_detail::ValueText(e.activations[0]) + "</p>\n";
            }
        }
    }
    if (doc.histogram_counts.empty()) return Page(heading, "", stats + examples + extra_html);
    HtmlOptions chart = options;
    chart.title = "Activations";
    return ChartPage(ToVegaLiteFeatureHistogram(doc, chart), heading, options, stats, examples + extra_html);
}

std::string RenderTokenRelevanceHtml(const TokenRelevanceDocument& doc, const HtmlOptions& options) {
    // RenderTokenStripSvg's checks; non-finite scores are allowed and drawn gray.
    if (doc.relevance.size() != doc.tokens.size()) throw std::invalid_argument("RenderTokenRelevanceHtml: needs one relevance score per piece");
    if (doc.tokens.empty()) throw std::invalid_argument("RenderTokenRelevanceHtml: there are no pieces");
    if (!doc.scored.empty() && doc.scored.size() != doc.tokens.size()) {
        throw std::invalid_argument("RenderTokenRelevanceHtml: needs one scored flag per piece");
    }
    double max_abs = 0;
    for (size_t i = 0; i < doc.relevance.size(); ++i) {
        if (doc.is_scored(i) && std::isfinite(doc.relevance[i])) max_abs = std::max(max_abs, std::abs(static_cast<double>(doc.relevance[i])));
    }
    const std::string kind = doc.granularity == "word" ? "Word" : "Token";
    const std::string heading = Heading(options, kind + " relevance (" + doc.method + ")");
    std::string body = "<h1>" + Escape(heading) + "</h1>\n<div class=\"tokens\" dir=\"auto\">" +
                       TokenSpans(doc.tokens, doc.relevance, doc.scored, max_abs, true) + "</div>\n";
    if (!doc.target.empty()) body += "<p class=\"muted\">explaining <span class=\"mono\">" + Escape(doc.target) + "</span></p>\n";
    if (doc.unassigned && std::isfinite(*doc.unassigned)) {
        body += "<p class=\"muted\">relevance outside the pieces shown: " + svg_detail::ValueText(*doc.unassigned) + "</p>\n";
    }
    body += DivergingLegend(max_abs > 0 ? max_abs : 1.0);
    return Page(heading, "", body);
}

}  // namespace pulsatrix
