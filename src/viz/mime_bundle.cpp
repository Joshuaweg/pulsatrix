#include "pulsatrix/viz/mime_bundle.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/circuit_graph.hpp"
#include "pulsatrix/tensor.hpp"
#include "pulsatrix/viz/colormap.hpp"
#include "pulsatrix/viz/html.hpp"
#include "pulsatrix/viz/plot_data.hpp"
#include "pulsatrix/viz/protein_views.hpp"
#include "pulsatrix/viz/svg.hpp"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace pulsatrix {

void MimeBundle::add(std::string mime, JsonValue data) {
    if (find(mime) != nullptr) throw std::invalid_argument("MimeBundle::add: " + mime + " is already in the bundle");
    entries.emplace_back(std::move(mime), std::move(data));
}

const JsonValue* MimeBundle::find(std::string_view mime) const {
    for (const auto& [type, data] : entries) {
        if (type == mime) return &data;
    }
    return nullptr;
}

JsonValue MimeBundle::to_json_value() const {
    JsonValue::Object o;
    for (const auto& entry : entries) o.push_back(entry);
    return o;
}

std::string Base64Encode(const std::vector<uint8_t>& bytes) {
    static constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 3 <= bytes.size(); i += 3) {
        const uint32_t v = (uint32_t{bytes[i]} << 16) | (uint32_t{bytes[i + 1]} << 8) | bytes[i + 2];
        out += kAlphabet[(v >> 18) & 63];
        out += kAlphabet[(v >> 12) & 63];
        out += kAlphabet[(v >> 6) & 63];
        out += kAlphabet[v & 63];
    }
    if (i + 1 == bytes.size()) {
        const uint32_t v = uint32_t{bytes[i]} << 16;
        out += kAlphabet[(v >> 18) & 63];
        out += kAlphabet[(v >> 12) & 63];
        out += "==";
    } else if (i + 2 == bytes.size()) {
        const uint32_t v = (uint32_t{bytes[i]} << 16) | (uint32_t{bytes[i + 1]} << 8);
        out += kAlphabet[(v >> 18) & 63];
        out += kAlphabet[(v >> 12) & 63];
        out += kAlphabet[(v >> 6) & 63];
        out += '=';
    }
    return out;
}

namespace {

constexpr int64_t kMaxVectorCells = 64 * 64;  // larger heatmaps are sent as a PNG
constexpr const char* kVegaLiteV5Schema = "https://vega.github.io/schema/vega-lite/v5.json";

std::string Number(double v) {
    if (!std::isfinite(v)) return std::isnan(v) ? "nan" : (v > 0 ? "inf" : "-inf");
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.4g", v);
    return buf;
}

std::string ShapeText(const std::vector<int64_t>& shape) {
    std::string s = "(";
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + (shape.size() == 1 ? ",)" : ")");
}

std::string Escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            default: out += c;
        }
    }
    return out;
}

/** @brief A pulsatrix spec (Vega-Lite 6 schema) as the version 5 one front ends render; the specs
 *         use no feature version 6 added. */
JsonValue AsVegaLite5(JsonValue spec) {
    JsonValue::Object o = spec.as_object();
    for (auto& [key, value] : o) {
        if (key == "$schema") value = kVegaLiteV5Schema;
    }
    return o;
}

MimeBundle Chart(std::string text, std::string svg, std::optional<JsonValue> vega_lite = std::nullopt) {
    MimeBundle b;
    if (vega_lite) b.add(kVegaLiteMimeType, AsVegaLite5(std::move(*vega_lite)));
    if (!svg.empty()) b.add("image/svg+xml", std::move(svg));
    b.add("text/plain", std::move(text));
    return b;
}

struct Stats {
    double min = std::numeric_limits<double>::infinity(), max = -std::numeric_limits<double>::infinity(), mean = 0, std = 0;
    int64_t nonfinite = 0;
};

Stats Summarize(const std::vector<float>& v) {
    Stats s;
    double sum = 0, sq = 0;
    int64_t n = 0;
    for (float x : v) {
        if (!std::isfinite(x)) {
            ++s.nonfinite;
            continue;
        }
        s.min = std::min(s.min, static_cast<double>(x));
        s.max = std::max(s.max, static_cast<double>(x));
        sum += x;
        ++n;
    }
    if (n == 0) {
        s.min = s.max = std::numeric_limits<double>::quiet_NaN();
        s.mean = s.std = std::numeric_limits<double>::quiet_NaN();
        return s;
    }
    s.mean = sum / static_cast<double>(n);
    for (float x : v) {
        if (std::isfinite(x)) sq += (x - s.mean) * (x - s.mean);
    }
    s.std = std::sqrt(sq / static_cast<double>(n));
    return s;
}

const char* DeviceName(DeviceType d) {
    switch (d) {
        case DeviceType::Cpu: return "cpu";
        case DeviceType::Cuda: return "cuda";
        case DeviceType::Hip: return "hip";
    }
    return "?";
}

/** @brief A heatmap as an RGB PNG, one pixel per cell, colored like RenderHeatmapSvg. */
std::string HeatmapPngBase64(const HeatmapDocument& doc) {
    HeatmapGrid finite{{}, 1, 0};
    for (float v : doc.values) {
        if (std::isfinite(v)) finite.values.push_back(v);
    }
    finite.cols = static_cast<int64_t>(finite.values.size());
    const HeatmapColorScale scale = finite.values.empty() ? HeatmapColorScale{0, 1, false} : ComputeHeatmapColorScale(finite);
    std::vector<uint8_t> rgb(doc.values.size() * 3);
    for (size_t i = 0; i < doc.values.size(); ++i) {
        const float v = doc.values[i];
        RgbColor c{0.74f, 0.74f, 0.74f};  // non-finite: gray
        if (std::isfinite(v)) {
            const float m = scale.scale_max > 0 ? scale.scale_max : 1.0f;
            c = scale.is_signed ? DivergingColormap(std::clamp(v / m, -1.0f, 1.0f)) : ViridisColormap(std::clamp(v / m, 0.0f, 1.0f));
        }
        rgb[3 * i] = static_cast<uint8_t>(std::lround(255.0f * c.r));
        rgb[3 * i + 1] = static_cast<uint8_t>(std::lround(255.0f * c.g));
        rgb[3 * i + 2] = static_cast<uint8_t>(std::lround(255.0f * c.b));
    }
    std::vector<uint8_t> png;
    stbi_write_png_to_func(
        [](void* context, void* data, int size) {
            auto* out = static_cast<std::vector<uint8_t>*>(context);
            const auto* bytes = static_cast<const uint8_t*>(data);
            out->insert(out->end(), bytes, bytes + size);
        },
        &png, static_cast<int>(doc.cols), static_cast<int>(doc.rows), 3, rgb.data(), static_cast<int>(doc.cols * 3));
    if (png.empty()) throw std::runtime_error("mime_bundle_repr: could not encode the heatmap as PNG");
    return Base64Encode(png);
}

std::string HeatmapText(const HeatmapDocument& doc) {
    const Stats s = Summarize(doc.values);
    return (doc.title.empty() ? std::string("Heatmap") : doc.title) + " (" + std::to_string(doc.rows) + " x " +
           std::to_string(doc.cols) + "): min " + Number(s.min) + ", max " + Number(s.max) + ", mean " + Number(s.mean);
}

std::string AttributionText(const AttributionDocument& doc) {
    std::string text = "Attribution (" + (doc.method.empty() ? std::string("unknown method") : doc.method) + ") " +
                       ShapeText(doc.shape);
    std::vector<size_t> order(doc.values.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    const size_t k = std::min<size_t>(5, order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(), [&](size_t a, size_t b) {
        const float x = std::isfinite(doc.values[a]) ? std::fabs(doc.values[a]) : -1.0f;
        const float y = std::isfinite(doc.values[b]) ? std::fabs(doc.values[b]) : -1.0f;
        return x > y || (x == y && a < b);
    });
    if (k > 0) {
        text += "\nlargest |value|:";
        for (size_t i = 0; i < k; ++i) text += " [" + std::to_string(order[i]) + "] " + Number(doc.values[order[i]]);
    }
    return text;
}

int64_t Depth(const CircuitGraphDocument& doc, std::map<int64_t, int64_t>& depth, int64_t id,
              const std::map<int64_t, std::vector<int64_t>>& parents, int guard) {
    if (auto it = depth.find(id); it != depth.end()) return it->second;
    if (guard > static_cast<int>(doc.nodes.size())) throw std::invalid_argument("ToVegaLiteCircuitGraph: the graph has a cycle");
    int64_t d = 0;
    if (auto it = parents.find(id); it != parents.end()) {
        for (int64_t p : it->second) d = std::max(d, Depth(doc, depth, p, parents, guard + 1) + 1);
    }
    depth[id] = d;
    return d;
}

}  // namespace

JsonValue ToVegaLiteCircuitGraph(const CircuitGraphDocument& doc) {
    using Obj = JsonValue::Object;
    using Arr = JsonValue::Array;
    if (doc.nodes.empty()) throw std::invalid_argument("ToVegaLiteCircuitGraph: the graph has no nodes");
    std::map<int64_t, size_t> index;
    for (size_t i = 0; i < doc.nodes.size(); ++i) index[doc.nodes[i].id] = i;
    std::map<int64_t, std::vector<int64_t>> parents;
    for (const auto& e : doc.edges) {
        if (!index.count(e.from) || !index.count(e.to)) {
            throw std::invalid_argument("ToVegaLiteCircuitGraph: an edge names a node that isn't in the graph");
        }
        parents[e.to].push_back(e.from);
    }
    std::map<int64_t, int64_t> depth;
    std::map<int64_t, int64_t> row_of, rows_at_depth;
    for (const auto& n : doc.nodes) {
        const int64_t d = Depth(doc, depth, n.id, parents, 0);
        row_of[n.id] = rows_at_depth[d]++;
    }
    double max_effect = 0, max_weight = 0;
    for (const auto& n : doc.nodes) max_effect = std::max(max_effect, std::fabs(static_cast<double>(n.ablation_effect)));
    for (const auto& e : doc.edges) max_weight = std::max(max_weight, std::fabs(static_cast<double>(e.weight)));
    auto num = [](double v) { return std::isfinite(v) ? JsonValue(v) : JsonValue(); };
    Arr nodes, edges;
    for (const auto& n : doc.nodes) {
        nodes.push_back(Obj{{"id", n.id},
                            {"x", depth[n.id]},
                            {"y", row_of[n.id]},
                            {"label", n.label.value_or(n.op_type + " " + std::to_string(n.id))},
                            {"op_type", n.op_type},
                            {"effect", num(n.ablation_effect)}});
    }
    for (const auto& e : doc.edges) {
        edges.push_back(Obj{{"x", depth[e.from]},
                            {"y", row_of[e.from]},
                            {"x2", depth[e.to]},
                            {"y2", row_of[e.to]},
                            {"weight", num(e.weight)},
                            {"magnitude", num(std::fabs(static_cast<double>(e.weight)))}});
    }
    const double m = max_effect > 0 ? max_effect : 1.0;
    const auto hex = [](float t) {
        const RgbColor c = DivergingColormap(t);
        char buf[8];
        std::snprintf(buf, sizeof buf, "#%02x%02x%02x", static_cast<int>(std::lround(255 * c.r)),
                      static_cast<int>(std::lround(255 * c.g)), static_cast<int>(std::lround(255 * c.b)));
        return std::string(buf);
    };
    Obj x{{"field", "x"}, {"type", "quantitative"}, {"axis", nullptr}};
    Obj y{{"field", "y"}, {"type", "quantitative"}, {"axis", nullptr}, {"scale", Obj{{"reverse", true}}}};
    Arr layers;
    layers.push_back(Obj{
        {"data", Obj{{"values", edges}}},
        {"mark", Obj{{"type", "rule"}, {"color", "#9e9e9e"}}},
        {"encoding", Obj{{"x", x},
                         {"y", y},
                         {"x2", Obj{{"field", "x2"}}},
                         {"y2", Obj{{"field", "y2"}}},
                         {"strokeWidth", Obj{{"field", "magnitude"}, {"type", "quantitative"}, {"legend", nullptr},
                                             {"scale", Obj{{"domain", Arr{JsonValue(0.0), JsonValue(max_weight > 0 ? max_weight : 1.0)}},
                                                           {"range", Arr{JsonValue(0.5), JsonValue(4.0)}}}}}},
                         {"tooltip", Arr{Obj{{"field", "weight"}, {"type", "quantitative"}, {"format", ".4~g"}}}}}}});
    layers.push_back(Obj{
        {"data", Obj{{"values", nodes}}},
        {"mark", Obj{{"type", "circle"}, {"size", 300}, {"stroke", "#424242"}, {"opacity", 1}}},
        {"encoding", Obj{{"x", x},
                         {"y", y},
                         {"color", Obj{{"field", "effect"}, {"type", "quantitative"}, {"title", "ablation effect"},
                                       {"scale", Obj{{"domain", Arr{JsonValue(-m), JsonValue(0.0), JsonValue(m)}},
                                                     {"range", Arr{hex(-1.0f), hex(0.0f), hex(1.0f)}}}}}},
                         {"tooltip", Arr{Obj{{"field", "label"}, {"type", "nominal"}},
                                         Obj{{"field", "op_type"}, {"type", "nominal"}},
                                         Obj{{"field", "effect"}, {"type", "quantitative"}, {"format", ".4~g"}}}}}}});
    layers.push_back(Obj{{"data", Obj{{"values", nodes}}},
                         {"mark", Obj{{"type", "text"}, {"dy", -16}, {"fontSize", 11}}},
                         {"encoding", Obj{{"x", x}, {"y", y}, {"text", Obj{{"field", "label"}}}}}});
    int64_t columns = 0, rows = 0;
    for (const auto& [d, count] : rows_at_depth) {
        columns = std::max(columns, d + 1);
        rows = std::max(rows, count);
    }
    return Obj{{"$schema", "https://vega.github.io/schema/vega-lite/v6.json"},
               {"title", Obj{{"text", "Circuit graph"}, {"anchor", "start"}, {"fontSize", 14}}},
               {"width", std::max<int64_t>(160, 120 * columns)},
               {"height", std::max<int64_t>(80, 60 * rows)},
               {"layer", layers},
               {"config", Obj{{"view", Obj{{"stroke", nullptr}}}}}};
}

// ---- Core types --------------------------------------------------------------------------------

MimeBundle mime_bundle_repr(const Tensor& tensor) {
    std::vector<int64_t> shape;
    for (int64_t d = 0; d < tensor.rank(); ++d) shape.push_back(tensor.shape().dim(static_cast<int>(d)));
    const std::vector<float> v = tensor.numel() > 0 ? tensor.to_host_vector() : std::vector<float>{};
    const Stats s = Summarize(v);
    std::string text = "Tensor " + ShapeText(shape) + " float32 on " + DeviceName(tensor.device()) + ", " +
                       std::to_string(v.size()) + " values";
    if (!v.empty()) {
        text += "\nmin " + Number(s.min) + ", max " + Number(s.max) + ", mean " + Number(s.mean) + ", std " + Number(s.std);
        if (s.nonfinite > 0) text += ", " + std::to_string(s.nonfinite) + " not finite";
    }
    // Leading values: up to 6 rows of 8 for a matrix, the first 8 otherwise.
    const int64_t cols = shape.size() >= 2 ? shape.back() : static_cast<int64_t>(v.size());
    const int64_t show_cols = std::min<int64_t>(cols, 8);
    const int64_t show_rows = shape.size() >= 2 ? std::min<int64_t>(static_cast<int64_t>(v.size()) / std::max<int64_t>(cols, 1), 6) : 1;
    std::string table;
    for (int64_t r = 0; r < show_rows && !v.empty(); ++r) {
        table += "<tr>";
        text += "\n";
        for (int64_t c = 0; c < show_cols; ++c) {
            const std::string cell = Number(v[static_cast<size_t>(r * cols + c)]);
            table += "<td style=\"padding:1px 6px;text-align:right\">" + cell + "</td>";
            text += (c ? " " : "  ") + cell;
        }
        if (show_cols < cols) {
            table += "<td>&hellip;</td>";
            text += " ...";
        }
        table += "</tr>";
    }
    std::string html = "<div style=\"font-family:monospace\"><b>Tensor</b> " + Escape(ShapeText(shape)) + " float32 on " +
                       DeviceName(tensor.device()) + "<br>";
    if (!v.empty()) {
        html += "min " + Number(s.min) + " &middot; max " + Number(s.max) + " &middot; mean " + Number(s.mean) +
                " &middot; std " + Number(s.std);
        if (s.nonfinite > 0) html += " &middot; " + std::to_string(s.nonfinite) + " not finite";
        html += "<table style=\"border-collapse:collapse;margin-top:4px\">" + table + "</table>";
    }
    html += "</div>";
    MimeBundle b;
    b.add("text/html", html);
    b.add("text/plain", text);
    return b;
}

MimeBundle mime_bundle_repr(const Attribution& attribution) {
    const AttributionDocument doc = ToAttributionDocument(attribution);
    const std::vector<int64_t>& shape = doc.shape;
    const std::string method = doc.method.empty() ? "attribution" : doc.method;
    if (shape.size() >= 3) {
        // An image: (N, C, H, W) with the channels summed, (N, H, W), or larger ranks by their
        // first entries; the first example of a batch.
        const int64_t h = shape[shape.size() - 2], w = shape.back();
        const int64_t channels = shape.size() >= 4 ? shape[shape.size() - 3] : 1;
        HeatmapDocument heat;
        heat.rows = h;
        heat.cols = w;
        heat.values.assign(static_cast<size_t>(h * w), 0.0f);
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t i = 0; i < h * w; ++i) heat.values[static_cast<size_t>(i)] += doc.values[static_cast<size_t>(c * h * w + i)];
        }
        const int64_t examples = static_cast<int64_t>(doc.values.size()) / (channels * h * w);
        heat.title = method + (channels > 1 ? ", channels summed" : "") +
                     (examples > 1 ? ", example 1 of " + std::to_string(examples) : "");
        MimeBundle b = mime_bundle_repr(heat);
        for (auto& [mime, data] : b.entries) {
            if (mime == "text/plain") data = AttributionText(doc) + "\n" + HeatmapText(heat);
        }
        return b;
    }
    // Features: the first row of a batch.
    AttributionDocument row = doc;
    if (shape.size() == 2 && shape[0] > 1) {
        row.shape = {shape[1]};
        row.values.resize(static_cast<size_t>(shape[1]));
    }
    SvgOptions svg;
    svg.title = method + (shape.size() == 2 && shape[0] > 1 ? ", example 1 of " + std::to_string(shape[0]) : "");
    HtmlOptions html;
    html.title = svg.title;
    return Chart(AttributionText(doc), RenderBarChartSvg(row, 10, svg), ToVegaLiteBarChart(row, 10, html));
}

MimeBundle mime_bundle_repr(const CircuitGraph& graph) { return mime_bundle_repr(ToCircuitGraphDocument(graph)); }

// ---- Documents ---------------------------------------------------------------------------------

MimeBundle mime_bundle_repr(const AttributionDocument& doc) {
    return Chart(AttributionText(doc), RenderBarChartSvg(doc), ToVegaLiteBarChart(doc));
}

MimeBundle mime_bundle_repr(const HeatmapDocument& doc) {
    if (doc.rows <= 0 || doc.cols <= 0 || static_cast<size_t>(doc.rows * doc.cols) != doc.values.size()) {
        throw std::invalid_argument("mime_bundle_repr: the heatmap's values don't fill its grid");
    }
    if (doc.rows * doc.cols > kMaxVectorCells) {
        MimeBundle b;
        b.add("image/png", HeatmapPngBase64(doc));
        b.add("text/plain", HeatmapText(doc));
        return b;
    }
    return Chart(HeatmapText(doc), RenderHeatmapSvg(doc), ToVegaLiteHeatmap(doc));
}

MimeBundle mime_bundle_repr(const TokenRelevanceDocument& doc) {
    std::string text = (doc.granularity == "word" ? "Word" : "Token") + std::string(" relevance (") + doc.method + ")";
    if (!doc.target.empty()) text += ", explaining " + doc.target;
    text += ":";
    for (size_t i = 0; i < doc.tokens.size() && i < doc.relevance.size(); ++i) {
        if (doc.is_scored(i)) text += "\n  " + doc.tokens[i] + "  " + Number(doc.relevance[i]);
    }
    return Chart(text, RenderTokenStripSvg(doc));
}

MimeBundle mime_bundle_repr(const CircuitGraphDocument& doc) {
    std::string text = "Circuit graph: " + std::to_string(doc.nodes.size()) + " nodes, " + std::to_string(doc.edges.size()) + " edges";
    for (const auto& n : doc.nodes) {
        text += "\n  " + std::to_string(n.id) + " " + n.label.value_or(n.op_type) + "  effect " + Number(n.ablation_effect);
    }
    return Chart(text, "", ToVegaLiteCircuitGraph(doc));
}

MimeBundle mime_bundle_repr(const TrainingLogDocument& doc) {
    std::string text = "Training log:";
    for (const auto& s : doc.scalars) {
        text += "\n  " + s.tag + ": " + std::to_string(s.values.size()) + " points" +
                (s.values.empty() ? "" : ", last " + Number(s.values.back()));
    }
    return Chart(text, "", ToVegaLiteTrainingLog(doc));
}

MimeBundle mime_bundle_repr(const PartialDependenceDocument& doc) {
    return Chart((doc.method == "ale" ? "Accumulated local effects" : "Partial dependence") + std::string(" of ") + doc.feature,
                 RenderPartialDependenceSvg(doc), ToVegaLitePartialDependence(doc));
}

MimeBundle mime_bundle_repr(const SensitivityDocument& doc) {
    return Chart("Local sensitivity of " + doc.target, RenderTornadoSvg(doc), ToVegaLiteTornado(doc));
}

MimeBundle mime_bundle_repr(const CounterfactualDocument& doc) {
    return Chart("Counterfactual for " + doc.target + (doc.valid ? "" : " (not valid)"), RenderCounterfactualSvg(doc),
                 ToVegaLiteCounterfactual(doc));
}

MimeBundle mime_bundle_repr(const MorrisDocument& doc) {
    return Chart("Morris screening of " + doc.target, RenderMorrisSvg(doc), ToVegaLiteMorris(doc));
}

MimeBundle mime_bundle_repr(const SobolDocument& doc) {
    return Chart("Sobol indices of " + doc.target, RenderSobolSvg(doc), ToVegaLiteSobol(doc));
}

MimeBundle mime_bundle_repr(const MutationMapDocument& doc) {
    return Chart("Mutation map: " + doc.title, RenderMutationMapSvg(doc));
}

MimeBundle mime_bundle_repr(const SequenceLogoDocument& doc) {
    return Chart("Sequence logo: " + doc.title, RenderSequenceLogoSvg(doc));
}

MimeBundle mime_bundle_repr(const ContactMapDocument& doc) {
    return Chart("Contact map: " + doc.title, RenderContactMapSvg(doc));
}

MimeBundle mime_bundle_repr(const ResidueTracksDocument& doc) {
    return Chart("Residue tracks: " + doc.title, RenderResidueTracksSvg(doc));
}

}  // namespace pulsatrix
