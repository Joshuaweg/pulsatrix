#include "pulsatrix/viz/document.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/json.hpp"
#include "document_io.hpp"

namespace pulsatrix {

using namespace document_io;

std::string VizDocumentKind(std::string_view json) { return KindOf(ParseJson(json)); }

// ---- attribution ---------------------------------------------------------------------------

AttributionDocument ToAttributionDocument(const Attribution& attr) {
    AttributionDocument doc;
    doc.method = attr.method;
    const Shape& shape = attr.values.shape();
    for (int64_t i = 0; i < shape.rank(); ++i) {
        doc.shape.push_back(shape.dim(static_cast<size_t>(i)));
    }
    doc.values = attr.values.to_host_vector();
    doc.metadata.insert(attr.metadata.begin(), attr.metadata.end());
    return doc;
}

Attribution ToAttribution(const AttributionDocument& doc, DeviceBackend* backend) {
    return Attribution{doc.method, Tensor(Shape(doc.shape), backend, doc.values),
                       std::unordered_map<std::string, std::string>(doc.metadata.begin(), doc.metadata.end())};
}

std::string ToJson(const AttributionDocument& doc) {
    size_t n = 0;
    if (!ProductAtMost(doc.shape, doc.values.size(), n) || n != doc.values.size()) {
        Invalid("attribution", "values: " + std::to_string(doc.values.size()) + " values don't fill the shape");
    }
    DocWriter w("attribution");
    w.root().add("method", doc.method);
    w.root().add("shape", Ints(doc.shape));
    w.root().add("values", w.floats(doc.values, "/values"));
    JsonValue metadata{JsonValue::Object{}};
    for (const auto& [k, v] : doc.metadata) {
        metadata.add(k, v);
    }
    w.root().add("metadata", std::move(metadata));
    return w.finish();
}

AttributionDocument ParseAttributionDocument(std::string_view json) {
    DocReader r(json, "attribution");
    const JsonValue& root = r.root();
    AttributionDocument doc;
    doc.method = r.string(r.member(root, "", "method"), "/method");
    doc.shape = r.ints(r.member(root, "", "shape"), "/shape", 0, kMaxIndex);
    doc.values = r.floats(r.member(root, "", "values"), "/values");
    size_t n = 0;
    if (!ProductAtMost(doc.shape, doc.values.size(), n) || n != doc.values.size()) {
        r.fail("/values", std::to_string(doc.values.size()) + " values don't fill the shape");
    }
    const JsonValue& metadata = r.member(root, "", "metadata");
    for (const auto& [k, v] : r.object(metadata, "/metadata")) {
        doc.metadata.emplace(k, r.string(v, "/metadata/" + k));
    }
    r.finish();
    return doc;
}

// ---- heatmap -------------------------------------------------------------------------------

HeatmapDocument ToHeatmapDocument(const HeatmapGrid& grid, std::string title) {
    HeatmapDocument doc;
    doc.title = std::move(title);
    doc.rows = grid.rows;
    doc.cols = grid.cols;
    doc.values = grid.values;
    return doc;
}

HeatmapGrid ToHeatmapGrid(const HeatmapDocument& doc) { return HeatmapGrid{doc.values, doc.rows, doc.cols}; }

namespace {

// The problem with a heatmap's sizes, as "<field>: <problem>", or empty if there is none.
std::string HeatmapProblem(const HeatmapDocument& doc) {
    size_t n = 0;
    if (!ProductAtMost({doc.rows, doc.cols}, doc.values.size(), n) || n != doc.values.size()) {
        return "/values: " + std::to_string(doc.values.size()) + " values for a " + std::to_string(doc.rows) + " x " +
               std::to_string(doc.cols) + " grid";
    }
    if (!doc.row_labels.empty() && static_cast<int64_t>(doc.row_labels.size()) != doc.rows) {
        return "/row_labels: needs one label per row, or none";
    }
    if (!doc.col_labels.empty() && static_cast<int64_t>(doc.col_labels.size()) != doc.cols) {
        return "/col_labels: needs one label per column, or none";
    }
    return "";
}

}  // namespace

std::string ToJson(const HeatmapDocument& doc) {
    if (std::string problem = HeatmapProblem(doc); !problem.empty()) {
        Invalid("heatmap", problem);
    }
    DocWriter w("heatmap");
    w.root().add("title", doc.title);
    w.root().add("rows", JsonValue(doc.rows));
    w.root().add("cols", JsonValue(doc.cols));
    w.root().add("values", w.floats(doc.values, "/values"));
    w.root().add("row_labels", Strings(doc.row_labels));
    w.root().add("col_labels", Strings(doc.col_labels));
    return w.finish();
}

HeatmapDocument ParseHeatmapDocument(std::string_view json) {
    DocReader r(json, "heatmap");
    const JsonValue& root = r.root();
    HeatmapDocument doc;
    doc.title = r.string(r.member(root, "", "title"), "/title");
    doc.rows = r.integer(r.member(root, "", "rows"), "/rows", 0, kMaxIndex);
    doc.cols = r.integer(r.member(root, "", "cols"), "/cols", 0, kMaxIndex);
    doc.values = r.floats(r.member(root, "", "values"), "/values");
    doc.row_labels = r.strings(r.member(root, "", "row_labels"), "/row_labels");
    doc.col_labels = r.strings(r.member(root, "", "col_labels"), "/col_labels");
    if (std::string problem = HeatmapProblem(doc); !problem.empty()) {
        size_t colon = problem.find(':');
        r.fail(problem.substr(0, colon), problem.substr(colon + 2));
    }
    r.finish();
    return doc;
}

// ---- token relevance -----------------------------------------------------------------------

std::string ToJson(const TokenRelevanceDocument& doc) {
    if (doc.tokens.size() != doc.relevance.size()) {
        Invalid("token_relevance", "/relevance: needs one score per token");
    }
    if (!doc.scored.empty() && doc.scored.size() != doc.tokens.size()) {
        Invalid("token_relevance", "/scored: needs one flag per token");
    }
    if (doc.granularity != "token" && doc.granularity != "word") {
        Invalid("token_relevance", "/granularity: must be \"token\" or \"word\"");
    }
    DocWriter w("token_relevance");
    w.root().add("method", doc.method);
    w.root().add("tokens", Strings(doc.tokens));
    w.root().add("relevance", w.floats(doc.relevance, "/relevance"));
    w.root().add("target", doc.target);
    // Optional fields, written only when they say something, so older documents stay unchanged.
    if (doc.granularity != "token") w.root().add("granularity", doc.granularity);
    bool all_scored = true;
    for (bool b : doc.scored) all_scored = all_scored && b;
    if (!all_scored) {
        JsonValue flags{JsonValue::Array{}};
        for (bool b : doc.scored) flags.push_back(JsonValue(b));
        w.root().add("scored", std::move(flags));
    }
    if (doc.unassigned) w.root().add("unassigned", w.num(*doc.unassigned, "/unassigned"));
    return w.finish();
}

TokenRelevanceDocument ParseTokenRelevanceDocument(std::string_view json) {
    DocReader r(json, "token_relevance");
    const JsonValue& root = r.root();
    TokenRelevanceDocument doc;
    doc.method = r.string(r.member(root, "", "method"), "/method");
    doc.tokens = r.strings(r.member(root, "", "tokens"), "/tokens");
    doc.relevance = r.floats(r.member(root, "", "relevance"), "/relevance");
    doc.target = r.string(r.member(root, "", "target"), "/target");
    if (doc.tokens.size() != doc.relevance.size()) {
        r.fail("/relevance", "needs one score per token");
    }
    if (const JsonValue* g = root.find("granularity")) {
        doc.granularity = r.string(*g, "/granularity");
        if (doc.granularity != "token" && doc.granularity != "word") r.fail("/granularity", "must be \"token\" or \"word\"");
    }
    if (const JsonValue* flags = root.find("scored")) {
        const JsonValue::Array& a = r.array(*flags, "/scored");
        if (a.size() != doc.tokens.size()) r.fail("/scored", "needs one flag per token");
        for (size_t i = 0; i < a.size(); ++i) doc.scored.push_back(r.boolean(a[i], "/scored/" + std::to_string(i)));
    }
    if (const JsonValue* u = root.find("unassigned")) doc.unassigned = r.number(*u, "/unassigned");
    r.finish();
    return doc;
}

// ---- circuit graph -------------------------------------------------------------------------

namespace {

struct OpTypeEntry {
    OpType type;
    const char* name;
};

constexpr OpTypeEntry kOpTypeNames[] = {
    {OpType::Linear, "Linear"},       {OpType::Conv, "Conv"},
    {OpType::Activation, "Activation"}, {OpType::Elementwise, "Elementwise"},
    {OpType::Reduction, "Reduction"}, {OpType::Normalization, "Normalization"},
    {OpType::Pooling, "Pooling"},     {OpType::Embedding, "Embedding"},
    {OpType::Composite, "Composite"}, {OpType::Recurrent, "Recurrent"},
    {OpType::Attention, "Attention"},
};

bool IsOpTypeName(std::string_view name) {
    return std::any_of(std::begin(kOpTypeNames), std::end(kOpTypeNames), [&](const OpTypeEntry& e) { return name == e.name; });
}

// The problem with a circuit's nodes and edges, as "<pointer>: <problem>", or empty.
std::string CircuitProblem(const CircuitGraphDocument& doc) {
    std::set<int64_t> ids;
    for (size_t i = 0; i < doc.nodes.size(); ++i) {
        const auto& n = doc.nodes[i];
        if (n.id < 0 || !ids.insert(n.id).second) {
            return Index("/nodes", i) + "/id: " + std::to_string(n.id) + " is negative or repeats";
        }
        if (!IsOpTypeName(n.op_type)) {
            return Index("/nodes", i) + "/op_type: \"" + n.op_type + "\" is not an OpType name";
        }
    }
    for (size_t i = 0; i < doc.edges.size(); ++i) {
        if (ids.count(doc.edges[i].from) == 0) {
            return Index("/edges", i) + "/from: no node " + std::to_string(doc.edges[i].from);
        }
        if (ids.count(doc.edges[i].to) == 0) {
            return Index("/edges", i) + "/to: no node " + std::to_string(doc.edges[i].to);
        }
    }
    return "";
}

}  // namespace

const char* OpTypeName(OpType op_type) {
    for (const OpTypeEntry& e : kOpTypeNames) {
        if (e.type == op_type) {
            return e.name;
        }
    }
    throw std::invalid_argument("OpTypeName: unknown OpType");
}

OpType OpTypeFromName(std::string_view name) {
    for (const OpTypeEntry& e : kOpTypeNames) {
        if (name == e.name) {
            return e.type;
        }
    }
    throw std::invalid_argument("\"" + std::string(name) + "\" is not an OpType name");
}

CircuitGraphDocument ToCircuitGraphDocument(const CircuitGraph& graph) {
    CircuitGraphDocument doc;
    for (const CircuitNode& n : graph.nodes()) {
        doc.nodes.push_back({static_cast<int64_t>(n.id), OpTypeName(n.op_type), n.label, n.ablation_effect});
    }
    for (const CircuitEdge& e : graph.edges()) {
        doc.edges.push_back({static_cast<int64_t>(e.from), static_cast<int64_t>(e.to), e.weight});
    }
    return doc;
}

CircuitGraph ToCircuitGraph(const CircuitGraphDocument& doc) {
    std::vector<CircuitNode> nodes;
    for (const auto& n : doc.nodes) {
        nodes.push_back({static_cast<NodeId>(n.id), OpTypeFromName(n.op_type), n.label, n.ablation_effect});
    }
    std::vector<CircuitEdge> edges;
    for (const auto& e : doc.edges) {
        edges.push_back({static_cast<NodeId>(e.from), static_cast<NodeId>(e.to), e.weight});
    }
    return CircuitGraph(std::move(nodes), std::move(edges));
}

std::string ToJson(const CircuitGraphDocument& doc) {
    if (std::string problem = CircuitProblem(doc); !problem.empty()) {
        Invalid("circuit_graph", problem);
    }
    DocWriter w("circuit_graph");
    JsonValue nodes{JsonValue::Array{}};
    for (size_t i = 0; i < doc.nodes.size(); ++i) {
        const auto& n = doc.nodes[i];
        JsonValue node{JsonValue::Object{}};
        node.add("id", JsonValue(n.id));
        node.add("op_type", n.op_type);
        node.add("label", n.label ? JsonValue(*n.label) : JsonValue());
        node.add("ablation_effect", w.num(n.ablation_effect, Index("/nodes", i) + "/ablation_effect"));
        nodes.push_back(std::move(node));
    }
    JsonValue edges{JsonValue::Array{}};
    for (size_t i = 0; i < doc.edges.size(); ++i) {
        const auto& e = doc.edges[i];
        JsonValue edge{JsonValue::Object{}};
        edge.add("from", JsonValue(e.from));
        edge.add("to", JsonValue(e.to));
        edge.add("weight", w.num(e.weight, Index("/edges", i) + "/weight"));
        edges.push_back(std::move(edge));
    }
    w.root().add("nodes", std::move(nodes));
    w.root().add("edges", std::move(edges));
    return w.finish();
}

CircuitGraphDocument ParseCircuitGraphDocument(std::string_view json) {
    DocReader r(json, "circuit_graph");
    const JsonValue& root = r.root();
    CircuitGraphDocument doc;
    const JsonValue::Array& nodes = r.array(r.member(root, "", "nodes"), "/nodes");
    for (size_t i = 0; i < nodes.size(); ++i) {
        const std::string p = Index("/nodes", i);
        r.object(nodes[i], p);
        CircuitGraphDocument::Node n;
        n.id = r.integer(r.member(nodes[i], p, "id"), p + "/id", 0, kMaxIndex);
        n.op_type = r.string(r.member(nodes[i], p, "op_type"), p + "/op_type");
        const JsonValue& label = r.member(nodes[i], p, "label");
        if (!label.is_null()) {
            n.label = r.string(label, p + "/label");
        }
        n.ablation_effect = r.number(r.member(nodes[i], p, "ablation_effect"), p + "/ablation_effect");
        doc.nodes.push_back(std::move(n));
    }
    const JsonValue::Array& edges = r.array(r.member(root, "", "edges"), "/edges");
    for (size_t i = 0; i < edges.size(); ++i) {
        const std::string p = Index("/edges", i);
        r.object(edges[i], p);
        CircuitGraphDocument::Edge e;
        e.from = r.integer(r.member(edges[i], p, "from"), p + "/from", 0, kMaxIndex);
        e.to = r.integer(r.member(edges[i], p, "to"), p + "/to", 0, kMaxIndex);
        e.weight = r.number(r.member(edges[i], p, "weight"), p + "/weight");
        doc.edges.push_back(e);
    }
    if (std::string problem = CircuitProblem(doc); !problem.empty()) {
        size_t colon = problem.find(':');
        r.fail(problem.substr(0, colon), problem.substr(colon + 2));
    }
    r.finish();
    return doc;
}

// ---- training log --------------------------------------------------------------------------

namespace {

constexpr int64_t kMinStep = std::numeric_limits<int>::min();
constexpr int64_t kMaxStep = std::numeric_limits<int>::max();

std::string TrainingLogProblem(const TrainingLogDocument& doc) {
    std::set<std::string> tags;
    for (size_t i = 0; i < doc.scalars.size(); ++i) {
        const auto& s = doc.scalars[i];
        if (!tags.insert(s.tag).second) {
            return Index("/scalars", i) + "/tag: \"" + s.tag + "\" repeats";
        }
        if (s.steps.size() != s.values.size()) {
            return Index("/scalars", i) + "/values: needs one value per step";
        }
        for (size_t j = 0; j < s.steps.size(); ++j) {
            if (s.steps[j] < kMinStep || s.steps[j] > kMaxStep) {
                return Index(Index("/scalars", i) + "/steps", j) + ": a step must fit in an int";
            }
        }
    }
    tags.clear();
    for (size_t i = 0; i < doc.histograms.size(); ++i) {
        if (!tags.insert(doc.histograms[i].tag).second) {
            return Index("/histograms", i) + "/tag: \"" + doc.histograms[i].tag + "\" repeats";
        }
    }
    return "";
}

}  // namespace

TrainingLogDocument ToTrainingLogDocument(const ImPlotMetricsSink& sink) {
    TrainingLogDocument doc;
    for (const auto& [tag, series] : sink.scalar_series()) {
        doc.scalars.push_back({tag, std::vector<int64_t>(series.steps.begin(), series.steps.end()), series.values});
    }
    for (const auto& [tag, values] : sink.latest_histograms()) {
        doc.histograms.push_back({tag, values});
    }
    auto by_tag = [](const auto& a, const auto& b) { return a.tag < b.tag; };
    std::sort(doc.scalars.begin(), doc.scalars.end(), by_tag);
    std::sort(doc.histograms.begin(), doc.histograms.end(), by_tag);
    return doc;
}

void ReplayTrainingLog(const TrainingLogDocument& doc, MetricsSink& sink) {
    if (std::string problem = TrainingLogProblem(doc); !problem.empty()) {
        Invalid("training_log", problem);
    }
    for (const auto& s : doc.scalars) {
        for (size_t i = 0; i < s.steps.size(); ++i) {
            sink.log_scalar(s.tag, s.values[i], static_cast<int>(s.steps[i]));
        }
    }
    CPUBackend backend;
    for (const auto& h : doc.histograms) {
        sink.log_histogram(h.tag, Tensor(Shape({static_cast<int64_t>(h.values.size())}), &backend, h.values), 0);
    }
}

std::string ToJson(const TrainingLogDocument& doc) {
    if (std::string problem = TrainingLogProblem(doc); !problem.empty()) {
        Invalid("training_log", problem);
    }
    DocWriter w("training_log");
    JsonValue scalars{JsonValue::Array{}};
    for (size_t i = 0; i < doc.scalars.size(); ++i) {
        const auto& s = doc.scalars[i];
        JsonValue series{JsonValue::Object{}};
        series.add("tag", s.tag);
        series.add("steps", Ints(s.steps));
        JsonValue values{JsonValue::Array{}};
        for (size_t j = 0; j < s.values.size(); ++j) {
            values.push_back(w.num_double(s.values[j], Index(Index("/scalars", i) + "/values", j)));
        }
        series.add("values", std::move(values));
        scalars.push_back(std::move(series));
    }
    JsonValue histograms{JsonValue::Array{}};
    for (size_t i = 0; i < doc.histograms.size(); ++i) {
        JsonValue h{JsonValue::Object{}};
        h.add("tag", doc.histograms[i].tag);
        h.add("values", w.floats(doc.histograms[i].values, Index("/histograms", i) + "/values"));
        histograms.push_back(std::move(h));
    }
    w.root().add("scalars", std::move(scalars));
    w.root().add("histograms", std::move(histograms));
    return w.finish();
}

TrainingLogDocument ParseTrainingLogDocument(std::string_view json) {
    DocReader r(json, "training_log");
    const JsonValue& root = r.root();
    TrainingLogDocument doc;
    const JsonValue::Array& scalars = r.array(r.member(root, "", "scalars"), "/scalars");
    for (size_t i = 0; i < scalars.size(); ++i) {
        const std::string p = Index("/scalars", i);
        r.object(scalars[i], p);
        TrainingLogDocument::Series s;
        s.tag = r.string(r.member(scalars[i], p, "tag"), p + "/tag");
        s.steps = r.ints(r.member(scalars[i], p, "steps"), p + "/steps", kMinStep, kMaxStep);
        const JsonValue::Array& values = r.array(r.member(scalars[i], p, "values"), p + "/values");
        for (size_t j = 0; j < values.size(); ++j) {
            s.values.push_back(r.number_double(values[j], Index(p + "/values", j)));
        }
        doc.scalars.push_back(std::move(s));
    }
    const JsonValue::Array& histograms = r.array(r.member(root, "", "histograms"), "/histograms");
    for (size_t i = 0; i < histograms.size(); ++i) {
        const std::string p = Index("/histograms", i);
        r.object(histograms[i], p);
        TrainingLogDocument::Histogram h;
        h.tag = r.string(r.member(histograms[i], p, "tag"), p + "/tag");
        h.values = r.floats(r.member(histograms[i], p, "values"), p + "/values");
        doc.histograms.push_back(std::move(h));
    }
    if (std::string problem = TrainingLogProblem(doc); !problem.empty()) {
        size_t colon = problem.find(':');
        r.fail(problem.substr(0, colon), problem.substr(colon + 2));
    }
    r.finish();
    return doc;
}

// ---- feature dashboard ---------------------------------------------------------------------

namespace {

std::string FeatureDashboardProblem(const FeatureDashboardDocument& doc) {
    bool both_empty = doc.histogram_edges.empty() && doc.histogram_counts.empty();
    if (!both_empty && doc.histogram_edges.size() != doc.histogram_counts.size() + 1) {
        return "/histogram_edges: needs one more edge than counts, or both empty";
    }
    for (size_t i = 0; i < doc.histogram_counts.size(); ++i) {
        if (doc.histogram_counts[i] < 0) {
            return Index("/histogram_counts", i) + ": a count can't be negative";
        }
    }
    for (size_t i = 0; i < doc.top_examples.size(); ++i) {
        const auto& e = doc.top_examples[i];
        if (!e.tokens.empty() && e.tokens.size() != e.activations.size()) {
            return Index("/top_examples", i) + "/activations: needs one activation per token";
        }
    }
    return "";
}

}  // namespace

std::string ToJson(const FeatureDashboardDocument& doc) {
    if (std::string problem = FeatureDashboardProblem(doc); !problem.empty()) {
        Invalid("feature_dashboard", problem);
    }
    DocWriter w("feature_dashboard");
    w.root().add("source", doc.source);
    w.root().add("feature_index", JsonValue(doc.feature_index));
    w.root().add("activation_density", w.num(doc.activation_density, "/activation_density"));
    w.root().add("max_activation", w.num(doc.max_activation, "/max_activation"));
    w.root().add("histogram_edges", w.floats(doc.histogram_edges, "/histogram_edges"));
    w.root().add("histogram_counts", Ints(doc.histogram_counts));
    JsonValue examples{JsonValue::Array{}};
    for (size_t i = 0; i < doc.top_examples.size(); ++i) {
        const auto& e = doc.top_examples[i];
        JsonValue example{JsonValue::Object{}};
        example.add("label", e.label);
        example.add("tokens", Strings(e.tokens));
        example.add("activations", w.floats(e.activations, Index("/top_examples", i) + "/activations"));
        examples.push_back(std::move(example));
    }
    w.root().add("top_examples", std::move(examples));
    return w.finish();
}

FeatureDashboardDocument ParseFeatureDashboardDocument(std::string_view json) {
    DocReader r(json, "feature_dashboard");
    const JsonValue& root = r.root();
    FeatureDashboardDocument doc;
    doc.source = r.string(r.member(root, "", "source"), "/source");
    doc.feature_index = r.integer(r.member(root, "", "feature_index"), "/feature_index", 0, kMaxIndex);
    doc.activation_density = r.number(r.member(root, "", "activation_density"), "/activation_density");
    doc.max_activation = r.number(r.member(root, "", "max_activation"), "/max_activation");
    doc.histogram_edges = r.floats(r.member(root, "", "histogram_edges"), "/histogram_edges");
    doc.histogram_counts = r.ints(r.member(root, "", "histogram_counts"), "/histogram_counts", 0, kMaxIndex);
    const JsonValue::Array& examples = r.array(r.member(root, "", "top_examples"), "/top_examples");
    for (size_t i = 0; i < examples.size(); ++i) {
        const std::string p = Index("/top_examples", i);
        r.object(examples[i], p);
        FeatureDashboardDocument::Example e;
        e.label = r.string(r.member(examples[i], p, "label"), p + "/label");
        e.tokens = r.strings(r.member(examples[i], p, "tokens"), p + "/tokens");
        e.activations = r.floats(r.member(examples[i], p, "activations"), p + "/activations");
        doc.top_examples.push_back(std::move(e));
    }
    if (std::string problem = FeatureDashboardProblem(doc); !problem.empty()) {
        size_t colon = problem.find(':');
        r.fail(problem.substr(0, colon), problem.substr(colon + 2));
    }
    r.finish();
    return doc;
}

}  // namespace pulsatrix
