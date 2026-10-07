#include "pulsatrix/viz/attribution_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "pulsatrix/json.hpp"

namespace pulsatrix {

namespace {

using Obj = JsonValue::Object;
using Arr = JsonValue::Array;

[[noreturn]] void Fail(const std::string& what) { throw std::invalid_argument("attribution graph: " + what); }

void CheckFinite(double v, const std::string& what) {
    if (!std::isfinite(v)) Fail(what + " is not finite");
}

void Validate(const AttributionGraph& g) {
    if (g.slug.empty()) Fail("the slug is empty");
    if (g.scan.empty()) Fail("the scan (model id) is empty");
    if (g.prompt_tokens.empty()) Fail("there are no prompt tokens");
    if (g.node_threshold) CheckFinite(*g.node_threshold, "the node threshold");
    std::unordered_set<std::string> ids;
    for (const auto& n : g.nodes) {
        if (n.node_id.empty() || n.layer.empty() || n.feature_type.empty() || n.js_node_id.empty()) {
            Fail("a node has an empty id, layer, type or viewer id");
        }
        if (!ids.insert(n.node_id).second) Fail("node id " + n.node_id + " repeats");
        if (n.ctx_idx < 0 || n.ctx_idx >= static_cast<int64_t>(g.prompt_tokens.size())) {
            Fail("node " + n.node_id + " is at a position outside the prompt");
        }
        if (n.feature_type == "logit" && n.clerp.find("(p=") == std::string::npos) {
            Fail("logit node " + n.node_id + " needs \"(p=PROBABILITY)\" in its label; the viewer reads it");
        }
        if (n.influence) CheckFinite(*n.influence, "node " + n.node_id + "'s influence");
        if (n.activation) CheckFinite(*n.activation, "node " + n.node_id + "'s activation");
        CheckFinite(n.token_prob, "node " + n.node_id + "'s token probability");
    }
    for (const auto& l : g.links) {
        if (!ids.count(l.source) || !ids.count(l.target)) Fail("link " + l.source + " -> " + l.target + " names an unknown node");
        CheckFinite(l.weight, "link " + l.source + " -> " + l.target + "'s weight");
    }
}

const JsonValue& Required(const JsonValue& v, std::string_view key) {
    const JsonValue* found = v.find(key);
    if (found == nullptr) Fail("missing \"" + std::string(key) + "\"");
    return *found;
}

std::optional<double> OptionalNumber(const JsonValue& v, std::string_view key) {
    const JsonValue* found = v.find(key);
    if (found == nullptr || found->is_null()) return std::nullopt;
    return found->as_double();
}

std::string String(const JsonValue& v, std::string_view key, const std::string& fallback = "") {
    const JsonValue* found = v.find(key);
    return found == nullptr || found->is_null() ? fallback : found->as_string();
}

bool IsSpecial(const AttributionGraph::Node& n) { return n.feature_type == "embedding" || n.feature_type == "logit"; }

}  // namespace

std::string ToNeuronpediaJson(const AttributionGraph& g) {
    Validate(g);
    Arr tokens(g.prompt_tokens.begin(), g.prompt_tokens.end());
    Obj metadata{{"slug", g.slug}, {"scan", g.scan}, {"prompt_tokens", std::move(tokens)}, {"prompt", g.prompt}};
    if (g.node_threshold) metadata.emplace_back("node_threshold", *g.node_threshold);
    metadata.emplace_back("schema_version", 1);
    Obj info{{"generator", Obj{{"name", "pulsatrix"}, {"url", "https://github.com/Joshuaweg/pulsatrix"}}}};
    if (!g.description.empty()) info.insert(info.begin(), {"description", g.description});
    metadata.emplace_back("info", std::move(info));

    Arr nodes;
    nodes.reserve(g.nodes.size());
    for (const auto& n : g.nodes) {
        nodes.push_back(Obj{{"node_id", n.node_id},
                            {"feature", n.feature ? JsonValue(*n.feature) : JsonValue()},
                            {"layer", n.layer},
                            {"ctx_idx", n.ctx_idx},
                            {"feature_type", n.feature_type},
                            {"token_prob", n.token_prob},
                            {"is_target_logit", n.is_target_logit},
                            {"jsNodeId", n.js_node_id},
                            {"clerp", n.clerp},
                            {"influence", n.influence ? JsonValue(*n.influence) : JsonValue()},
                            {"activation", n.activation ? JsonValue(*n.activation) : JsonValue()}});
    }
    Arr links;
    links.reserve(g.links.size());
    for (const auto& l : g.links) links.push_back(Obj{{"source", l.source}, {"target", l.target}, {"weight", l.weight}});
    return WriteJson(Obj{{"metadata", std::move(metadata)},
                         {"qParams", Obj{{"pinnedIds", Arr{}}, {"supernodes", Arr{}}, {"linkType", "both"}, {"clickedId", ""}, {"sg_pos", ""}}},
                         {"nodes", std::move(nodes)},
                         {"links", std::move(links)}});
}

AttributionGraph ParseNeuronpediaGraph(std::string_view json) {
    const JsonValue root = ParseJson(json);
    const JsonValue& meta = Required(root, "metadata");
    AttributionGraph g;
    g.slug = Required(meta, "slug").as_string();
    g.scan = Required(meta, "scan").as_string();
    g.prompt = Required(meta, "prompt").as_string();
    for (const auto& t : Required(meta, "prompt_tokens").as_array()) g.prompt_tokens.push_back(t.as_string());
    g.node_threshold = OptionalNumber(meta, "node_threshold");
    if (const JsonValue* info = meta.find("info"); info != nullptr && !info->is_null()) g.description = String(*info, "description");
    for (const auto& v : Required(root, "nodes").as_array()) {
        AttributionGraph::Node n;
        n.node_id = Required(v, "node_id").as_string();
        if (const JsonValue& f = Required(v, "feature"); !f.is_null()) n.feature = f.as_int64();
        const JsonValue& layer = Required(v, "layer");
        n.layer = layer.type() == JsonValue::Type::String ? layer.as_string() : std::to_string(layer.as_int64());
        n.ctx_idx = Required(v, "ctx_idx").as_int64();
        n.feature_type = Required(v, "feature_type").as_string();
        n.js_node_id = Required(v, "jsNodeId").as_string();
        n.clerp = String(v, "clerp");
        n.influence = OptionalNumber(v, "influence");
        n.activation = OptionalNumber(v, "activation");
        n.token_prob = OptionalNumber(v, "token_prob").value_or(0.0);
        if (const JsonValue* t = v.find("is_target_logit"); t != nullptr && !t->is_null()) n.is_target_logit = t->as_bool();
        g.nodes.push_back(std::move(n));
    }
    for (const auto& v : Required(root, "links").as_array()) {
        g.links.push_back({Required(v, "source").as_string(), Required(v, "target").as_string(), Required(v, "weight").as_double()});
    }
    return g;
}

void ComputeInfluence(AttributionGraph& g, const std::vector<double>& scores) {
    if (scores.size() != g.nodes.size()) Fail("ComputeInfluence needs one score per node");
    std::vector<size_t> order;
    double total = 0;
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        CheckFinite(scores[i], "a score");
        if (IsSpecial(g.nodes[i])) continue;
        order.push_back(i);
        total += std::abs(scores[i]);
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return std::abs(scores[a]) > std::abs(scores[b]); });
    double running = 0;
    for (size_t i : order) {
        running += std::abs(scores[i]);
        g.nodes[i].influence = total > 0 ? std::min(1.0, running / total) : 1.0;
    }
}

void PruneLinks(AttributionGraph& g, double edge_threshold) {
    if (!(edge_threshold > 0 && edge_threshold <= 1)) Fail("the edge threshold must be in (0, 1]");
    std::unordered_set<std::string> logits;
    for (const auto& n : g.nodes) {
        if (n.feature_type == "logit") logits.insert(n.node_id);
    }
    std::vector<size_t> order(g.links.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return std::abs(g.links[a].weight) > std::abs(g.links[b].weight); });
    double total = 0;
    for (const auto& l : g.links) total += std::abs(l.weight);
    std::vector<bool> keep(g.links.size(), false);
    double running = 0;
    for (size_t i : order) {
        if (running >= edge_threshold * total && total > 0) break;
        keep[i] = true;
        running += std::abs(g.links[i].weight);
    }
    std::vector<AttributionGraph::Link> kept;
    for (size_t i = 0; i < g.links.size(); ++i) {
        if (keep[i] || logits.count(g.links[i].target)) kept.push_back(std::move(g.links[i]));
    }
    g.links = std::move(kept);
}

AttributionGraph FromCircuitGraph(const CircuitGraphDocument& doc, const std::string& slug, const std::string& scan) {
    if (doc.nodes.empty()) Fail("the circuit graph has no nodes");
    AttributionGraph g;
    g.slug = slug;
    g.scan = scan;
    // Two columns: the input, then the operations. With one position the viewer would hide every
    // node, since it drops features seen at more than 2/3 of the positions.
    g.prompt = "input network";
    g.prompt_tokens = {"input", "network"};
    g.node_threshold = 1.0;  // a chain is short: show every operation
    g.description = "A pulsatrix circuit graph: each operation in order, scored by how much zeroing it changes the output";
    g.nodes.push_back({"E_0_0", 0, "E", 0, "embedding", "E_0-0", "Input", std::nullopt, std::nullopt, 0.0, false});
    std::unordered_map<int64_t, std::string> ids;
    std::vector<double> scores{0.0};
    const size_t last = doc.nodes.size() - 1;
    const std::string logit_layer = std::to_string(last);
    for (size_t i = 0; i < doc.nodes.size(); ++i) {
        const auto& n = doc.nodes[i];
        CheckFinite(n.ablation_effect, "node " + std::to_string(n.id) + "'s ablation effect");
        const std::string name = n.label ? *n.label : n.op_type + " " + std::to_string(n.id);
        AttributionGraph::Node out;
        out.feature = n.id;
        out.ctx_idx = 1;
        if (i == last && i > 0) {
            out.layer = logit_layer;
            out.node_id = logit_layer + "_" + std::to_string(n.id) + "_1";
            out.feature_type = "logit";
            out.js_node_id = "L_" + std::to_string(n.id) + "-1";
            out.clerp = "Output \"" + name + "\" (p=1.000)";
            out.token_prob = 1.0;
            out.is_target_logit = true;
        } else {
            out.layer = std::to_string(i);
            out.node_id = out.layer + "_" + std::to_string(n.id) + "_1";
            out.feature_type = n.op_type;
            out.js_node_id = out.layer + "_" + std::to_string(n.id) + "-1";
            out.clerp = name;
            out.activation = n.ablation_effect;
        }
        ids[n.id] = out.node_id;
        g.nodes.push_back(std::move(out));
        scores.push_back(n.ablation_effect);
    }
    g.links.push_back({"E_0_0", g.nodes[1].node_id, 1.0});  // the input feeds the first operation
    for (const auto& e : doc.edges) {
        CheckFinite(e.weight, "an edge weight");
        if (!ids.count(e.from) || !ids.count(e.to)) Fail("an edge names a node that isn't in the graph");
        g.links.push_back({ids[e.from], ids[e.to], e.weight});
    }
    ComputeInfluence(g, scores);
    return g;
}

}  // namespace pulsatrix
