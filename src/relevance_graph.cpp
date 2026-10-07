#include "pulsatrix/relevance_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <unordered_set>

namespace pulsatrix {

namespace {

/** @brief Sums a (1, n, d) relevance over d: one value per position. */
std::vector<double> PerPosition(const std::vector<float>& r, int64_t n, int64_t d) {
    std::vector<double> out(static_cast<size_t>(n), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t k = 0; k < d; ++k) out[static_cast<size_t>(i)] += r[static_cast<size_t>(i * d + k)];
    }
    return out;
}

std::string Quoted(const std::string& token) { return "\xE2\x80\x9C" + token + "\xE2\x80\x9D"; }

}  // namespace

AttributionGraph BuildRelevanceGraph(CausalLM& model, DeviceBackend* backend, const std::vector<int64_t>& ids, int64_t target,
                                     const RelevanceGraphOptions& options) {
    const auto n = static_cast<int64_t>(ids.size());
    const int64_t V = model.config().vocab_size;
    const int64_t d = model.config().hidden_size;
    const int64_t layers = model.num_layers();
    if (n == 0) throw std::invalid_argument("BuildRelevanceGraph: the prompt is empty");
    if (n > options.max_tokens) {
        throw std::invalid_argument("BuildRelevanceGraph: the prompt has " + std::to_string(n) + " tokens, more than max_tokens (" +
                                    std::to_string(options.max_tokens) + ")");
    }
    if (target < 0 || target >= V) throw std::invalid_argument("BuildRelevanceGraph: the target is outside the vocabulary");
    if (!options.prompt_tokens.empty() && static_cast<int64_t>(options.prompt_tokens.size()) != n) {
        throw std::invalid_argument("BuildRelevanceGraph: needs one prompt token string per id");
    }
    const DeviceType device = model.compute_device().value_or(DeviceType::Cpu);

    std::vector<float> id_values(ids.begin(), ids.end());
    const Tensor logits = model.forward(Tensor(Shape({1, n}), backend, id_values, device));
    const std::vector<float> all = logits.to_host_vector();
    const float* last = all.data() + (n - 1) * V;
    const float max_logit = *std::max_element(last, last + V);
    double z = 0;
    for (int64_t v = 0; v < V; ++v) z += std::exp(static_cast<double>(last[v] - max_logit));
    const double probability = std::exp(static_cast<double>(last[target] - max_logit)) / z;

    std::vector<float> seed(all.size(), 0.0f);
    seed[static_cast<size_t>((n - 1) * V + target)] = last[target];
    const std::vector<Tensor> boundaries = model.propagate_relevance_by_layer(Tensor(logits.shape(), backend, seed, device), options.config);
    std::vector<std::vector<float>> host;  // boundary b: (1, n, d)
    std::vector<std::vector<double>> relevance;  // boundary b: per position
    for (const Tensor& b : boundaries) {
        host.push_back(b.to_host_vector());
        relevance.push_back(PerPosition(host.back(), n, d));
    }

    AttributionGraph g;
    g.slug = options.slug;
    g.scan = options.scan;
    g.prompt = options.prompt;
    g.node_threshold = options.node_threshold;
    for (int64_t i = 0; i < n; ++i) {
        g.prompt_tokens.push_back(options.prompt_tokens.empty() ? std::to_string(ids[static_cast<size_t>(i)])
                                                                : options.prompt_tokens[static_cast<size_t>(i)]);
    }
    g.description = "AttnLRP relevance for " + Quoted(options.target_text) +
                    " through the residual stream: one node per layer and token, links carry relevance between adjacent layers";

    // Node ids by boundary and position. Boundary 0 is the embeddings; boundary b > 0 is block b-1's output.
    auto node_id = [&](int64_t b, int64_t pos) {
        if (b == 0) return "E_" + std::to_string(ids[static_cast<size_t>(pos)]) + "_" + std::to_string(pos);
        return std::to_string(b - 1) + "_" + std::to_string(pos) + "_" + std::to_string(pos);
    };
    std::vector<double> scores;
    for (int64_t b = 0; b <= layers; ++b) {
        for (int64_t pos = 0; pos < n; ++pos) {
            AttributionGraph::Node node;
            node.node_id = node_id(b, pos);
            node.feature = pos;
            node.ctx_idx = pos;
            node.activation = relevance[static_cast<size_t>(b)][static_cast<size_t>(pos)];
            if (b == 0) {
                node.layer = "E";
                node.feature_type = "embedding";
                node.js_node_id = "E_" + std::to_string(ids[static_cast<size_t>(pos)]) + "-" + std::to_string(pos);
            } else {
                node.layer = std::to_string(b - 1);
                node.feature_type = "residual stream";
                node.js_node_id = node.layer + "_" + std::to_string(pos) + "-0";
                node.clerp = "L" + node.layer + " " + Quoted(g.prompt_tokens[static_cast<size_t>(pos)]);
            }
            scores.push_back(*node.activation);
            g.nodes.push_back(std::move(node));
        }
    }
    AttributionGraph::Node out;
    out.layer = std::to_string(layers + 1);
    out.node_id = out.layer + "_" + std::to_string(target) + "_" + std::to_string(n - 1);
    out.feature = target;
    out.ctx_idx = n - 1;
    out.feature_type = "logit";
    out.js_node_id = "L_" + std::to_string(target) + "-" + std::to_string(n - 1);
    char p[32];
    std::snprintf(p, sizeof(p), "%.3f", probability);
    out.clerp = "Output \"" + options.target_text + "\" (p=" + p + ")";
    out.token_prob = probability;
    out.is_target_logit = true;
    out.activation = last[target];
    scores.push_back(0.0);
    g.nodes.push_back(out);

    // Links: block b-1 run with relevance only at position j, summed over d at every input position.
    std::vector<float> masked(host[0].size());
    for (int64_t b = 1; b <= layers; ++b) {
        const std::vector<float>& r_out = host[static_cast<size_t>(b)];
        for (int64_t j = 0; j < n; ++j) {
            if (relevance[static_cast<size_t>(b)][static_cast<size_t>(j)] == 0.0) {
                bool any = false;
                for (int64_t k = 0; k < d && !any; ++k) any = r_out[static_cast<size_t>(j * d + k)] != 0.0f;
                if (!any) continue;  // nothing to pass back from j
            }
            std::fill(masked.begin(), masked.end(), 0.0f);
            std::copy_n(r_out.begin() + j * d, d, masked.begin() + j * d);
            const Tensor r_in = model.layer(b - 1).propagate_relevance(Tensor(Shape({1, n, d}), backend, masked, device), options.config);
            const std::vector<double> flow = PerPosition(r_in.to_host_vector(), n, d);
            for (int64_t i = 0; i < n; ++i) {
                const double w = flow[static_cast<size_t>(i)];
                if (w != 0.0 && std::isfinite(w)) g.links.push_back({node_id(b - 1, i), node_id(b, j), w});
            }
        }
    }
    g.links.push_back({node_id(layers, n - 1), out.node_id, relevance[static_cast<size_t>(layers)][static_cast<size_t>(n - 1)]});

    ComputeInfluence(g, scores);
    if (options.edge_threshold < 1.0) PruneLinks(g, options.edge_threshold);
    // Drop residual nodes left with no links, as circuit-tracer does; embeddings and the output stay.
    std::unordered_set<std::string> linked;
    for (const auto& l : g.links) {
        linked.insert(l.source);
        linked.insert(l.target);
    }
    g.nodes.erase(std::remove_if(g.nodes.begin(), g.nodes.end(),
                                 [&](const AttributionGraph::Node& node) {
                                     return node.feature_type == "residual stream" && !linked.count(node.node_id);
                                 }),
                  g.nodes.end());
    return g;
}

}  // namespace pulsatrix
