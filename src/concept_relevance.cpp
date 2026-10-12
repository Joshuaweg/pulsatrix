#include "pulsatrix/concept_relevance.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace pulsatrix {
namespace {

void CheckLayer(const ExplainerContext& ctx, NodeId layer, const char* function) {
    if (layer < 1 || layer > ctx.modules().size()) {
        throw std::invalid_argument(std::string(function) + ": layer " + std::to_string(layer) + " must be between 1 and " +
                                    std::to_string(ctx.modules().size()) + " (module i's output is node i + 1)");
    }
}

// (rows, channels, positions per channel) of a layer's tensor.
struct Layout {
    int64_t rows = 0;
    int64_t channels = 0;
    int64_t positions = 0;
};

Layout LayoutOf(const Tensor& t, const char* function) {
    if (t.rank() < 2) throw std::invalid_argument(std::string(function) + ": the layer must have shape (N, C, ...)");
    Layout l;
    l.rows = t.shape().dim(0);
    l.channels = t.shape().dim(1);
    l.positions = l.rows * l.channels == 0 ? 0 : static_cast<int64_t>(t.numel()) / (l.rows * l.channels);
    return l;
}

// The relevance with every channel outside @p keep set to zero.
Tensor Mask(const Tensor& relevance, const std::vector<int64_t>& keep) {
    const Layout l = LayoutOf(relevance, "ConditionalHeatmaps");
    std::vector<bool> kept(static_cast<std::size_t>(l.channels), false);
    for (int64_t c : keep) {
        if (c < 0 || c >= l.channels) {
            throw std::invalid_argument("ConditionalHeatmaps: channel " + std::to_string(c) + " is outside the layer's " +
                                        std::to_string(l.channels));
        }
        kept[static_cast<std::size_t>(c)] = true;
    }
    std::vector<float> v = relevance.to_host_vector();
    for (int64_t n = 0; n < l.rows; ++n) {
        for (int64_t c = 0; c < l.channels; ++c) {
            if (kept[static_cast<std::size_t>(c)]) continue;
            const auto first = static_cast<std::size_t>((n * l.channels + c) * l.positions);
            std::fill(v.begin() + static_cast<std::ptrdiff_t>(first), v.begin() + static_cast<std::ptrdiff_t>(first + static_cast<std::size_t>(l.positions)), 0.0f);
        }
    }
    return Tensor(relevance.shape(), relevance.backend(), v, relevance.device());
}

std::string Describe(const ConceptCondition& condition) {
    std::string s;
    for (const ConceptMask& m : condition) {
        if (!s.empty()) s += ";";
        s += std::to_string(m.layer) + ":";
        for (std::size_t i = 0; i < m.channels.size(); ++i) s += (i ? "," : "") + std::to_string(m.channels[i]);
    }
    return s;
}

}  // namespace

Tensor LayerRelevance(const LRP& lrp, ExplainerContext& ctx, const Tensor& input, const LRPTarget& target, NodeId layer,
                      DeviceBackend* backend) {
    CheckLayer(ctx, layer, "LayerRelevance");
    const Tensor seed = lrp.seed_relevance(ctx, input, target, backend);
    return ctx.relevance_pass_through(seed, lrp.rule_configs(ctx), layer, ctx.modules().size());
}

std::vector<double> ChannelRelevance(const Tensor& layer_relevance) {
    const Layout l = LayoutOf(layer_relevance, "ChannelRelevance");
    const std::vector<float> v = layer_relevance.to_host_vector();
    std::vector<double> out(static_cast<std::size_t>(l.rows * l.channels), 0.0);
    for (int64_t rc = 0; rc < l.rows * l.channels; ++rc) {
        double s = 0;
        for (int64_t p = 0; p < l.positions; ++p) s += v[static_cast<std::size_t>(rc * l.positions + p)];
        out[static_cast<std::size_t>(rc)] = s;
    }
    return out;
}

std::vector<Attribution> ConditionalHeatmaps(const LRP& lrp, ExplainerContext& ctx, const Tensor& input, const LRPTarget& target,
                                             const std::vector<ConceptCondition>& conditions, DeviceBackend* backend) {
    for (const ConceptCondition& condition : conditions) {
        if (condition.empty()) throw std::invalid_argument("ConditionalHeatmaps: a condition has no masks");
        for (std::size_t i = 0; i < condition.size(); ++i) {
            CheckLayer(ctx, condition[i].layer, "ConditionalHeatmaps");
            for (std::size_t j = 0; j < i; ++j) {
                if (condition[j].layer == condition[i].layer) throw std::invalid_argument("ConditionalHeatmaps: a condition masks a layer twice");
            }
        }
    }
    // One forward pass and one seed; every condition propagates from them.
    const Tensor seed = lrp.seed_relevance(ctx, input, target, backend);
    const std::vector<LRPRuleConfig> configs = lrp.rule_configs(ctx);
    std::vector<Attribution> out;
    out.reserve(conditions.size());
    for (const ConceptCondition& condition : conditions) {
        std::vector<ConceptMask> masks = condition;
        std::sort(masks.begin(), masks.end(), [](const ConceptMask& a, const ConceptMask& b) { return a.layer > b.layer; });
        Tensor relevance = seed;
        std::size_t end = ctx.modules().size();
        for (const ConceptMask& m : masks) {
            relevance = Mask(ctx.relevance_pass_through(relevance, configs, m.layer, end), m.channels);
            end = m.layer;
        }
        relevance = ctx.relevance_pass_through(relevance, configs, 0, end);
        const std::vector<float> host = relevance.to_host_vector();
        const double sum = std::accumulate(host.begin(), host.end(), 0.0);
        out.push_back(Attribution{"crp", std::move(relevance), {{"condition", Describe(condition)}, {"relevance_in_sum", std::to_string(sum)}}});
    }
    return out;
}

std::vector<int64_t> ConceptReferenceSamples(const LRP& lrp, ExplainerContext& ctx, const Tensor& candidates, int64_t target, NodeId layer,
                                             int64_t channel, int64_t k, ReferenceMode mode, DeviceBackend* backend) {
    if (k < 1) throw std::invalid_argument("ConceptReferenceSamples: k must be at least 1");
    CheckLayer(ctx, layer, "ConceptReferenceSamples");
    std::vector<double> score;
    Layout l;
    if (mode == ReferenceMode::Relevance) {
        const Tensor relevance = LayerRelevance(lrp, ctx, candidates, LRPTarget{{target}}, layer, backend);
        l = LayoutOf(relevance, "ConceptReferenceSamples");
        if (channel < 0 || channel >= l.channels) throw std::invalid_argument("ConceptReferenceSamples: channel is outside the layer");
        const std::vector<double> per_channel = ChannelRelevance(relevance);
        for (int64_t n = 0; n < l.rows; ++n) score.push_back(per_channel[static_cast<std::size_t>(n * l.channels + channel)]);
    } else {
        (void)ctx.forward_pass(candidates);
        const Tensor& activation = ctx.activation(layer);
        l = LayoutOf(activation, "ConceptReferenceSamples");
        if (channel < 0 || channel >= l.channels) throw std::invalid_argument("ConceptReferenceSamples: channel is outside the layer");
        const std::vector<float> v = activation.to_host_vector();
        for (int64_t n = 0; n < l.rows; ++n) {
            double best = -std::numeric_limits<double>::infinity();
            for (int64_t p = 0; p < l.positions; ++p) best = std::max(best, static_cast<double>(v[static_cast<std::size_t>((n * l.channels + channel) * l.positions + p)]));
            score.push_back(best);
        }
    }
    std::vector<int64_t> order(score.size());
    std::iota(order.begin(), order.end(), int64_t{0});
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return score[static_cast<std::size_t>(a)] > score[static_cast<std::size_t>(b)]; });
    order.resize(std::min<std::size_t>(order.size(), static_cast<std::size_t>(k)));
    return order;
}

}  // namespace pulsatrix
