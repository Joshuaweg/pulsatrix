// VIZ-4: the AttnLRP attribution graph on the tiny Hugging Face models in tests/fixtures/hf_tiny.
#include "pulsatrix/relevance_graph.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

std::string Tiny(const std::string& name) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/" + name; }

class RelevanceGraphTest : public ::testing::TestWithParam<const char*> {
protected:
    void SetUp() override {
        model_ = LoadCausalLM(Tiny(GetParam()), &backend_);
        ids_ = {1, 17, 42, 5, 33, 3, 60};
        RelevanceGraphOptions o;
        o.edge_threshold = 1.0;  // every link, for the conservation checks
        o.target_text = "t";
        graph_ = BuildRelevanceGraph(*model_, &backend_, ids_, target_, o);
        for (const auto& n : graph_.nodes) nodes_[n.node_id] = &n;
    }
    double Relevance(const std::string& id) const {
        auto it = nodes_.find(id);
        return it == nodes_.end() ? 0.0 : it->second->activation.value_or(0.0);
    }
    CPUBackend backend_;
    std::unique_ptr<CausalLM> model_;
    std::vector<int64_t> ids_;
    int64_t target_ = 7;
    AttributionGraph graph_;
    std::map<std::string, const AttributionGraph::Node*> nodes_;
};

// The embedding nodes hold exactly the per-token relevance a plain LRP run gives.
TEST_P(RelevanceGraphTest, EmbeddingsMatchPropagateRelevance) {
    const int64_t n = static_cast<int64_t>(ids_.size()), V = model_->config().vocab_size;
    std::vector<float> ids(ids_.begin(), ids_.end());
    const Tensor logits = model_->forward(Tensor(Shape({1, n}), &backend_, ids));
    std::vector<float> seed(static_cast<size_t>(n * V), 0.0f);
    seed[static_cast<size_t>((n - 1) * V + target_)] = logits.to_host_vector()[static_cast<size_t>((n - 1) * V + target_)];
    const std::vector<float> tokens = model_->propagate_relevance(Tensor(logits.shape(), &backend_, seed), LxtAttnLrpConfig()).to_host_vector();
    ASSERT_EQ(tokens.size(), ids_.size());
    for (int64_t i = 0; i < n; ++i) {
        const double r = Relevance("E_" + std::to_string(ids_[static_cast<size_t>(i)]) + "_" + std::to_string(i));
        EXPECT_NEAR(r, tokens[static_cast<size_t>(i)], 1e-4 * (1.0 + std::abs(tokens[static_cast<size_t>(i)]))) << "position " << i;
    }
}

// Linearity: the links leaving a node sum to its relevance, so splitting each block's LRP run by
// output position loses nothing.
TEST_P(RelevanceGraphTest, LinksConserveRelevance) {
    std::map<std::string, double> out_sum, in_sum;
    for (const auto& l : graph_.links) {
        out_sum[l.source] += l.weight;
        in_sum[l.target] += l.weight;
    }
    double scale = 0;
    for (const auto& n : graph_.nodes) scale = std::max(scale, std::abs(n.activation.value_or(0.0)));
    int checked = 0;
    for (const auto& n : graph_.nodes) {
        if (n.feature_type == "logit") continue;
        const bool top = n.layer == std::to_string(model_->num_layers() - 1);
        if (!top || n.ctx_idx == static_cast<int64_t>(ids_.size()) - 1) {
            EXPECT_NEAR(out_sum[n.node_id], *n.activation, 1e-4 * scale) << n.node_id << " outgoing";
            ++checked;
        }
    }
    EXPECT_GT(checked, static_cast<int>(ids_.size()));
    // Not checked: that each layer's total equals the next one's. LRP through norms and biases
    // isn't conservative (on these random tiny models the totals differ by up to 2x), and LXT's
    // reference sums differ the same way.
}

TEST_P(RelevanceGraphTest, LinksJoinAdjacentLayersAndRespectCausality) {
    const auto layer_index = [&](const AttributionGraph::Node& n) { return n.layer == "E" ? -1 : std::stoi(n.layer); };
    ASSERT_FALSE(graph_.links.empty());
    for (const auto& l : graph_.links) {
        const auto& s = *nodes_.at(l.source);
        const auto& t = *nodes_.at(l.target);
        if (t.feature_type == "logit") {
            EXPECT_EQ(s.ctx_idx, static_cast<int64_t>(ids_.size()) - 1);
            continue;
        }
        EXPECT_EQ(layer_index(t), layer_index(s) + 1) << l.source << " -> " << l.target;
        EXPECT_LE(s.ctx_idx, t.ctx_idx) << "attention only looks back";
    }
}

TEST_P(RelevanceGraphTest, OutputNodeCarriesTheProbability) {
    const auto& out = graph_.nodes.back();
    ASSERT_EQ(out.feature_type, "logit");
    EXPECT_TRUE(out.is_target_logit);
    EXPECT_GT(out.token_prob, 0.0);
    EXPECT_LT(out.token_prob, 1.0);
    char p[32];
    std::snprintf(p, sizeof(p), "(p=%.3f)", out.token_prob);
    EXPECT_NE(out.clerp.find(p), std::string::npos) << out.clerp;
    EXPECT_NO_THROW((void)ToNeuronpediaJson(graph_));
}

TEST_P(RelevanceGraphTest, PruningKeepsTheStrongestLinks) {
    RelevanceGraphOptions o;
    o.edge_threshold = 0.5;
    o.target_text = "t";
    const AttributionGraph pruned = BuildRelevanceGraph(*model_, &backend_, ids_, target_, o);
    EXPECT_LT(pruned.links.size(), graph_.links.size());
    for (const auto& n : pruned.nodes) {
        if (n.feature_type != "residual stream") continue;
        bool linked = false;
        for (const auto& l : pruned.links) linked = linked || l.source == n.node_id || l.target == n.node_id;
        EXPECT_TRUE(linked) << n.node_id << " is left with no links";
    }
}

TEST_P(RelevanceGraphTest, RejectsBadInputs) {
    RelevanceGraphOptions o;
    EXPECT_THROW((void)BuildRelevanceGraph(*model_, &backend_, {}, 0, o), std::invalid_argument);
    EXPECT_THROW((void)BuildRelevanceGraph(*model_, &backend_, ids_, model_->config().vocab_size, o), std::invalid_argument);
    o.max_tokens = 3;
    EXPECT_THROW((void)BuildRelevanceGraph(*model_, &backend_, ids_, 0, o), std::invalid_argument);
    o.max_tokens = 256;
    o.prompt_tokens = {"a"};
    EXPECT_THROW((void)BuildRelevanceGraph(*model_, &backend_, ids_, 0, o), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(TinyModels, RelevanceGraphTest, ::testing::Values("llama", "qwen2", "qwen3", "gemma3"));

}  // namespace
}  // namespace pulsatrix
