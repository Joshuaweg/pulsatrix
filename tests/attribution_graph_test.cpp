// VIZ-4: attribution graphs in Neuronpedia's and circuit-tracer's JSON.
#include "pulsatrix/viz/attribution_graph.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "pulsatrix/json.hpp"

namespace pulsatrix {
namespace {

std::string ReadFixture(const std::string& name) {
    std::ifstream in(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/viz/" + name, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

AttributionGraph Small() {
    AttributionGraph g;
    g.slug = "s";
    g.scan = "m";
    g.prompt = "a b";
    g.prompt_tokens = {"a", " b"};
    g.nodes.push_back({"E_5_0", 0, "E", 0, "embedding", "E_5-0", "", std::nullopt, std::nullopt, 0.0, false});
    g.nodes.push_back({"0_1_1", 1, "0", 1, "residual stream", "0_1-0", "L0", std::nullopt, 2.0, 0.0, false});
    g.nodes.push_back({"2_9_1", 9, "2", 1, "logit", "L_9-1", "Output \"x\" (p=0.500)", std::nullopt, std::nullopt, 0.5, true});
    g.links = {{"E_5_0", "0_1_1", 2.0}, {"0_1_1", "2_9_1", 2.0}};
    return g;
}

// A graph in circuit-tracer's own output format reads back, field for field.
TEST(AttributionGraph, ReadsCircuitTracerOutput) {
    const AttributionGraph g = ParseNeuronpediaGraph(ReadFixture("neuronpedia_graph.json"));
    EXPECT_EQ(g.slug, "tiny-example");
    EXPECT_EQ(g.scan, "gemma-2-2b");
    ASSERT_EQ(g.prompt_tokens.size(), 3u);
    EXPECT_EQ(g.prompt_tokens[2], " cat");
    ASSERT_EQ(g.nodes.size(), 4u);
    EXPECT_EQ(g.nodes[1].feature, 253);
    EXPECT_EQ(g.nodes[2].layer, "1");  // an integer layer reads as text
    EXPECT_FALSE(g.nodes[0].activation.has_value());
    EXPECT_TRUE(g.nodes[3].is_target_logit);
    EXPECT_DOUBLE_EQ(g.nodes[3].token_prob, 0.31);
    ASSERT_EQ(g.links.size(), 3u);
    EXPECT_DOUBLE_EQ(g.links[2].weight, -0.75);
}

TEST(AttributionGraph, RoundTrips) {
    const AttributionGraph g = ParseNeuronpediaGraph(ReadFixture("neuronpedia_graph.json"));
    const std::string json = ToNeuronpediaJson(g);
    EXPECT_EQ(ToNeuronpediaJson(ParseNeuronpediaGraph(json)), json);
}

TEST(AttributionGraph, WritesEveryFieldTheSchemaRequires) {
    const JsonValue root = ParseJson(ToNeuronpediaJson(Small()));
    for (const char* key : {"metadata", "qParams", "nodes", "links"}) EXPECT_NE(root.find(key), nullptr) << key;
    for (const char* key : {"slug", "scan", "prompt_tokens", "prompt"}) EXPECT_NE(root.find("metadata")->find(key), nullptr) << key;
    for (const auto& n : root.find("nodes")->as_array()) {
        for (const char* key : {"node_id", "feature", "layer", "ctx_idx", "feature_type", "jsNodeId", "clerp"}) {
            EXPECT_NE(n.find(key), nullptr) << key;
        }
    }
    for (const auto& l : root.find("links")->as_array()) {
        for (const char* key : {"source", "target", "weight"}) EXPECT_NE(l.find(key), nullptr) << key;
    }
}

TEST(AttributionGraph, RejectsWhatTheViewerCantShow) {
    auto broken = [](auto change) {
        AttributionGraph g = Small();
        change(g);
        return g;
    };
    EXPECT_THROW((void)ToNeuronpediaJson(broken([](AttributionGraph& g) { g.nodes[2].clerp = "Output x"; })), std::invalid_argument);
    EXPECT_THROW((void)ToNeuronpediaJson(broken([](AttributionGraph& g) { g.nodes[1].node_id = "E_5_0"; })), std::invalid_argument);
    EXPECT_THROW((void)ToNeuronpediaJson(broken([](AttributionGraph& g) { g.links[0].target = "nowhere"; })), std::invalid_argument);
    EXPECT_THROW((void)ToNeuronpediaJson(broken([](AttributionGraph& g) { g.links[0].weight = std::numeric_limits<double>::infinity(); })),
                 std::invalid_argument);
    EXPECT_THROW((void)ToNeuronpediaJson(broken([](AttributionGraph& g) { g.nodes[1].ctx_idx = 2; })), std::invalid_argument);
    EXPECT_THROW((void)ToNeuronpediaJson(broken([](AttributionGraph& g) { g.scan.clear(); })), std::invalid_argument);
    EXPECT_THROW((void)ToNeuronpediaJson(broken([](AttributionGraph& g) { g.prompt_tokens.clear(); })), std::invalid_argument);
}

TEST(AttributionGraph, InfluenceIsTheCumulativeShareLargestFirst) {
    AttributionGraph g = Small();
    for (int i = 0; i < 3; ++i) {
        g.nodes.push_back({"0_" + std::to_string(10 + i) + "_0", 10 + i, "0", 0, "residual stream", "0_" + std::to_string(10 + i) + "-0", "",
                           std::nullopt, std::nullopt, 0.0, false});
    }
    ComputeInfluence(g, {100.0, 5.0, 0.0, -3.0, 1.0, 1.0});
    EXPECT_FALSE(g.nodes[0].influence.has_value());  // embeddings and logits are always shown
    EXPECT_FALSE(g.nodes[2].influence.has_value());
    EXPECT_DOUBLE_EQ(*g.nodes[1].influence, 0.5);    // 5 of 10
    EXPECT_DOUBLE_EQ(*g.nodes[3].influence, 0.8);    // then |-3|
    EXPECT_DOUBLE_EQ(*g.nodes[4].influence, 0.9);    // ties keep their order
    EXPECT_DOUBLE_EQ(*g.nodes[5].influence, 1.0);
    EXPECT_THROW(ComputeInfluence(g, {1.0}), std::invalid_argument);
}

TEST(AttributionGraph, PruningKeepsTheFewestLinksReachingTheThreshold) {
    AttributionGraph g = Small();
    g.nodes.push_back({"0_3_0", 3, "0", 0, "residual stream", "0_3-0", "", std::nullopt, std::nullopt, 0.0, false});
    g.links = {{"E_5_0", "0_1_1", 6.0}, {"E_5_0", "0_3_0", -3.0}, {"0_3_0", "0_1_1", 1.0}, {"0_1_1", "2_9_1", 0.1}};
    PruneLinks(g, 0.55);  // 6 of 10.1 is enough
    ASSERT_EQ(g.links.size(), 2u);
    EXPECT_EQ(g.links[0].target, "0_1_1");
    EXPECT_EQ(g.links[1].target, "2_9_1");  // links into the output always stay
    EXPECT_THROW(PruneLinks(g, 0.0), std::invalid_argument);
    EXPECT_THROW(PruneLinks(g, 1.5), std::invalid_argument);
}

TEST(AttributionGraph, CircuitGraphBecomesAChainForTheViewer) {
    CircuitGraphDocument doc = ParseCircuitGraphDocument(ReadFixture("circuit_graph.v1.json"));
    doc.nodes[1].ablation_effect = 0.5f;  // the fixture's is infinite on purpose
    const AttributionGraph g = FromCircuitGraph(doc, "chain", "xor-mlp");
    ASSERT_EQ(g.nodes.size(), doc.nodes.size() + 1);
    EXPECT_EQ(g.nodes.front().feature_type, "embedding");
    EXPECT_EQ(g.nodes[1].clerp, "fc1");
    EXPECT_EQ(g.nodes[2].feature_type, "Activation");
    EXPECT_EQ(g.nodes.back().feature_type, "logit");
    EXPECT_EQ(g.nodes.back().clerp, "Output \"fc2\" (p=1.000)");
    EXPECT_EQ(g.links.size(), doc.edges.size() + 1);
    EXPECT_DOUBLE_EQ(*g.nodes[1].influence, 0.75);  // 1.5 of 2.0
    EXPECT_NO_THROW((void)ToNeuronpediaJson(g));
    EXPECT_THROW((void)FromCircuitGraph(ParseCircuitGraphDocument(ReadFixture("circuit_graph.v1.json")), "c", "m"), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
