#include "pulsatrix/viz/document.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

std::string ReadFixture(const std::string& name) {
    std::ifstream in(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/viz/" + name, std::ios::binary);
    EXPECT_TRUE(in) << "missing fixture " << name;
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool SameBits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

void ExpectSameFloats(const std::vector<float>& a, const std::vector<float>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_TRUE(SameBits(a[i], b[i]) || (std::isnan(a[i]) && std::isnan(b[i]))) << "index " << i << ": " << a[i] << " vs " << b[i];
    }
}

// Asserts the reader rejects @p json with a message that contains @p fragment (typically the
// JSON Pointer of the problem).
template <typename Parse>
void ExpectRejected(Parse parse, const std::string& json, const std::string& fragment) {
    try {
        (void)parse(json);
        ADD_FAILURE() << "accepted:\n" << json;
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << "message: " << e.what();
    }
}

constexpr float kInf = std::numeric_limits<float>::infinity();

// ---- golden files: the writer reproduces each fixture byte for byte, and the reader reads it
// back to the same struct.

AttributionDocument GoldenAttribution() {
    AttributionDocument doc;
    doc.method = "integrated_gradients";
    doc.shape = {2, 3};
    doc.values = {0.5f, -1.25f, 0.0f, 3.0f, 0.1f, -0.001f};
    doc.metadata = {{"steps", "50"}, {"baseline", "zero"}};
    return doc;
}

TEST(VizDocumentGoldenTest, Attribution) {
    std::string golden = ReadFixture("attribution.v1.json");
    EXPECT_EQ(ToJson(GoldenAttribution()), golden);
    AttributionDocument back = ParseAttributionDocument(golden);
    EXPECT_EQ(back.method, "integrated_gradients");
    EXPECT_EQ(back.shape, (std::vector<int64_t>{2, 3}));
    ExpectSameFloats(back.values, GoldenAttribution().values);
    EXPECT_EQ(back.metadata, GoldenAttribution().metadata);
}

TEST(VizDocumentGoldenTest, NonFiniteValuesBecomeNullPlusAPointer) {
    AttributionDocument doc;
    doc.method = "saliency";
    doc.shape = {4};
    doc.values = {1.0f, std::numeric_limits<float>::quiet_NaN(), kInf, -kInf};
    std::string golden = ReadFixture("attribution_nonfinite.v1.json");
    EXPECT_EQ(ToJson(doc), golden);
    AttributionDocument back = ParseAttributionDocument(golden);
    ASSERT_EQ(back.values.size(), 4u);
    EXPECT_EQ(back.values[0], 1.0f);
    EXPECT_TRUE(std::isnan(back.values[1]));
    EXPECT_EQ(back.values[2], kInf);
    EXPECT_EQ(back.values[3], -kInf);
}

TEST(VizDocumentGoldenTest, Heatmap) {
    HeatmapDocument doc;
    doc.title = "Attention, head 0";
    doc.rows = 2;
    doc.cols = 2;
    doc.values = {0.75f, 0.25f, 0.0f, 1.0f};
    doc.row_labels = {"the", "cat"};
    doc.col_labels = {"the", "cat"};
    std::string golden = ReadFixture("heatmap.v1.json");
    EXPECT_EQ(ToJson(doc), golden);
    HeatmapDocument back = ParseHeatmapDocument(golden);
    EXPECT_EQ(back.title, doc.title);
    EXPECT_EQ(back.rows, 2);
    EXPECT_EQ(back.cols, 2);
    ExpectSameFloats(back.values, doc.values);
    EXPECT_EQ(back.row_labels, doc.row_labels);
    EXPECT_EQ(back.col_labels, doc.col_labels);
}

TEST(VizDocumentGoldenTest, TokenRelevance) {
    TokenRelevanceDocument doc;
    doc.method = "attn_lrp";
    doc.tokens = {"The", " caf\xC3\xA9", " \"quote\""};
    doc.relevance = {0.125f, -0.5f, 2.0f};
    doc.target = " is";
    std::string golden = ReadFixture("token_relevance.v1.json");
    EXPECT_EQ(ToJson(doc), golden);
    TokenRelevanceDocument back = ParseTokenRelevanceDocument(golden);
    EXPECT_EQ(back.method, doc.method);
    EXPECT_EQ(back.tokens, doc.tokens);
    ExpectSameFloats(back.relevance, doc.relevance);
    EXPECT_EQ(back.target, doc.target);
}

CircuitGraphDocument GoldenCircuit() {
    CircuitGraphDocument doc;
    doc.nodes = {{0, "Linear", "fc1", 1.5f}, {1, "Activation", std::nullopt, kInf}, {2, "Linear", "fc2", 0.0f}};
    doc.edges = {{0, 1, 1.5f}, {1, 2, 0.25f}};
    return doc;
}

TEST(VizDocumentGoldenTest, CircuitGraph) {
    std::string golden = ReadFixture("circuit_graph.v1.json");
    EXPECT_EQ(ToJson(GoldenCircuit()), golden);
    CircuitGraphDocument back = ParseCircuitGraphDocument(golden);
    ASSERT_EQ(back.nodes.size(), 3u);
    EXPECT_EQ(back.nodes[0].label, std::optional<std::string>("fc1"));
    EXPECT_EQ(back.nodes[1].label, std::nullopt);
    EXPECT_EQ(back.nodes[1].op_type, "Activation");
    EXPECT_EQ(back.nodes[1].ablation_effect, kInf);
    ASSERT_EQ(back.edges.size(), 2u);
    EXPECT_EQ(back.edges[1].from, 1);
    EXPECT_EQ(back.edges[1].to, 2);
    EXPECT_EQ(back.edges[1].weight, 0.25f);
}

TEST(VizDocumentGoldenTest, TrainingLog) {
    TrainingLogDocument doc;
    doc.scalars = {{"loss", {0, 1, 2}, {0.9, 0.5, 0.1}}, {"lr", {0, 2}, {0.001, 0.0005}}};
    doc.histograms = {{"weights", {-0.5f, 0.0f, 0.5f}}};
    std::string golden = ReadFixture("training_log.v1.json");
    EXPECT_EQ(ToJson(doc), golden);
    TrainingLogDocument back = ParseTrainingLogDocument(golden);
    ASSERT_EQ(back.scalars.size(), 2u);
    EXPECT_EQ(back.scalars[1].tag, "lr");
    EXPECT_EQ(back.scalars[1].steps, (std::vector<int64_t>{0, 2}));
    EXPECT_EQ(back.scalars[1].values, (std::vector<double>{0.001, 0.0005}));
    ASSERT_EQ(back.histograms.size(), 1u);
    ExpectSameFloats(back.histograms[0].values, doc.histograms[0].values);
}

TEST(VizDocumentGoldenTest, FeatureDashboard) {
    FeatureDashboardDocument doc;
    doc.source = "sae.blocks.3";
    doc.feature_index = 42;
    doc.activation_density = 0.015f;
    doc.max_activation = 7.5f;
    doc.histogram_edges = {0.0f, 2.5f, 5.0f, 7.5f};
    doc.histogram_counts = {10, 4, 1};
    doc.top_examples = {{"doc 17", {"Paris", " is"}, {7.5f, 0.25f}}, {"image 3", {}, {6.0f}}};
    std::string golden = ReadFixture("feature_dashboard.v1.json");
    EXPECT_EQ(ToJson(doc), golden);
    FeatureDashboardDocument back = ParseFeatureDashboardDocument(golden);
    EXPECT_EQ(back.source, doc.source);
    EXPECT_EQ(back.feature_index, 42);
    EXPECT_EQ(back.activation_density, 0.015f);
    EXPECT_EQ(back.histogram_counts, doc.histogram_counts);
    ASSERT_EQ(back.top_examples.size(), 2u);
    EXPECT_EQ(back.top_examples[0].tokens, doc.top_examples[0].tokens);
    EXPECT_TRUE(back.top_examples[1].tokens.empty());
    ExpectSameFloats(back.top_examples[1].activations, {6.0f});
}

TEST(VizDocumentGoldenTest, PartialDependence) {
    PartialDependenceDocument doc;
    doc.feature = "age";
    doc.target = "risk";
    doc.grid = {0.0f, 0.5f, 1.0f};
    doc.partial_dependence = {0.25f, 0.5f, 1.0f};
    doc.num_instances = 2;
    doc.ice = {0.0f, 0.5f, 1.0f, 0.5f, 0.5f, 1.0f};
    doc.feature_values = {0.25f, 0.75f};
    std::string golden = ReadFixture("partial_dependence.v1.json");
    EXPECT_EQ(ToJson(doc), golden);
    PartialDependenceDocument back = ParsePartialDependenceDocument(golden);
    EXPECT_EQ(back.feature, "age");
    EXPECT_EQ(back.num_instances, 2);
    ExpectSameFloats(back.ice, doc.ice);
    ExpectSameFloats(back.feature_values, doc.feature_values);
}

TEST(VizDocumentGoldenTest, Sensitivity) {
    SensitivityDocument doc;
    doc.target = "risk";
    doc.output = 0.5f;
    doc.features = {{"age", 40.0f, 30.0f, 50.0f, 0.25f, 1.0f}, {"dose", 2.0f, 1.0f, 3.0f, 0.625f, 0.375f}};
    std::string golden = ReadFixture("sensitivity.v1.json");
    EXPECT_EQ(ToJson(doc), golden);
    SensitivityDocument back = ParseSensitivityDocument(golden);
    ASSERT_EQ(back.features.size(), 2u);
    EXPECT_EQ(back.features[1].name, "dose");
    EXPECT_EQ(back.features[1].output_low, 0.625f);
    EXPECT_EQ(back.output, 0.5f);
}

TEST(VizDocumentConversionTest, LocalSensitivityNamesItsFeatures) {
    LocalSensitivityResult r;
    r.output = 1.0f;
    r.features = {{2, 0.5f, 0.0f, 1.0f, 0.75f, 1.25f}, {0, 3.0f, 2.0f, 4.0f, 1.0f, 1.0f}};
    SensitivityDocument doc = ToSensitivityDocument(r, {"a", "b", "c"}, "y");
    EXPECT_EQ(doc.features[0].name, "c");
    EXPECT_EQ(doc.features[1].name, "a");
    EXPECT_EQ(ToSensitivityDocument(r).features[0].name, "feature_2");
}

TEST(VizDocumentGoldenTest, Counterfactual) {
    CounterfactualDocument doc;
    doc.target = "class approved";
    doc.valid = true;
    doc.output_before = -0.5f;
    doc.output_after = 0.125f;
    doc.features = {{"income", 40.0f, 52.0f, 8.0f}, {"age", 35.0f, 35.0f, 10.0f}, {"debt", 0.5f, 0.25f, 0.25f}};
    std::string golden = ReadFixture("counterfactual.v1.json");
    EXPECT_EQ(ToJson(doc), golden);
    CounterfactualDocument back = ParseCounterfactualDocument(golden);
    EXPECT_TRUE(back.valid);
    ASSERT_EQ(back.features.size(), 3u);
    EXPECT_EQ(back.features[2].counterfactual, 0.25f);
    std::string bad = golden;
    bad.replace(bad.find("\"scale\": 8"), 10, "\"scale\": 0");
    ExpectRejected(ParseCounterfactualDocument, bad, "/features/0/scale");
    bad = golden;
    bad.replace(bad.find("true"), 4, "1");
    ExpectRejected(ParseCounterfactualDocument, bad, "/valid");
}

// ---- conversions from and to the in-memory types the widgets draw today.

TEST(VizDocumentConversionTest, AttributionRoundTripsThroughTensor) {
    CPUBackend backend;
    Attribution attr{"lrp_epsilon", Tensor(Shape({1, 2, 2}), &backend, {0.25f, -0.5f, 1e-30f, 3.0f}),
                     {{"rule", "epsilon"}, {"epsilon", "1e-6"}}};
    AttributionDocument doc = ToAttributionDocument(attr);
    EXPECT_EQ(doc.shape, (std::vector<int64_t>{1, 2, 2}));
    Attribution back = ToAttribution(ParseAttributionDocument(ToJson(doc)), &backend);
    EXPECT_EQ(back.method, attr.method);
    EXPECT_EQ(back.values.shape(), attr.values.shape());
    ExpectSameFloats(back.values.to_host_vector(), attr.values.to_host_vector());
    EXPECT_EQ(back.metadata, attr.metadata);
}

TEST(VizDocumentConversionTest, HeatmapGridRoundTrips) {
    HeatmapGrid grid{{0.0f, 1.0f, -2.0f, 3.0f, 4.0f, 5.0f}, 2, 3};
    HeatmapDocument doc = ToHeatmapDocument(grid, "saliency");
    EXPECT_EQ(doc.title, "saliency");
    HeatmapGrid back = ToHeatmapGrid(ParseHeatmapDocument(ToJson(doc)));
    EXPECT_EQ(back.rows, 2);
    EXPECT_EQ(back.cols, 3);
    ExpectSameFloats(back.values, grid.values);
}

TEST(VizDocumentConversionTest, CircuitGraphRoundTrips) {
    CircuitGraph graph({{0, OpType::Linear, "fc1", 2.0f}, {1, OpType::Attention, std::nullopt, 0.5f}},
                       {{0, 1, 2.0f}});
    CircuitGraph back = ToCircuitGraph(ParseCircuitGraphDocument(ToJson(ToCircuitGraphDocument(graph))));
    ASSERT_EQ(back.nodes().size(), 2u);
    EXPECT_EQ(back.nodes()[1].op_type, OpType::Attention);
    EXPECT_EQ(back.nodes()[1].label, std::nullopt);
    EXPECT_EQ(back.nodes()[0].ablation_effect, 2.0f);
    ASSERT_EQ(back.edges().size(), 1u);
    EXPECT_EQ(back.edges()[0].to, 1u);
}

TEST(VizDocumentConversionTest, EveryOpTypeHasAName) {
    for (OpType t : {OpType::Linear, OpType::Conv, OpType::Activation, OpType::Elementwise, OpType::Reduction,
                     OpType::Normalization, OpType::Pooling, OpType::Embedding, OpType::Composite,
                     OpType::Recurrent, OpType::Attention}) {
        EXPECT_EQ(OpTypeFromName(OpTypeName(t)), t);
    }
    EXPECT_THROW((void)OpTypeFromName("linear"), std::invalid_argument);
}

TEST(VizDocumentConversionTest, TrainingLogRoundTripsThroughAMetricsSink) {
    CPUBackend backend;
    ImPlotMetricsSink sink;
    sink.log_scalar("loss", 1.0, 0);
    sink.log_scalar("loss", 0.5, 1);
    sink.log_scalar("acc", 0.25, 1);
    sink.log_histogram("w", Tensor(Shape({3}), &backend, {1.0f, 2.0f, 3.0f}), 1);
    TrainingLogDocument doc = ToTrainingLogDocument(sink);
    ASSERT_EQ(doc.scalars.size(), 2u);
    EXPECT_EQ(doc.scalars[0].tag, "acc");  // sorted, not hash-map order

    ImPlotMetricsSink replayed;
    ReplayTrainingLog(ParseTrainingLogDocument(ToJson(doc)), replayed);
    ASSERT_EQ(replayed.scalar_series().size(), 2u);
    EXPECT_EQ(replayed.scalar_series().at("loss").steps, (std::vector<int>{0, 1}));
    EXPECT_EQ(replayed.scalar_series().at("loss").values, (std::vector<double>{1.0, 0.5}));
    EXPECT_EQ(replayed.latest_histograms().at("w"), (std::vector<float>{1.0f, 2.0f, 3.0f}));
}

// ---- reading: version and kind checks, and errors that name where the problem is.

TEST(VizDocumentReadTest, KindComesFromTheSchema) {
    EXPECT_EQ(VizDocumentKind(ReadFixture("heatmap.v1.json")), "heatmap");
    EXPECT_EQ(VizDocumentKind(ReadFixture("circuit_graph.v1.json")), "circuit_graph");
    ExpectRejected(VizDocumentKind, R"({"schema": "pulsatrix.heatmap.v2"})", "v2");
    ExpectRejected(VizDocumentKind, R"({"schema": "other.heatmap.v1"})", "schema");
    ExpectRejected(VizDocumentKind, R"({"schema": "pulsatrix.heatmap"})", "schema");
    ExpectRejected(VizDocumentKind, R"({"kind": "heatmap"})", "schema");
    ExpectRejected(VizDocumentKind, R"([1])", "object");
}

TEST(VizDocumentReadTest, ReaderChecksTheKind) {
    ExpectRejected(ParseHeatmapDocument, ReadFixture("attribution.v1.json"), "pulsatrix.attribution.v1");
}

TEST(VizDocumentReadTest, IgnoresUnknownFields) {
    std::string json = R"({"schema": "pulsatrix.token_relevance.v1", "method": "m", "tokens": ["a"],
                           "relevance": [1], "target": "", "added_later": {"x": [1, 2]}})";
    EXPECT_EQ(ParseTokenRelevanceDocument(json).tokens, (std::vector<std::string>{"a"}));
}

TEST(VizDocumentReadTest, NullNeedsANonFiniteEntry) {
    ExpectRejected(ParseAttributionDocument,
                   R"({"schema": "pulsatrix.attribution.v1", "method": "m", "shape": [2], "values": [1, null], "metadata": {}})",
                   "/values/1");
    ExpectRejected(ParseAttributionDocument,
                   R"({"schema": "pulsatrix.attribution.v1", "method": "m", "shape": [1], "values": [null], "metadata": {},
                       "nonfinite": {"/values/0": "NaN"}})",
                   "/nonfinite");
    // An entry for a number that isn't null: the two disagree, so the document is corrupt.
    ExpectRejected(ParseAttributionDocument,
                   R"({"schema": "pulsatrix.attribution.v1", "method": "m", "shape": [1], "values": [2], "metadata": {},
                       "nonfinite": {"/values/0": "nan"}})",
                   "/values/0");
    // An entry that points at nothing.
    ExpectRejected(ParseAttributionDocument,
                   R"({"schema": "pulsatrix.attribution.v1", "method": "m", "shape": [1], "values": [2], "metadata": {},
                       "nonfinite": {"/values/9": "nan"}})",
                   "/values/9");
}

TEST(VizDocumentReadTest, ErrorsNameTheField) {
    ExpectRejected(ParseAttributionDocument, R"({"schema": "pulsatrix.attribution.v1", "shape": [1], "values": [1], "metadata": {}})",
                   "/method");
    ExpectRejected(ParseAttributionDocument,
                   R"({"schema": "pulsatrix.attribution.v1", "method": "m", "shape": [2, 2], "values": [1], "metadata": {}})",
                   "/values");
    ExpectRejected(ParseAttributionDocument,
                   R"({"schema": "pulsatrix.attribution.v1", "method": "m", "shape": [-1], "values": [], "metadata": {}})",
                   "/shape/0");
    ExpectRejected(ParseAttributionDocument,
                   R"({"schema": "pulsatrix.attribution.v1", "method": "m", "shape": [1], "values": ["1"], "metadata": {}})",
                   "/values/0");
    ExpectRejected(ParseHeatmapDocument,
                   R"({"schema": "pulsatrix.heatmap.v1", "title": "", "rows": 1, "cols": 2, "values": [1, 2],
                       "row_labels": ["a", "b"], "col_labels": []})",
                   "/row_labels");
    ExpectRejected(ParseTokenRelevanceDocument,
                   R"({"schema": "pulsatrix.token_relevance.v1", "method": "m", "tokens": ["a", "b"], "relevance": [1], "target": ""})",
                   "/relevance");
    ExpectRejected(ParseTrainingLogDocument,
                   R"({"schema": "pulsatrix.training_log.v1", "scalars": [{"tag": "a", "steps": [0, 1], "values": [1]}], "histograms": []})",
                   "/scalars/0/values");
    ExpectRejected(ParseTrainingLogDocument,
                   R"({"schema": "pulsatrix.training_log.v1", "scalars": [{"tag": "a", "steps": [3000000000], "values": [1]}], "histograms": []})",
                   "/scalars/0/steps/0");
    ExpectRejected(ParseFeatureDashboardDocument,
                   R"({"schema": "pulsatrix.feature_dashboard.v1", "source": "", "feature_index": 0, "activation_density": 0,
                       "max_activation": 0, "histogram_edges": [0, 1], "histogram_counts": [1, 2], "top_examples": []})",
                   "/histogram_edges");
}

TEST(VizDocumentReadTest, PartialDependenceChecksItsShape) {
    const std::string golden = ReadFixture("partial_dependence.v1.json");
    auto with = [&](const std::string& from, const std::string& to) {
        std::string s = golden;
        s.replace(s.find(from), from.size(), to);
        return s;
    };
    ExpectRejected(ParsePartialDependenceDocument, with("[0, 0.5, 1]", "[0, 1, 0.5]"), "/grid");
    ExpectRejected(ParsePartialDependenceDocument, with("[0.25, 0.5, 1]", "[0.25, 0.5]"), "/partial_dependence");
    ExpectRejected(ParsePartialDependenceDocument, with("\"num_instances\": 2", "\"num_instances\": 3"), "/ice");
    ExpectRejected(ParsePartialDependenceDocument, with("[0.25, 0.75]", "[0.25]"), "/feature_values");
}

TEST(VizDocumentConversionTest, AleIsAPartialDependenceDocumentWithMethodAle) {
    AleResult r;
    r.edges = {0.0f, 1.0f, 3.0f};
    r.effects = {-1.0f, 0.5f, 2.0f};
    r.counts = {4, 2};
    r.feature_values = {0.0f, 0.5f, 1.0f, 2.0f, 3.0f, 0.25f};
    PartialDependenceDocument doc = ToPartialDependenceDocument(r, "x", "y");
    EXPECT_EQ(doc.method, "ale");
    PartialDependenceDocument back = ParsePartialDependenceDocument(ToJson(doc));
    EXPECT_EQ(back.method, "ale");
    ExpectSameFloats(back.partial_dependence, r.effects);
    EXPECT_EQ(back.feature_values.size(), 6u);

    // A document written before the field existed reads as partial dependence.
    std::string old = ReadFixture("partial_dependence.v1.json");
    old.erase(old.find("  \"method\""), std::string("  \"method\": \"partial_dependence\",\n").size());
    EXPECT_EQ(ParsePartialDependenceDocument(old).method, "partial_dependence");
    std::string unknown = ToJson(doc);
    unknown.replace(unknown.find("\"ale\""), 5, "\"pdp\"");
    ExpectRejected(ParsePartialDependenceDocument, unknown, "/method");
    doc.num_instances = 1;
    doc.ice = {1.0f, 2.0f, 3.0f};
    EXPECT_THROW((void)ToJson(doc), std::invalid_argument);
}

TEST(VizDocumentConversionTest, IceRoundTripsThroughAPartialDependenceDocument) {
    IceResult r;
    r.grid = {0.0f, 1.0f, 2.0f};
    r.num_instances = 2;
    r.curves = {1.0f, 2.0f, 4.0f, 0.0f, 0.0f, 1.0f};
    r.feature_values = {0.5f, 1.5f};
    PartialDependenceDocument doc = ToPartialDependenceDocument(r, "x", "y");
    ExpectSameFloats(doc.partial_dependence, {0.5f, 1.0f, 2.5f});
    IceResult back = ToIceResult(ParsePartialDependenceDocument(ToJson(doc)));
    ExpectSameFloats(back.curves, r.curves);
    ExpectSameFloats(back.centered(), r.centered());
    doc.num_instances = 0;
    doc.ice.clear();
    doc.feature_values.clear();
    EXPECT_THROW((void)ToIceResult(doc), std::invalid_argument);

    PartialDependence2D pd{0, 1, 0, {0.0f, 0.5f}, {1.0f, 2.0f, 3.0f}, {1, 2, 3, 4, 5, 6}};
    HeatmapDocument h = ToHeatmapDocument(pd, "pd");
    EXPECT_EQ(h.rows, 3);
    EXPECT_EQ(h.cols, 2);
    EXPECT_EQ(h.row_labels, (std::vector<std::string>{"1", "2", "3"}));
    EXPECT_EQ(h.col_labels, (std::vector<std::string>{"0", "0.5"}));
}

TEST(VizDocumentReadTest, CircuitGraphChecksItsNodes) {
    auto graph = [](const std::string& nodes, const std::string& edges) {
        return R"({"schema": "pulsatrix.circuit_graph.v1", "nodes": )" + nodes + R"(, "edges": )" + edges + "}";
    };
    std::string two = R"([{"id": 0, "op_type": "Linear", "label": null, "ablation_effect": 1},
                          {"id": 1, "op_type": "Linear", "label": null, "ablation_effect": 1}])";
    EXPECT_NO_THROW((void)ParseCircuitGraphDocument(graph(two, R"([{"from": 0, "to": 1, "weight": 1}])")));
    ExpectRejected(ParseCircuitGraphDocument, graph(two, R"([{"from": 0, "to": 5, "weight": 1}])"), "/edges/0/to");
    ExpectRejected(ParseCircuitGraphDocument,
                   graph(R"([{"id": 0, "op_type": "Linear", "label": null, "ablation_effect": 1},
                             {"id": 0, "op_type": "Linear", "label": null, "ablation_effect": 1}])",
                         "[]"),
                   "/nodes/1/id");
    ExpectRejected(ParseCircuitGraphDocument,
                   graph(R"([{"id": 0, "op_type": "Dense", "label": null, "ablation_effect": 1}])", "[]"),
                   "/nodes/0/op_type");
}

// ---- writing: invalid documents are refused instead of written.

TEST(VizDocumentWriteTest, RefusesInconsistentDocuments) {
    AttributionDocument a = GoldenAttribution();
    a.values.pop_back();
    EXPECT_THROW((void)ToJson(a), std::invalid_argument);

    TokenRelevanceDocument t;
    t.tokens = {"a"};
    EXPECT_THROW((void)ToJson(t), std::invalid_argument);
    t.relevance = {1.0f};
    t.tokens = {"\xFF"};  // a raw byte-level token: not UTF-8
    EXPECT_THROW((void)ToJson(t), std::invalid_argument);

    CircuitGraphDocument c = GoldenCircuit();
    c.edges.push_back({0, 7, 1.0f});
    EXPECT_THROW((void)ToJson(c), std::invalid_argument);

    TrainingLogDocument l;
    l.scalars = {{"a", {0}, {1.0}}, {"a", {1}, {2.0}}};
    EXPECT_THROW((void)ToJson(l), std::invalid_argument);
}

TEST(VizDocumentWriteTest, NonFiniteDoublesInATrainingLogRoundTrip) {
    TrainingLogDocument doc;
    doc.scalars = {{"loss", {0, 1}, {std::numeric_limits<double>::infinity(), 1.0}}};
    std::string json = ToJson(doc);
    EXPECT_NE(json.find("\"/scalars/0/values/0\": \"inf\""), std::string::npos) << json;
    EXPECT_EQ(ParseTrainingLogDocument(json).scalars[0].values[0], std::numeric_limits<double>::infinity());
}

}  // namespace
}  // namespace pulsatrix
