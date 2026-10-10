// NB-1: mime_bundle_repr() for notebooks. The conversion to nlohmann::json and the lookup are
// tested the way xeus-cpp 0.10 does them: XeusDisplay() below is xcpp::display()'s body
// (include/xcpp/xdisplay.hpp) and xcpp::mime_bundle_repr its generic fallback
// (include/xcpp/xmime.hpp), so these tests are the roadmap's falsifier for NB-1.
#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/viz/mime_bundle.hpp"

namespace xcpp {
// xeus-cpp's default: anything printable becomes text/plain.
template <class T>
nlohmann::json mime_bundle_repr(const T& value) {
    auto bundle = nlohmann::json::object();
    std::ostringstream oss;
    oss << value;
    bundle["text/plain"] = oss.str();
    return bundle;
}

// What xcpp::display(t) hands to the kernel's display_data().
template <class T>
nlohmann::json XeusDisplay(const T& t) {
    using ::xcpp::mime_bundle_repr;
    nlohmann::json data = mime_bundle_repr(t);
    return data;
}
}  // namespace xcpp

namespace pulsatrix {
namespace {

using nlohmann::json;

Tensor Values(CPUBackend& cpu, Shape shape, unsigned seed) {
    std::vector<float> v(static_cast<size_t>(shape.numel()));
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(((i * 7919 + seed * 31) % 201)) / 100.0f - 1.0f;
    return Tensor(std::move(shape), &cpu, v);
}

// Writes each Vega-Lite spec when PULSATRIX_DUMP_VEGA_LITE names a directory, so they can be
// checked against Vega-Lite 5's JSON schema (docs/visualization/notebooks.md).
void Dump(const json& j, const std::string& name) {
    if (const char* dir = std::getenv("PULSATRIX_DUMP_VEGA_LITE"); dir != nullptr && j.contains(kVegaLiteMimeType)) {
        std::ofstream(std::string(dir) + "/" + name + ".json") << j[kVegaLiteMimeType].dump(1);
    }
}

TEST(MimeBundle, XeusFindsTheOverloadsAndConvertsThem) {
    CPUBackend cpu;
    const json t = xcpp::XeusDisplay(Values(cpu, Shape({2, 3}), 1));
    ASSERT_TRUE(t.is_object());
    EXPECT_TRUE(t["text/plain"].is_string());
    EXPECT_TRUE(t["text/html"].is_string());
    EXPECT_NE(t["text/plain"].get<std::string>().find("Tensor (2, 3) float32 on cpu, 6 values"), std::string::npos)
        << t["text/plain"];
    // A type pulsatrix doesn't cover still gets xeus's own text/plain.
    EXPECT_EQ(xcpp::XeusDisplay(42), json({{"text/plain", "42"}}));
}

TEST(MimeBundle, VegaLiteIsAnObjectWithIntegersKept) {
    HeatmapDocument doc;
    doc.title = "grid";
    doc.rows = 2;
    doc.cols = 3;
    doc.values = {0.5f, -1.0f, 2.0f, 0.0f, 1.25f, -0.25f};
    const json j = xcpp::XeusDisplay(doc);
    Dump(j, "heatmap");
    ASSERT_TRUE(j[kVegaLiteMimeType].is_object());
    EXPECT_EQ(j[kVegaLiteMimeType]["$schema"], "https://vega.github.io/schema/vega-lite/v5.json");
    EXPECT_TRUE(j[kVegaLiteMimeType]["width"].is_number_integer());
    EXPECT_NE(j["image/svg+xml"].get<std::string>().find("<svg"), std::string::npos);
    // Richest first in the bundle itself (nlohmann::json sorts its keys).
    const MimeBundle b = mime_bundle_repr(doc);
    EXPECT_EQ(b.entries.front().first, kVegaLiteMimeType);
    EXPECT_EQ(b.entries.back().first, "text/plain");
}

TEST(MimeBundle, TensorSummary) {
    CPUBackend cpu;
    const MimeBundle b = mime_bundle_repr(Tensor(Shape({2, 10}), &cpu, std::vector<float>{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10}));
    const std::string text = b.find("text/plain")->as_string();
    EXPECT_NE(text.find("min -10, max 10, mean 0, std 6.205"), std::string::npos) << text;
    EXPECT_NE(text.find("  1 2 3 4 5 6 7 8 ..."), std::string::npos) << text;  // 8 of 10 columns
    const std::string html = b.find("text/html")->as_string();
    EXPECT_EQ(html.find("<style"), std::string::npos);  // inline styles only: nothing leaks into the notebook
    EXPECT_EQ(html.find("<script"), std::string::npos);
}

TEST(MimeBundle, ImageAttributionsAreHeatmapsAndLargeOnesPng) {
    CPUBackend cpu;
    const MimeBundle small = mime_bundle_repr(Attribution{"lrp", Values(cpu, Shape({1, 3, 8, 8}), 2), {}});
    EXPECT_NE(small.find(kVegaLiteMimeType), nullptr);
    EXPECT_NE(small.find("image/svg+xml"), nullptr);
    EXPECT_NE(small.find("text/plain")->as_string().find("lrp, channels summed (8 x 8)"), std::string::npos)
        << small.find("text/plain")->as_string();

    const MimeBundle large = mime_bundle_repr(Attribution{"saliency", Values(cpu, Shape({2, 1, 80, 96}), 3), {}});
    ASSERT_NE(large.find("image/png"), nullptr);
    EXPECT_EQ(large.find(kVegaLiteMimeType), nullptr);
    EXPECT_EQ(large.find("image/png")->as_string().rfind("iVBORw0KGgo", 0), 0u);  // the PNG signature
    EXPECT_NE(large.find("text/plain")->as_string().find("example 1 of 2 (80 x 96)"), std::string::npos);
}

TEST(MimeBundle, FeatureAttributionsAreBarCharts) {
    CPUBackend cpu;
    const json j = xcpp::XeusDisplay(Attribution{"kernel_shap", Values(cpu, Shape({2, 5}), 4), {}});
    Dump(j, "bar_chart");
    ASSERT_TRUE(j.contains(kVegaLiteMimeType));
    EXPECT_NE(j["image/svg+xml"].get<std::string>().find("example 1 of 2"), std::string::npos);
    EXPECT_NE(j["text/plain"].get<std::string>().find("largest |value|"), std::string::npos);
}

TEST(MimeBundle, TokenRelevance) {
    TokenRelevanceDocument doc;
    doc.method = "attnlrp";
    doc.tokens = {"The", " cat", " sat"};
    doc.relevance = {0.1f, 0.7f, -0.2f};
    doc.target = " down";
    const json j = xcpp::XeusDisplay(doc);
    EXPECT_TRUE(j.contains("image/svg+xml"));
    EXPECT_FALSE(j.contains(kVegaLiteMimeType));
    EXPECT_NE(j["text/plain"].get<std::string>().find(" cat  0.7"), std::string::npos) << j["text/plain"];
}

TEST(MimeBundle, CircuitGraphLaysNodesOutByDepth) {
    CircuitGraphDocument doc;
    doc.nodes = {{0, "Linear", std::string("in"), 0.1f}, {1, "Relu", std::nullopt, -0.4f}, {2, "Linear", std::nullopt, 0.9f},
                 {3, "Linear", std::string("skip"), 0.0f}};
    doc.edges = {{0, 1, 0.5f}, {1, 2, -1.5f}, {0, 3, 0.25f}, {3, 2, 1.0f}};
    const json j = xcpp::XeusDisplay(doc);
    Dump(j, "circuit_graph");
    const json& layers = j[kVegaLiteMimeType]["layer"];
    ASSERT_EQ(layers.size(), 3u);
    std::vector<int> depth(4);
    for (const json& node : layers[1]["data"]["values"]) depth[node["id"].get<int>()] = node["x"].get<int>();
    EXPECT_EQ(depth, (std::vector<int>{0, 1, 2, 1}));
    EXPECT_EQ(layers[0]["data"]["values"].size(), 4u);
    doc.edges.push_back({2, 0, 1.0f});
    EXPECT_THROW((void)mime_bundle_repr(doc), std::invalid_argument);
    doc.edges.back() = {2, 9, 1.0f};
    EXPECT_THROW((void)mime_bundle_repr(doc), std::invalid_argument);
}

std::string ReadFixture(const std::string& name) {
    std::ifstream in(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/viz/" + name, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Every document kind with a figure gets a bundle: text/plain, its SVG, and its Vega-Lite spec when
// it has one.
TEST(MimeBundle, EveryDocumentKind) {
    struct Case {
        const char* name;
        json bundle;
        bool vega_lite;
        bool svg;
    };
    const std::vector<Case> cases = {
        {"attribution", xcpp::XeusDisplay(ParseAttributionDocument(ReadFixture("attribution_features.v1.json"))), true, true},
        {"heatmap_fixture", xcpp::XeusDisplay(ParseHeatmapDocument(ReadFixture("heatmap.v1.json"))), true, true},
        {"token_relevance", xcpp::XeusDisplay(ParseTokenRelevanceDocument(ReadFixture("token_relevance.v1.json"))), false, true},
        {"circuit_graph_fixture", xcpp::XeusDisplay(ParseCircuitGraphDocument(ReadFixture("circuit_graph_finite.v1.json"))), true, false},
        {"training_log", xcpp::XeusDisplay(ParseTrainingLogDocument(ReadFixture("training_log.v1.json"))), true, false},
        {"partial_dependence", xcpp::XeusDisplay(ParsePartialDependenceDocument(ReadFixture("partial_dependence.v1.json"))), true, true},
        {"sensitivity", xcpp::XeusDisplay(ParseSensitivityDocument(ReadFixture("sensitivity.v1.json"))), true, true},
        {"counterfactual", xcpp::XeusDisplay(ParseCounterfactualDocument(ReadFixture("counterfactual.v1.json"))), true, true},
        {"morris", xcpp::XeusDisplay(ParseMorrisDocument(ReadFixture("morris.v1.json"))), true, true},
        {"sobol", xcpp::XeusDisplay(ParseSobolDocument(ReadFixture("sobol.v1.json"))), true, true},
        {"mutation_map", xcpp::XeusDisplay(ParseMutationMapDocument(ReadFixture("mutation_map.v1.json"))), false, true},
        {"sequence_logo", xcpp::XeusDisplay(ParseSequenceLogoDocument(ReadFixture("sequence_logo.v1.json"))), false, true},
        {"contact_map", xcpp::XeusDisplay(ParseContactMapDocument(ReadFixture("contact_map.v1.json"))), false, true},
        {"residue_tracks", xcpp::XeusDisplay(ParseResidueTracksDocument(ReadFixture("residue_tracks.v1.json"))), false, true},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        Dump(c.bundle, c.name);
        EXPECT_TRUE(c.bundle["text/plain"].is_string());
        EXPECT_EQ(c.bundle.contains(kVegaLiteMimeType), c.vega_lite);
        EXPECT_EQ(c.bundle.contains("image/svg+xml"), c.svg);
        if (c.vega_lite) EXPECT_TRUE(c.bundle[kVegaLiteMimeType].is_object());
    }
}

TEST(MimeBundle, Base64MatchesRfc4648) {
    auto b64 = [](const std::string& s) { return Base64Encode(std::vector<uint8_t>(s.begin(), s.end())); };
    EXPECT_EQ(b64(""), "");
    EXPECT_EQ(b64("f"), "Zg==");
    EXPECT_EQ(b64("fo"), "Zm8=");
    EXPECT_EQ(b64("foo"), "Zm9v");
    EXPECT_EQ(b64("foob"), "Zm9vYg==");
    EXPECT_EQ(b64("fooba"), "Zm9vYmE=");
    EXPECT_EQ(b64("foobar"), "Zm9vYmFy");
}

TEST(MimeBundle, AddRefusesDuplicates) {
    MimeBundle b;
    b.add("text/plain", "a");
    EXPECT_THROW(b.add("text/plain", "b"), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
