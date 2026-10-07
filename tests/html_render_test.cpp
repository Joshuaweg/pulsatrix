#include "pulsatrix/viz/html.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsatrix {
namespace {

std::string ReadFixture(const std::string& name) {
    std::ifstream in(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/viz/" + name, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

const JsonValue& At(const JsonValue& v, std::string_view key) {
    const JsonValue* found = v.find(key);
    if (found == nullptr) throw std::runtime_error("no member " + std::string(key));
    return *found;
}

/** @brief The spec a page embeds, read back the way the page's script reads it. */
JsonValue EmbeddedSpec(const std::string& html) {
    const std::string open = "<script type=\"application/json\" id=\"spec\">";
    const size_t begin = html.find(open);
    if (begin == std::string::npos) throw std::runtime_error("no embedded spec");
    const size_t start = begin + open.size();
    return ParseJson(html.substr(start, html.find("</script>", start) - start));
}

size_t Count(const std::string& haystack, const std::string& needle) {
    size_t n = 0;
    for (size_t at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + 1)) ++n;
    return n;
}

TEST(HtmlRender, BarChartKeepsTheTopFeaturesInOrder) {
    const AttributionDocument doc = ParseAttributionDocument(ReadFixture("attribution_features.v1.json"));
    const JsonValue spec = ToVegaLiteBarChart(doc, 3);
    EXPECT_EQ(At(spec, "$schema").as_string(), "https://vega.github.io/schema/vega-lite/v6.json");
    const auto& values = At(At(spec, "data"), "values").as_array();
    ASSERT_EQ(values.size(), 3u);
    EXPECT_EQ(At(values[0], "feature").as_string(), "age");  // the largest |attribution|
    for (size_t i = 1; i < values.size(); ++i) {
        EXPECT_GE(std::abs(At(values[i - 1], "value").as_double()), std::abs(At(values[i], "value").as_double()));
    }
    EXPECT_TRUE(At(At(spec, "encoding"), "tooltip").as_array().size() > 0);
}

TEST(HtmlRender, PageEmbedsTheSpecItDraws) {
    const AttributionDocument doc = ParseAttributionDocument(ReadFixture("attribution_features.v1.json"));
    const std::string html = RenderBarChartHtml(doc, 4);
    EXPECT_EQ(html.rfind("<!doctype html>", 0), 0u);
    EXPECT_EQ(WriteJson(EmbeddedSpec(html)), WriteJson(ToVegaLiteBarChart(doc, 4)));
    EXPECT_NE(html.find("vegaEmbed('#vis'"), std::string::npos);
}

TEST(HtmlRender, CdnScriptsArePinnedAndIntegrityChecked) {
    const std::string html = RenderBarChartHtml(ParseAttributionDocument(ReadFixture("attribution_features.v1.json")));
    EXPECT_NE(html.find(std::string("vega@") + kVegaVersion + "/build/vega.min.js"), std::string::npos);
    EXPECT_NE(html.find(std::string("vega-lite@") + kVegaLiteVersion + "/build/vega-lite.min.js"), std::string::npos);
    EXPECT_NE(html.find(std::string("vega-embed@") + kVegaEmbedVersion + "/build/vega-embed.min.js"), std::string::npos);
    EXPECT_EQ(Count(html, "integrity=\"sha384-"), 3u);
    EXPECT_EQ(Count(html, "crossorigin=\"anonymous\""), 3u);
}

TEST(HtmlRender, LabelsCantCloseTheScriptElement) {
    AttributionDocument doc = ParseAttributionDocument(ReadFixture("attribution_features.v1.json"));
    doc.metadata["feature_names"] = "</script><b>x,b,c,d,e,f";
    const std::string html = RenderBarChartHtml(doc);
    EXPECT_EQ(html.find("</script><b>"), std::string::npos);
    bool found = false;
    const JsonValue spec = EmbeddedSpec(html);
    for (const auto& v : At(At(spec, "data"), "values").as_array()) {
        found = found || At(v, "feature").as_string() == "</script><b>x";
    }
    EXPECT_TRUE(found) << "the label survives intact once the JSON is parsed";
}

TEST(HtmlRender, InlineScriptsAreCopiedInAndEscaped) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "pulsatrix_html_inline_test";
    std::filesystem::create_directories(dir);
    for (const char* name : {"vega.min.js", "vega-lite.min.js", "vega-embed.min.js"}) {
        std::ofstream(dir / name) << "var lib = \"" << name << " </script> end\";";
    }
    HtmlOptions options;
    options.scripts = HtmlScripts::Inline;
    options.script_dir = dir.string();
    const std::string html = RenderBarChartHtml(ParseAttributionDocument(ReadFixture("attribution_features.v1.json")), 10, options);
    EXPECT_EQ(html.find("cdn.jsdelivr.net"), std::string::npos);
    EXPECT_EQ(Count(html, "<\\/script> end"), 3u);
    EXPECT_NE(html.find("vega-lite.min.js <\\/script>"), std::string::npos);
    std::filesystem::remove_all(dir);

    options.script_dir = (dir / "missing").string();
    EXPECT_THROW((void)RenderBarChartHtml(ParseAttributionDocument(ReadFixture("attribution_features.v1.json")), 10, options),
                 std::runtime_error);
    options.script_dir.clear();
    EXPECT_THROW((void)RenderBarChartHtml(ParseAttributionDocument(ReadFixture("attribution_features.v1.json")), 10, options),
                 std::invalid_argument);
}

TEST(HtmlRender, WaterfallRunsFromBaselineToPrediction) {
    const AttributionDocument doc = ParseAttributionDocument(ReadFixture("attribution_features.v1.json"));
    const JsonValue spec = ToVegaLiteWaterfall(doc, 0.5f);
    const auto& values = At(At(spec, "data"), "values").as_array();
    ASSERT_EQ(values.size(), doc.values.size() + 2);
    EXPECT_EQ(At(values.front(), "step").as_string(), "baseline");
    EXPECT_EQ(At(values.back(), "step").as_string(), "prediction");
    double total = 0.5;
    for (float v : doc.values) total += v;
    EXPECT_NEAR(At(values.back(), "top").as_double(), total, 1e-5);
    EXPECT_THROW((void)ToVegaLiteWaterfall(doc, std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);
}

TEST(HtmlRender, HeatmapCellsAreRowMajorAndNonFiniteCellsAreNull) {
    HeatmapDocument doc = ParseHeatmapDocument(ReadFixture("heatmap.v1.json"));
    doc.values[1] = std::numeric_limits<float>::infinity();
    const JsonValue spec = ToVegaLiteHeatmap(doc);
    const auto& values = At(At(spec, "data"), "values").as_array();
    ASSERT_EQ(values.size(), static_cast<size_t>(doc.rows * doc.cols));
    EXPECT_EQ(At(values[1], "r").as_int64(), 0);
    EXPECT_EQ(At(values[1], "c").as_int64(), 1);
    EXPECT_TRUE(At(values[1], "v").is_null());
    EXPECT_NE(WriteJson(spec).find("datum.v === null"), std::string::npos);
}

TEST(HtmlRender, TrainingLogSkipsADivergedLoss) {
    TrainingLogDocument doc = ParseTrainingLogDocument(ReadFixture("training_log.v1.json"));
    doc.scalars[0].values[1] = std::numeric_limits<double>::quiet_NaN();
    const JsonValue spec = ToVegaLiteTrainingLog(doc);
    EXPECT_EQ(At(At(At(spec, "vconcat").as_array()[0], "data"), "values").as_array().size(), doc.scalars[0].steps.size() - 1);
}

TEST(HtmlRender, EveryContinuousChartZooms) {
    const auto zooms = [](const JsonValue& spec) { return WriteJson(spec).find("\"bind\": \"scales\"") != std::string::npos; };
    EXPECT_TRUE(zooms(ToVegaLitePartialDependence(ParsePartialDependenceDocument(ReadFixture("partial_dependence.v1.json")))));
    EXPECT_TRUE(zooms(ToVegaLiteMorris(ParseMorrisDocument(ReadFixture("morris.v1.json")))));
    EXPECT_TRUE(zooms(ToVegaLiteTrainingLog(ParseTrainingLogDocument(ReadFixture("training_log.v1.json")))));
    const AttributionDocument a = ParseAttributionDocument(ReadFixture("attribution.v1.json"));
    EXPECT_TRUE(zooms(ToVegaLiteBeeswarm({a, a}, {0, 1})));
}

TEST(HtmlRender, CenteredIceStartsAtZero) {
    const PartialDependenceDocument doc = ParsePartialDependenceDocument(ReadFixture("partial_dependence.v1.json"));
    PartialDependenceSvgOptions pd;
    pd.style = IceStyle::Centered;
    const JsonValue spec = ToVegaLitePartialDependence(doc, pd);
    for (const auto& layer : At(spec, "layer").as_array()) {
        const JsonValue* data = layer.find("data");
        if (data == nullptr) continue;
        const auto& values = At(*data, "values").as_array();
        if (values.empty() || values[0].find("instance") == nullptr) continue;
        EXPECT_NEAR(At(values[0], "y").as_double(), 0.0, 1e-6);  // each curve's first point
    }
}

TEST(HtmlRender, SensitivityViewsKeepTheirRankings) {
    const JsonValue sobol_spec = ToVegaLiteSobol(ParseSobolDocument(ReadFixture("sobol.v1.json")), 2);
    const auto& sobol = At(At(sobol_spec, "data"), "values").as_array();
    EXPECT_EQ(sobol.size(), 4u);  // two features, two indices each
    EXPECT_EQ(At(sobol[0], "feature").as_string(), "rainfall");
    const JsonValue tornado_spec = ToVegaLiteTornado(ParseSensitivityDocument(ReadFixture("sensitivity.v1.json")), 1);
    const auto& tornado = At(At(tornado_spec, "data"), "values").as_array();
    EXPECT_EQ(tornado.size(), 2u);  // one feature, low and high
    EXPECT_EQ(At(tornado[0], "feature").as_string(), "age");
}

TEST(HtmlRender, CounterfactualShowsOnlyChangedFeatures) {
    const CounterfactualDocument doc = ParseCounterfactualDocument(ReadFixture("counterfactual.v1.json"));
    const JsonValue spec = ToVegaLiteCounterfactual(doc);
    size_t changed = 0;
    for (const auto& f : doc.features) changed += f.counterfactual != f.original;
    EXPECT_EQ(At(At(spec, "data"), "values").as_array().size(), changed);
    EXPECT_NO_THROW((void)ToVegaLiteCounterfactualSet({doc, doc}));
    CounterfactualDocument other = doc;
    other.features[0].original += 1;
    EXPECT_THROW((void)ToVegaLiteCounterfactualSet({doc, other}), std::invalid_argument);
}

TEST(HtmlRender, FeatureDashboardHasStatsChartAndExamples) {
    const std::string html = RenderFeatureDashboardHtml(ParseFeatureDashboardDocument(ReadFixture("feature_dashboard.v1.json")));
    EXPECT_NE(html.find("feature 42"), std::string::npos);
    EXPECT_NE(html.find("Top examples"), std::string::npos);
    EXPECT_NE(html.find(">Paris</span>"), std::string::npos);
    EXPECT_LT(html.find("active on"), html.find("<div id=\"vis\">"));
}

TEST(HtmlRender, TokenRelevanceIsPlainHtmlTheBrowserShapes) {
    TokenRelevanceDocument doc = ParseTokenRelevanceDocument(ReadFixture("token_relevance.v1.json"));
    doc.tokens[0] = "\xD8\xA7\xD9\x84\xD8\xB9\xD8\xB1\xD8\xA8\xD9\x8A\xD8\xA9 <&>";  // Arabic, and markup to escape
    const std::string html = RenderTokenRelevanceHtml(doc);
    EXPECT_EQ(html.find("<script"), std::string::npos);
    EXPECT_NE(html.find("dir=\"auto\""), std::string::npos);
    EXPECT_NE(html.find("\xD8\xA7\xD9\x84\xD8\xB9\xD8\xB1\xD8\xA8\xD9\x8A\xD8\xA9 &lt;&amp;&gt;"), std::string::npos);
    EXPECT_EQ(Count(html, "<span class=\"s\""), static_cast<size_t>(std::count(doc.scored.begin(), doc.scored.end(), true)) +
                                                    (doc.scored.empty() ? doc.tokens.size() : 0));
}

TEST(HtmlRender, RejectsWhatTheSvgViewsReject) {
    AttributionDocument nonfinite = ParseAttributionDocument(ReadFixture("attribution_nonfinite.v1.json"));
    EXPECT_THROW((void)ToVegaLiteBarChart(nonfinite), std::invalid_argument);
    const AttributionDocument doc = ParseAttributionDocument(ReadFixture("attribution_features.v1.json"));
    EXPECT_THROW((void)ToVegaLiteBarChart(doc, 0), std::invalid_argument);
    HtmlOptions narrow;
    narrow.width = 10;
    EXPECT_THROW((void)ToVegaLiteBarChart(doc, 5, narrow), std::invalid_argument);
    EXPECT_THROW((void)ToVegaLiteBeeswarm({doc}, {99}), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
