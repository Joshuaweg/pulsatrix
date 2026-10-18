// VIZ-9: the series document (pulsatrix.series.v1) and its SVG and Vega-Lite views.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "pulsatrix/viz/document.hpp"
#include "pulsatrix/viz/html.hpp"
#include "pulsatrix/viz/mime_bundle.hpp"
#include "pulsatrix/viz/series.hpp"

namespace pulsatrix {
namespace {

std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string FixturePath(const std::string& name) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/viz/" + name; }

// Regenerate with PULSATRIX_UPDATE_GOLDEN=1 after a deliberate change, and look at the result.
void ExpectGolden(const std::string& path, const std::string& text) {
    const char* update = std::getenv("PULSATRIX_UPDATE_GOLDEN");
    if (update != nullptr && std::string(update) == "1") {
        std::ofstream(path, std::ios::binary) << text;
        return;
    }
    const std::string golden = ReadFile(path);
    ASSERT_FALSE(golden.empty()) << "missing golden file " << path;
    EXPECT_EQ(text, golden) << path << " changed";
}

size_t Count(const std::string& haystack, const std::string& needle) {
    size_t n = 0;
    for (size_t at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + 1)) ++n;
    return n;
}

// Two panels, two series; one series has intervals. Linear axes and short decimals keep the
// golden SVG free of libm results, so it matches byte for byte on every platform.
SeriesDocument Example() {
    SeriesDocument doc;
    doc.title = "Catastrophe rate by penalty";
    doc.x_label = "penalty";
    doc.y_label = "catastrophe rate";
    doc.interval = "95% interval over 5 seeds";
    SeriesPanel a;
    a.title = "world A";
    a.series.push_back({"standard starts", {1, 2, 4, 8}, {0.12, 0.1, 0.05, 0.02}, {0.1, 0.08, 0.04, 0.01}, {0.14, 0.12, 0.06, 0.03}});
    a.series.push_back({"exploring starts", {1, 2, 4, 8}, {0.16, 0.11, 0.06, 0.02}, {}, {}});
    SeriesPanel b;
    b.title = "world B";
    b.series.push_back({"standard starts", {1, 2, 4, 8}, {0.01, 0.01, 0.005, 0.0025}, {0.0, 0.005, 0.0, 0.0}, {0.02, 0.015, 0.01, 0.005}});
    doc.panels = {a, b};
    return doc;
}

SeriesDocument LogExample() {
    SeriesDocument doc;
    doc.x_label = "lambda";
    doc.y_label = "rate";
    doc.x_scale = AxisScale::Log;
    doc.y_scale = AxisScale::Log;
    SeriesPanel p;
    p.series.push_back({"agent", {1, 10, 100, 1000}, {0.1, 0.05, 0.01, 0.002}, {0.08, 0.0, 0.005, 0.001}, {0.12, 0.07, 0.02, 0.004}});
    doc.panels = {p};
    return doc;
}

TEST(SeriesDocumentTest, RoundTripsAndMatchesTheFixture) {
    const SeriesDocument doc = Example();
    const std::string json = ToJson(doc);
    ExpectGolden(FixturePath("series.v1.json"), json);
    EXPECT_EQ(VizDocumentKind(json), "series");
    const SeriesDocument back = ParseSeriesDocument(ReadFile(FixturePath("series.v1.json")));
    EXPECT_EQ(ToJson(back), json);
    ASSERT_EQ(back.panels.size(), 2u);
    EXPECT_EQ(back.panels[0].series[1].name, "exploring starts");
    EXPECT_FALSE(back.panels[0].series[1].has_intervals());
    EXPECT_EQ(back.panels[1].series[0].hi, (std::vector<double>{0.02, 0.015, 0.01, 0.005}));
    EXPECT_EQ(ParseSeriesDocument(ToJson(LogExample())).x_scale, AxisScale::Log);
}

TEST(SeriesDocumentTest, NonFiniteValuesUseTheNonfiniteMember) {
    SeriesDocument doc = LogExample();
    doc.panels[0].series[0].y[1] = std::numeric_limits<double>::quiet_NaN();
    doc.panels[0].series[0].hi[3] = std::numeric_limits<double>::infinity();
    const std::string json = ToJson(doc);
    EXPECT_NE(json.find("\"/panels/0/series/0/y/1\": \"nan\""), std::string::npos) << json;
    const SeriesDocument back = ParseSeriesDocument(json);
    EXPECT_TRUE(std::isnan(back.panels[0].series[0].y[1]));
    EXPECT_TRUE(std::isinf(back.panels[0].series[0].hi[3]));
    EXPECT_THROW((void)RenderSeriesSvg(doc), std::invalid_argument);  // a chart can't place them
}

TEST(SeriesDocumentTest, RejectsBrokenDocuments) {
    SeriesDocument none;
    EXPECT_THROW((void)ToJson(none), std::invalid_argument);
    SeriesDocument empty_panel = Example();
    empty_panel.panels[1].series.clear();
    EXPECT_THROW((void)ToJson(empty_panel), std::invalid_argument);
    SeriesDocument twice = Example();
    twice.panels[0].series[1].name = "standard starts";
    EXPECT_THROW((void)ToJson(twice), std::invalid_argument);
    SeriesDocument short_y = Example();
    short_y.panels[0].series[0].y.pop_back();
    EXPECT_THROW((void)ToJson(short_y), std::invalid_argument);
    SeriesDocument short_lo = Example();
    short_lo.panels[0].series[0].lo.pop_back();
    EXPECT_THROW((void)ToJson(short_lo), std::invalid_argument);
    SeriesDocument inverted = Example();
    inverted.panels[0].series[0].lo[0] = 0.5;
    EXPECT_THROW((void)ToJson(inverted), std::invalid_argument);
    SeriesDocument unnamed = Example();
    unnamed.panels[0].series[0].name.clear();
    EXPECT_THROW((void)ToJson(unnamed), std::invalid_argument);

    std::string json = ToJson(Example());
    std::string bad_scale = json;
    bad_scale.replace(bad_scale.find("\"x_scale\": \"linear\""), 19, "\"x_scale\": \"cubic\"");
    try {
        (void)ParseSeriesDocument(bad_scale);
        FAIL() << "a bad scale was accepted";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("/x_scale"), std::string::npos) << e.what();
    }
    std::string lo_only = json;
    lo_only.replace(lo_only.find("\"hi\":"), 5, "\"hx\":");
    EXPECT_THROW((void)ParseSeriesDocument(lo_only), std::invalid_argument);
    std::string inverted_json = ToJson(Example());
    const size_t at = inverted_json.find("\"lo\": [0.1,");
    ASSERT_NE(at, std::string::npos);
    inverted_json.replace(at, 11, "\"lo\": [0.9,");
    try {
        (void)ParseSeriesDocument(inverted_json);
        FAIL() << "lo > hi was accepted";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("/panels/0/series/0/lo/0"), std::string::npos) << e.what();
    }
}

TEST(SeriesViewTest, MatchesTheGoldenFile) { ExpectGolden(FixturePath("svg/series.svg"), RenderSeriesSvg(Example())); }

TEST(SeriesViewTest, DrawsEveryMark) {
    const std::string svg = RenderSeriesSvg(Example());
    EXPECT_EQ(Count(svg, "<polyline class=\"series\""), 3u);
    EXPECT_EQ(Count(svg, "<circle class=\"point\""), 12u);
    EXPECT_EQ(Count(svg, "<circle class=\"legend-marker\""), 2u);  // one legend for both panels
    EXPECT_EQ(Count(svg, "<line class=\"whisker\""), 8u);
    EXPECT_EQ(Count(svg, "<line class=\"whisker-cap\""), 16u);
    EXPECT_NE(svg.find("world B"), std::string::npos);
    EXPECT_NE(svg.find("Whiskers: 95% interval over 5 seeds"), std::string::npos);
    EXPECT_NE(svg.find("<title>standard starts: penalty 4, catastrophe rate 0.05 [0.04, 0.06]</title>"), std::string::npos);
    // A series keeps its color in every panel.
    EXPECT_EQ(Count(svg, "stroke=\"#4e79a7\" stroke-width=\"2\" stroke-linejoin"), 2u);
}

TEST(SeriesViewTest, LogAxes) {
    const std::string svg = RenderSeriesSvg(LogExample());
    for (const char* label : {">1<", ">10<", ">100<", ">1000<"}) EXPECT_NE(svg.find(label), std::string::npos) << label;
    // The lower bound at 0 can't go on a log axis: its whisker runs to the floor, with no lower cap.
    EXPECT_EQ(Count(svg, "<line class=\"whisker\""), 4u);
    EXPECT_EQ(Count(svg, "<line class=\"whisker-cap\""), 7u);

    SeriesDocument zero = LogExample();
    zero.panels[0].series[0].y[0] = 0.0;
    EXPECT_THROW((void)RenderSeriesSvg(zero), std::invalid_argument);
    SeriesDocument negative_x = LogExample();
    negative_x.panels[0].series[0].x[0] = -1.0;
    EXPECT_THROW((void)RenderSeriesSvg(negative_x), std::invalid_argument);
    SvgOptions narrow;
    narrow.width = 100;
    EXPECT_THROW((void)RenderSeriesSvg(LogExample(), {}, narrow), std::invalid_argument);
}

TEST(SeriesViewTest, LongYLabelsFitTheirPanel) {
    SeriesDocument doc = Example();
    doc.y_label = std::string(200, 'y');
    const std::string svg = RenderSeriesSvg(doc);
    const size_t label = svg.find("rotate(-90");
    ASSERT_NE(label, std::string::npos);
    const std::string text = svg.substr(svg.find('>', label) + 1, svg.find("</text>", label) - svg.find('>', label) - 1);
    EXPECT_LT(text.size(), 200u);
    EXPECT_EQ(text.substr(text.size() - 3), "\xE2\x80\xA6");  // cut, with an ellipsis
    EXPECT_NE(svg.find(std::string(200, 'y')), std::string::npos);  // tooltips keep the full label
}

TEST(SeriesViewTest, SharedYGivesEveryPanelTheSameTicks) {
    SeriesViewOptions shared;
    shared.shared_y = true;
    const std::string own = RenderSeriesSvg(Example());
    const std::string same = RenderSeriesSvg(Example(), shared);
    // World B's values are small: on its own axis it gets finer tick labels than world A's.
    EXPECT_EQ(Count(own, ">0.16<"), 1u);
    EXPECT_EQ(Count(same, ">0.16<"), 2u);
}

TEST(SeriesHtmlTest, OneLayeredChartPerPanel) {
    const JsonValue spec = ToVegaLiteSeries(Example());
    const JsonValue::Array& charts = spec.find("vconcat")->as_array();
    ASSERT_EQ(charts.size(), 2u);
    const JsonValue::Array& layers = charts[0].find("layer")->as_array();
    ASSERT_EQ(layers.size(), 3u);  // interval rules, lines, points
    EXPECT_EQ(layers[0].find("mark")->find("type")->as_string(), "rule");
    EXPECT_EQ(spec.find("resolve")->find("scale")->find("y")->as_string(), "independent");
    const JsonValue& domain = *layers[1].find("encoding")->find("color")->find("scale")->find("domain");
    ASSERT_EQ(domain.as_array().size(), 2u);
    EXPECT_EQ(domain.as_array()[0].as_string(), "standard starts");
    EXPECT_EQ(domain.as_array()[1].as_string(), "exploring starts");

    const JsonValue log_spec = ToVegaLiteSeries(LogExample());
    const JsonValue& log_line = log_spec.find("vconcat")->as_array()[0].find("layer")->as_array()[1];
    EXPECT_EQ(log_line.find("encoding")->find("x")->find("scale")->find("type")->as_string(), "log");
    EXPECT_EQ(log_line.find("encoding")->find("y")->find("scale")->find("type")->as_string(), "log");
    SeriesViewOptions shared;
    shared.shared_y = true;
    EXPECT_EQ(ToVegaLiteSeries(Example(), shared).find("resolve"), nullptr);

    const std::string page = RenderSeriesHtml(Example());
    EXPECT_NE(page.find("vegaEmbed"), std::string::npos);
    SeriesDocument zero = LogExample();
    zero.panels[0].series[0].y[0] = 0.0;
    EXPECT_THROW((void)ToVegaLiteSeries(zero), std::invalid_argument);
}

TEST(SeriesHtmlTest, NotebookBundleHasSvgAndVegaLite) {
    const MimeBundle bundle = mime_bundle_repr(Example());
    EXPECT_NE(bundle.find("image/svg+xml"), nullptr);
    EXPECT_NE(bundle.find(kVegaLiteMimeType), nullptr);
    ASSERT_NE(bundle.find("text/plain"), nullptr);
    EXPECT_EQ(bundle.find("text/plain")->as_string(), "Catastrophe rate by penalty");
}

}  // namespace
}  // namespace pulsatrix
